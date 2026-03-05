"""Single CLI entry point.

Usage:
    python -m graphids.pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_ch
    python -m graphids.pipeline.cli curriculum  --model gat --scale small --auxiliaries kd_standard --dataset hcrl_sa
    python -m graphids.pipeline.cli fusion      --config path/to/config.json
"""

from __future__ import annotations

import torch.multiprocessing as mp

# Must be called before any CUDA or multiprocessing usage.
# Prevents "Cannot re-initialize CUDA in forked subprocess" errors
# when DataLoader workers collate tensors after CUDA has been initialized
# in the main process (e.g. by _score_difficulty in the curriculum stage).
mp.set_start_method("spawn", force=True)

import argparse
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from graphids.config import STAGES, PipelineConfig, config_path, run_id, stage_dir
from graphids.config.resolver import resolve

from .validate import validate

_ON_COMPUTE_NODE = bool(os.environ.get("SLURM_JOB_ID"))


def _parse_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="KD-GAT training pipeline",
    )
    p.add_argument(
        "stage",
        choices=list(STAGES.keys()) + ["flow", "tune", "sweep-pipeline"],
        help="Training stage, 'flow' for Ray pipeline, 'tune' for HPO sweep, or 'sweep-pipeline' for full DAG",
    )

    # Config source
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Load a frozen config JSON (e.g. from a previous run)",
    )

    # Identity flags (used by resolver)
    p.add_argument("--model", type=str, default="vgae", help="Model type: vgae, gat, dqn")
    p.add_argument("--scale", type=str, default="large", help="Model scale: large, small")
    p.add_argument(
        "--auxiliaries", type=str, default="none", help="Auxiliary config: none, kd_standard"
    )
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Infrastructure overrides
    p.add_argument("--experiment-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--mp-start-method", type=str, default=None)
    p.add_argument("--run-test", type=_parse_bool, default=None)

    # KD shorthand: --teacher-path sets auxiliaries + model_path
    p.add_argument(
        "--teacher-path",
        type=str,
        default=None,
        help="Shorthand: implies kd_standard aux with given model_path",
    )

    # Flow subcommand options
    p.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="(flow) Re-run evaluation only, skip training",
    )
    p.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="(flow) Use Ray local mode instead of cluster",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(flow) Print the stage chain without executing",
    )

    # Tune subcommand options
    p.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="(tune) Number of HPO trials",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="(tune) Max concurrent trials",
    )
    p.add_argument(
        "--grace-period",
        type=int,
        default=10,
        help="(tune) ASHA grace period (epochs before early stopping a trial)",
    )
    p.add_argument(
        "--tune-epochs",
        type=int,
        default=50,
        help="(tune) Max epochs per trial",
    )
    p.add_argument(
        "--tune-patience",
        type=int,
        default=15,
        help="(tune) Early stopping patience per trial",
    )

    # Sweep-pipeline options
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(sweep-pipeline) Resume from previous state file (default: True)",
    )

    # Metadata for datalake enrichment
    p.add_argument(
        "--tags", type=str, default="", help="Comma-separated tags for run classification"
    )
    p.add_argument(
        "--sweep-id", type=str, default="", help="Parent sweep ID (set by sweep_pipeline)"
    )

    # Nested overrides via dot-path: --training.lr 0.001, --vgae.latent-dim 16
    p.add_argument(
        "--override",
        "-O",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "VALUE"),
        help="Nested override as 'section.field value' (e.g. -O training.lr 0.001)",
    )

    return p


def _parse_dot_overrides(pairs: list[list[str]]) -> dict:
    """Parse -O key value pairs into a nested dict."""
    result: dict = {}
    for key, value in pairs:
        parts = key.replace("-", "_").split(".")
        # Auto-coerce types
        try:
            typed_value: object = int(value)
        except ValueError:
            try:
                typed_value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    typed_value = value.lower() == "true"
                else:
                    typed_value = value

        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = typed_value
    return result


def _run_flow(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch pipeline flow via Ray.

    --scale filters to a single variant (large, small_kd, small_nokd).
    Without --scale, all variants run.  The argparse default "large" is
    for single-stage dispatch; for flows, we treat it as "run all" unless
    the user explicitly passes a flow-relevant scale value.
    """
    datasets = [args.dataset] if args.dataset else None

    # Detect if --scale was explicitly provided on the CLI
    # (argparse default is "large", which we ignore for flow mode)
    _flow_scales = ("large", "small_kd", "small_nokd")
    scale = args.scale if args.scale in _flow_scales else None
    # Check if user actually passed --scale or if it's the default
    import sys

    if "--scale" not in sys.argv:
        scale = None

    # Dry-run: print the stage chain without executing
    if args.dry_run:
        from graphids.config.resolver import resolve

        cfg = resolve("vgae", "large", dataset=args.dataset or "hcrl_ch")
        variants = cfg.variants
        if scale is not None:
            variants = [v for v in variants if v.name == scale]
        log.info("Dry-run: datasets=%s, scale=%s", datasets, scale)
        for v in variants:
            dep = " (needs teacher)" if v.needs_teacher else ""
            log.info("  Variant '%s' (scale=%s, aux=%s)%s:", v.name, v.scale, v.auxiliaries, dep)
            for s in v.stages:
                log.info("    → %s", s)
        return

    from .orchestration.ray_pipeline import eval_pipeline, train_pipeline

    if args.eval_only:
        log.info("Starting Ray evaluation flow (datasets=%s, scale=%s)", datasets, scale)
        eval_pipeline(datasets=datasets, scale=scale, local=args.local)
    else:
        log.info("Starting Ray training flow (datasets=%s, scale=%s)", datasets, scale)
        train_pipeline(datasets=datasets, scale=scale, local=args.local)


def _init_wandb(cfg: PipelineConfig, stage: str, run_name: str):
    """Initialize a W&B run. Returns the run object, or None on failure."""
    try:
        import wandb
    except ImportError:
        return None

    # Offline mode on SLURM compute nodes (no internet); sync later via onsuccess
    if _ON_COMPUTE_NODE and not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "offline"

    try:
        return wandb.init(
            project="kd-gat",
            name=run_name,
            config=cfg.model_dump(),
            tags=[cfg.dataset, cfg.model_type, cfg.scale, stage],
            reinit=True,
        )
    except Exception as e:
        logging.getLogger("pipeline").warning("wandb.init() failed: %s", e)
        return None


def _wandb_log_metrics(result: dict) -> None:
    """Log final result metrics to the active W&B run."""
    try:
        import wandb

        if wandb.run is None:
            return
        flat: dict[str, float] = {}
        for model_key, model_metrics in result.items():
            if model_key == "test":
                continue  # test metrics are nested differently
            if isinstance(model_metrics, dict) and "core" in model_metrics:
                for k, v in model_metrics["core"].items():
                    if isinstance(v, (int, float)):
                        flat[f"{model_key}/{k}"] = v
        if flat:
            wandb.log(flat)
    except Exception:
        pass


def _sync_lakehouse(
    cfg: PipelineConfig,
    stage: str,
    run_name: str,
    result: object = None,
    success: bool = True,
    failure_reason: str | None = None,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_seconds: float | None = None,
    peak_gpu_mb: float | None = None,
    slurm_job_id: str | None = None,
    gpu_name: str | None = None,
    batch_size_used: int | None = None,
    run_type: str = "production",
    sweep_id: str | None = None,
    teacher_run_id: str | None = None,
    config_hash: str | None = None,
    tags: str | None = None,
) -> None:
    """Fire-and-forget sync to datalake (Parquet)."""
    try:
        from .lakehouse import sync_to_lakehouse

        sync_to_lakehouse(
            run_id=run_name,
            dataset=cfg.dataset,
            model_type=cfg.model_type,
            scale=cfg.scale,
            stage=stage,
            has_kd=cfg.has_kd,
            metrics=result if isinstance(result, dict) else None,
            success=success,
            failure_reason=failure_reason,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            peak_gpu_mb=peak_gpu_mb,
            slurm_job_id=slurm_job_id,
            gpu_name=gpu_name,
            batch_size_used=batch_size_used,
            run_type=run_type,
            sweep_id=sweep_id,
            teacher_run_id=teacher_run_id,
            config_hash=config_hash,
            tags=tags,
        )
    except Exception as e:
        logging.getLogger("pipeline").debug("Lakehouse sync skipped: %s", e)


def _finish_wandb() -> None:
    """Finish the active W&B run if one exists."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _resolve_tune_stage(args: argparse.Namespace, log: logging.Logger) -> str | None:
    """Resolve tune stage name from --model argument."""
    from .orchestration.tune_config import _STAGE_MODEL

    _model_to_stage = {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}

    if args.model in _STAGE_MODEL:
        return args.model
    elif args.model in _model_to_stage:
        return _model_to_stage[args.model]
    else:
        log.error(
            "For 'tune', --model must be a stage name (autoencoder, curriculum, fusion) "
            "or model type (vgae, gat, dqn). Got: %s",
            args.model,
        )
        return None


def _run_tune(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch HPO sweep via Ray Tune."""
    tune_stage = _resolve_tune_stage(args, log)
    if tune_stage is None:
        return

    # Dry-run: validate everything without running trials
    if args.dry_run:
        from .orchestration.tune_config import dry_run_tune

        dry_run_tune(
            stage=tune_stage,
            dataset=args.dataset or "hcrl_sa",
            scale=args.scale,
            num_samples=args.num_samples,
            max_concurrent=args.max_concurrent,
            max_epochs=args.tune_epochs,
            patience=args.tune_patience,
        )
        return

    from .orchestration.tune_config import run_tune

    log.info(
        "Starting tune: stage=%s, dataset=%s, scale=%s, samples=%d, epochs=%d, patience=%d",
        tune_stage,
        args.dataset or "hcrl_sa",
        args.scale,
        args.num_samples,
        args.tune_epochs,
        args.tune_patience,
    )

    results = run_tune(
        stage=tune_stage,
        dataset=args.dataset or "hcrl_sa",
        scale=args.scale,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        grace_period=args.grace_period,
        local=args.local,
        max_epochs=args.tune_epochs,
        patience=args.tune_patience,
    )

    best = results.get_best_result(metric="val_loss", mode="min")
    log.info("Tune complete. Best val_loss=%.6f", best.metrics.get("val_loss", float("inf")))
    log.info("Best config: %s", best.config)


def _run_sweep_pipeline(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch full sweep pipeline DAG."""
    from .orchestration.sweep_pipeline import run_sweep_pipeline

    log.info(
        "Starting sweep pipeline: dataset=%s, scale=%s, samples=%d, resume=%s, dry_run=%s",
        args.dataset or "hcrl_sa",
        args.scale,
        args.num_samples,
        args.resume,
        args.dry_run,
    )

    run_sweep_pipeline(
        dataset=args.dataset or "hcrl_sa",
        scale=args.scale,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        resume=args.resume,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )
    log = logging.getLogger("pipeline")

    # ---- Handle non-training subcommands ----
    if args.stage == "flow":
        _run_flow(args, log)
        return

    if args.stage == "tune":
        _run_tune(args, log)
        return

    if args.stage == "sweep-pipeline":
        _run_sweep_pipeline(args, log)
        return

    # ---- Build config ----
    if args.config:
        cfg = PipelineConfig.load(args.config)
        log.info("Loaded frozen config: %s", args.config)
    else:
        # Build overrides dict
        _OVERRIDE_FIELDS = (
            "dataset",
            "seed",
            "experiment_root",
            "device",
            "num_workers",
            "mp_start_method",
            "run_test",
        )
        overrides = {f: getattr(args, f) for f in _OVERRIDE_FIELDS if getattr(args, f) is not None}

        # Handle --teacher-path shorthand
        aux_name = args.auxiliaries
        if args.teacher_path:
            if aux_name == "none":
                aux_name = "kd_standard"
            overrides.setdefault("auxiliaries", [{"type": "kd", "model_path": args.teacher_path}])

        # Parse dot-path overrides
        dot_overrides = _parse_dot_overrides(args.override)
        if dot_overrides:
            from graphids.config.resolver import _deep_merge

            _deep_merge(overrides, dot_overrides)

        cfg = resolve(args.model, args.scale, auxiliaries=aux_name, **overrides)
        log.info("Resolved config: model=%s, scale=%s, aux=%s", args.model, args.scale, aux_name)

    # ---- Validate ----
    validate(cfg, args.stage)

    # ---- Archive completed run if re-running same config ----
    sdir = stage_dir(cfg, args.stage)
    archive = None
    if (sdir / "metrics.json").exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = sdir.parent / f"{sdir.name}.archive_{ts}"
        sdir.rename(archive)
        log.warning("Archived completed run → %s", archive)

    # ---- Save frozen config ----
    cfg_out = config_path(cfg, args.stage)
    cfg.save(cfg_out)
    log.info("Frozen config: %s", cfg_out)

    # ---- Run ID ----
    run_name = run_id(cfg, args.stage)
    log.info("Run started: %s", run_name)

    # ---- W&B init ----
    _wandb_run = _init_wandb(cfg, args.stage, run_name)

    # ---- Collect environment metadata ----
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # ---- Compute datalake enrichment fields ----
    import hashlib
    import json

    config_hash = hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]

    # Detect run_type
    sweep_id = args.sweep_id or None
    tags_str = args.tags or None
    if sweep_id:
        run_type = "sweep_best"
    elif cfg.training.max_epochs < 10:
        run_type = "smoke_test"
    else:
        run_type = "production"

    # Extract teacher lineage for KD runs
    teacher_run_id = None
    if cfg.has_kd and cfg.kd and cfg.kd.model_path:
        teacher_path = cfg.kd.model_path
        # model_path points to a checkpoint; extract the run_id from its parent dirs
        # e.g. "experimentruns/hcrl_ch/vgae_large_autoencoder/best_model.pt" → "hcrl_ch/vgae_large_autoencoder"
        tp = Path(teacher_path)
        if tp.parent.parent.name and tp.parent.name:
            teacher_run_id = f"{tp.parent.parent.name}/{tp.parent.name}"

    # ---- Dispatch ----
    started_at = datetime.now(UTC).isoformat()
    t_start = time.monotonic()
    try:
        from .stages import STAGE_FNS

        result = STAGE_FNS[args.stage](cfg)

        t_end = time.monotonic()
        completed_at = datetime.now(UTC).isoformat()
        duration_seconds = t_end - t_start

        # Capture GPU peak memory
        peak_gpu_mb = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
        except Exception:
            pass

        log.info(
            "Stage '%s' complete (%.1fs, peak_gpu=%.0fMB). Result: %s",
            args.stage,
            duration_seconds,
            peak_gpu_mb or 0.0,
            result,
        )

        # Log final metrics to W&B
        if _wandb_run is not None and isinstance(result, dict):
            _wandb_log_metrics(result)

        # Sync to datalake (fire-and-forget)
        _sync_lakehouse(
            cfg,
            args.stage,
            run_name,
            result,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            peak_gpu_mb=peak_gpu_mb,
            slurm_job_id=slurm_job_id,
            gpu_name=gpu_name,
            batch_size_used=cfg.training.batch_size,
            run_type=run_type,
            sweep_id=sweep_id,
            teacher_run_id=teacher_run_id,
            config_hash=config_hash,
            tags=tags_str,
        )

        # Register artifacts in datalake (fire-and-forget)
        try:
            from .lakehouse import register_artifacts

            register_artifacts(run_name, sdir)
        except Exception as e:
            log.debug("Artifact registration skipped: %s", e)

        # Success → delete archive
        if archive and archive.exists():
            import shutil

            shutil.rmtree(archive, ignore_errors=True)

        log.info("Run completed successfully")

    except Exception as e:
        t_end = time.monotonic()
        completed_at = datetime.now(UTC).isoformat()
        duration_seconds = t_end - t_start
        # Failure → restore archive
        if archive and archive.exists():
            if sdir.exists():
                import shutil

                shutil.rmtree(sdir, ignore_errors=True)
            archive.rename(sdir)
            log.warning("Restored archive after failure: %s", sdir)
        _sync_lakehouse(
            cfg,
            args.stage,
            run_name,
            None,
            success=False,
            failure_reason=str(e),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            slurm_job_id=slurm_job_id,
            gpu_name=gpu_name,
            run_type=run_type,
            sweep_id=sweep_id,
            teacher_run_id=teacher_run_id,
            config_hash=config_hash,
            tags=tags_str,
        )
        log.error("Run failed: %s", str(e))
        raise

    finally:
        _finish_wandb()


if __name__ == "__main__":
    main()
