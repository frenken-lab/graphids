"""Single CLI entry point.

Usage:
    python -m graphids.pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_ch
    python -m graphids.pipeline.cli curriculum  --model gat --scale small --auxiliaries kd_standard --dataset hcrl_sa
    python -m graphids.pipeline.cli fusion      --config path/to/config.json
    python -m graphids.pipeline.cli autoencoder --model vgae --scale large --seeds 42,123,456
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
from datetime import datetime
from pathlib import Path

from graphids.config import (
    MLFLOW_TRACKING_URI,
    STAGES,
    PipelineConfig,
    config_path,
    get_resolver,
    run_id,
    run_metadata,
    stage_dir,
)
from graphids.config.resolver import resolve

from .validate import validate


def _setup_mlflow(run_name: str, cfg: PipelineConfig, stage: str, tags: dict | None = None):
    """Set up MLflow tracking and start a run. Returns the active run."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"kd-gat-{stage}")

    # Use run_metadata() as the base tags — single source of truth
    mlflow_tags = run_metadata(cfg, stage)
    if tags:
        mlflow_tags.update(tags)

    return mlflow.start_run(run_name=run_name, tags=mlflow_tags)


def _parse_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _parse_seeds(value: str) -> list[int]:
    """Parse --seeds argument: comma-separated ints or a count for random seeds."""
    from graphids.config.constants import parse_seeds

    try:
        return parse_seeds(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid --seeds value '{value}': {e}") from e


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="KD-GAT training pipeline",
    )
    p.add_argument(
        "stage",
        choices=list(STAGES.keys()) + ["flow", "tune", "sweep-pipeline", "coordinator"],
        help="Training stage, 'flow' for Ray pipeline, 'tune' for HPO, 'sweep-pipeline' for full DAG, or 'coordinator' for SLURM orchestration",
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

    # Multi-seed support
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Multi-seed dispatch: comma-separated (42,123,456) or count (5 = first 5 defaults)",
    )

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
    p.add_argument(
        "--inprocess",
        action="store_true",
        default=False,
        help="(tune) Use in-process trainable with per-epoch ASHA reporting (~2.5x faster)",
    )
    p.add_argument(
        "--warm-start-from",
        type=str,
        default=None,
        help="(tune) Dataset name to warm-start from (loads prior sweep results)",
    )

    # Sweep-pipeline options
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(sweep-pipeline) Resume from previous state file (default: True)",
    )

    # Coordinator options
    p.add_argument(
        "--resume-state",
        type=str,
        default="",
        help="(coordinator) Resume from state JSON file path",
    )
    p.add_argument("--poll-interval", type=int, default=30, help="(coordinator) Polling interval")

    # Checkpoint resume (set by coordinator on TIMEOUT resubmit)
    p.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Lightning .ckpt path to resume training from (set by coordinator on TIMEOUT)",
    )

    # Metadata tags
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

    # Parse seeds for multi-seed flow dispatch
    seeds = _parse_seeds(args.seeds) if args.seeds else None

    if args.eval_only:
        log.info("Starting Ray evaluation flow (datasets=%s, scale=%s)", datasets, scale)
        eval_pipeline(datasets=datasets, scale=scale, local=args.local)
    else:
        log.info(
            "Starting Ray training flow (datasets=%s, scale=%s, seeds=%s)",
            datasets,
            scale,
            seeds,
        )
        train_pipeline(datasets=datasets, scale=scale, local=args.local, seeds=seeds)


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

    warm_start = getattr(args, "warm_start_from", None)
    log.info(
        "Starting tune: stage=%s, dataset=%s, scale=%s, samples=%d, epochs=%d, patience=%d, inprocess=%s, warm_start_from=%s",
        tune_stage,
        args.dataset or "hcrl_sa",
        args.scale,
        args.num_samples,
        args.tune_epochs,
        args.tune_patience,
        args.inprocess,
        warm_start,
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
        inprocess=args.inprocess,
        warm_start_from=warm_start,
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
        inprocess=args.inprocess,
    )


def _run_coordinator(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch stateful SLURM coordinator."""
    from pathlib import Path

    from .coordinator import PipelineCoordinator
    from .state import load_state

    if args.resume_state:
        # Resume from existing state file
        state_path = Path(args.resume_state)
        state = load_state(state_path)
        if not state or "stages" not in state:
            log.error("Invalid or empty state file: %s", state_path)
            return

        log.info("Resuming coordinator from %s (%d stages)", state_path, len(state["stages"]))
        coordinator = PipelineCoordinator(
            datasets=state["datasets"],
            seeds=state["seeds"],
            scale=state.get("scale", "large"),
            auxiliaries=state.get("auxiliaries", "none"),
            state_path=state_path,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )
    else:
        if not args.dataset:
            log.error("--dataset is required for coordinator mode (or use --resume-state)")
            return

        datasets = [d.strip() for d in args.dataset.split(",")]
        seeds = _parse_seeds(args.seeds) if args.seeds else [42]

        log.info(
            "Starting coordinator: datasets=%s, seeds=%s, scale=%s, dry_run=%s",
            datasets,
            seeds,
            args.scale,
            args.dry_run,
        )

        coordinator = PipelineCoordinator(
            datasets=datasets,
            seeds=seeds,
            scale=args.scale,
            auxiliaries=args.auxiliaries,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )

    coordinator.run()


def _run_single_stage(
    cfg: PipelineConfig,
    stage: str,
    args: argparse.Namespace,
    log: logging.Logger,
) -> None:
    """Execute a single training stage with MLflow tracking and artifact caching."""
    # ---- Validate ----
    validate(cfg, stage)

    # ---- Archive completed run if re-running same config ----
    sdir = stage_dir(cfg, stage)
    archive = None
    if (sdir / "metrics.json").exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = sdir.parent / f"{sdir.name}.archive_{ts}"
        sdir.rename(archive)
        log.warning("Archived completed run → %s", archive)

    # ---- Save frozen config ----
    cfg_out = config_path(cfg, stage)
    cfg.save(cfg_out)
    log.info("Frozen config: %s", cfg_out)

    # ---- Run ID ----
    run_name = run_id(cfg, stage)
    log.info("Run started: %s (seed=%d)", run_name, cfg.seed)

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

    # ---- Enrichment tags (beyond run_metadata) ----
    sweep_id = args.sweep_id or None
    tags_str = args.tags or None
    if sweep_id:
        run_type = "sweep_best"
    elif cfg.training.max_epochs < 10:
        run_type = "smoke_test"
    else:
        run_type = "production"

    teacher_run_id_str = None
    if cfg.has_kd and cfg.kd and cfg.kd.model_path:
        tp = Path(cfg.kd.model_path)
        if tp.parent.parent.name and tp.parent.name:
            teacher_run_id_str = f"{tp.parent.parent.name}/{tp.parent.name}"

    # ---- MLflow run context ----
    import mlflow

    extra_tags = {
        "slurm_job_id": slurm_job_id or "",
        "gpu_name": gpu_name or "",
        "run_type": run_type,
    }
    if sweep_id:
        extra_tags["sweep_id"] = sweep_id
    if tags_str:
        extra_tags["user_tags"] = tags_str
    if teacher_run_id_str:
        extra_tags["teacher_run_id"] = teacher_run_id_str

    # ---- Checkpoint resume (coordinator TIMEOUT resubmit) ----
    ckpt_path = getattr(args, "ckpt_path", None)
    if ckpt_path:
        os.environ["KD_GAT_CKPT_PATH"] = str(ckpt_path)

    # ---- Dispatch ----
    t_start = time.monotonic()
    resolver = get_resolver()

    with _setup_mlflow(run_name, cfg, stage, tags=extra_tags):
        try:
            # Log config as params
            mlflow.log_params(
                {
                    "dataset": cfg.dataset,
                    "model_type": cfg.model_type,
                    "scale": cfg.scale,
                    "stage": stage,
                    "has_kd": cfg.has_kd,
                    "seed": cfg.seed,
                    "batch_size": cfg.training.batch_size,
                    "max_epochs": cfg.training.max_epochs,
                    "lr": cfg.training.lr,
                }
            )

            # Log frozen config as artifact
            mlflow.log_artifact(str(cfg_out))

            from .stages import STAGE_FNS

            result = STAGE_FNS[stage](cfg)

            t_end = time.monotonic()
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
                stage,
                duration_seconds,
                peak_gpu_mb or 0.0,
                result,
            )

            # Log post-training metrics to MLflow
            post_metrics = {"duration_seconds": duration_seconds}
            if peak_gpu_mb is not None:
                post_metrics["peak_gpu_mb"] = peak_gpu_mb
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        post_metrics[k] = v
            mlflow.log_metrics(post_metrics)

            # Log artifacts to MLflow AND populate cache via resolver.put()
            for artifact_name in [
                "best_model.pt",
                "config.json",
                "metrics.json",
                "embeddings.npz",
                "attention_weights.npz",
                "dqn_policy.json",
                "explanations.npz",
            ]:
                artifact_path = sdir / artifact_name
                if artifact_path.exists():
                    resolver.put(cfg, stage, artifact_path)

            # Success → delete archive
            if archive and archive.exists():
                import shutil

                shutil.rmtree(archive, ignore_errors=True)

            mlflow.set_tag("status", "success")
            log.info("Run completed successfully")

        except Exception as e:
            t_end = time.monotonic()
            duration_seconds = t_end - t_start

            # Failure → restore archive
            if archive and archive.exists():
                if sdir.exists():
                    import shutil

                    shutil.rmtree(sdir, ignore_errors=True)
                archive.rename(sdir)
                log.warning("Restored archive after failure: %s", sdir)

            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", str(e)[:250])
            mlflow.log_metrics({"duration_seconds": duration_seconds})

            log.error("Run failed: %s", str(e))
            raise


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

    if args.stage == "coordinator":
        _run_coordinator(args, log)
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

    # ---- Multi-seed dispatch ----
    if args.seeds:
        seeds = _parse_seeds(args.seeds)
        log.info("Multi-seed dispatch: seeds=%s, stage=%s", seeds, args.stage)
        for i, seed in enumerate(seeds):
            log.info("=== Seed %d/%d: %d ===", i + 1, len(seeds), seed)
            seed_cfg = cfg.model_copy(update={"seed": seed})
            _run_single_stage(seed_cfg, args.stage, args, log)
        log.info("All %d seeds completed for stage '%s'", len(seeds), args.stage)
        return

    # ---- Single-seed dispatch ----
    _run_single_stage(cfg, args.stage, args, log)


if __name__ == "__main__":
    main()
