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
    DEFAULT_DATASET,
    MLFLOW_TRACKING_URI,
    STAGES,
    PipelineConfig,
    config_path,
    resolve,
    run_id,
    run_metadata,
    stage_dir,
)

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
    from graphids.config import parse_seeds

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
        choices=list(STAGES.keys())
        + ["preprocess", "tune", "sweep-pipeline", "orchestrate", "plan", "lake"],
        help="Training stage, 'preprocess' for graph cache, 'tune' for HPO, 'sweep-pipeline' for full DAG, 'orchestrate' for Dagster pipeline, 'plan' for execution plan, or 'lake' for data lake commands",
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
        help="Multi-seed dispatch: single seed (42) or comma-separated (42,123,456)",
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

    # General options
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print plan without executing",
    )

    # Tune subcommand options
    p.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="(tune) Use Ray local mode instead of cluster",
    )
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

    # Plan options
    p.add_argument(
        "--variant",
        type=str,
        default="large",
        help="(plan) Pipeline variant: large, small, small_kd (default: large)",
    )
    p.add_argument(
        "--plan-output",
        type=str,
        default=None,
        help="(plan) Output path for plan.json (default: experimentruns/{dataset}/plan.json)",
    )

    # Orchestrate options (Dagster fire-and-forget)
    p.add_argument(
        "--fire-and-forget",
        action="store_true",
        default=False,
        help="(orchestrate) Submit all jobs with --dependency chains, no polling",
    )

    # Lake subcommand options
    p.add_argument(
        "--lake-action",
        type=str,
        default="status",
        choices=["rebuild-catalog", "verify", "status"],
        help="(lake) Action: rebuild-catalog, verify, or status",
    )

    # Checkpoint resume (set by orchestrator on TIMEOUT resubmit)
    p.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Lightning .ckpt path to resume training from",
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


def _run_preprocess(args: argparse.Namespace, log: logging.Logger) -> None:
    """Build preprocessed graph cache for a dataset.

    Calls load_dataset() which checks cache validity (version, feature dims)
    and rebuilds if needed.
    """
    from graphids.config import resolve

    dataset = args.dataset or DEFAULT_DATASET
    cfg = resolve("vgae", "large", dataset=dataset)

    log.info("Preprocessing dataset: %s", dataset)
    from graphids.core.preprocessing import PreprocessingPipeline

    PreprocessingPipeline(cfg).load_dataset()
    log.info("Preprocessed cache ready for %s", dataset)


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

    from .orchestration.tune_config import run_tune

    warm_start = getattr(args, "warm_start_from", None)
    log.info(
        "Starting tune: stage=%s, dataset=%s, scale=%s, samples=%d, epochs=%d, patience=%d, warm_start_from=%s",
        tune_stage,
        args.dataset or DEFAULT_DATASET,
        args.scale,
        args.num_samples,
        args.tune_epochs,
        args.tune_patience,
        warm_start,
    )

    results = run_tune(
        stage=tune_stage,
        dataset=args.dataset or DEFAULT_DATASET,
        scale=args.scale,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        grace_period=args.grace_period,
        local=args.local,
        max_epochs=args.tune_epochs,
        patience=args.tune_patience,
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
        args.dataset or DEFAULT_DATASET,
        args.scale,
        args.num_samples,
        args.resume,
        args.dry_run,
    )

    run_sweep_pipeline(
        dataset=args.dataset or DEFAULT_DATASET,
        scale=args.scale,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        resume=args.resume,
        dry_run=args.dry_run,
    )


def _run_lake(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch lake management commands."""
    from graphids.lake.config import LakeConfig

    lake = LakeConfig.from_env()
    if lake is None:
        log.error("KD_GAT_LAKE_ROOT not set. Run: export KD_GAT_LAKE_ROOT=/fs/ess/PAS1266/kd-gat")
        return

    action = args.lake_action

    if action == "rebuild-catalog":
        from graphids.lake.catalog import rebuild_catalog

        catalog_path = rebuild_catalog(lake.lake_root)
        log.info("Catalog rebuilt: %s", catalog_path)

    elif action == "verify":
        from graphids.lake.manifest import verify_manifest

        errors_total = 0
        run_count = 0
        for tier_dir in [lake.lake_root / "production", lake.lake_root / "dev"]:
            if not tier_dir.exists():
                continue
            for manifest_file in tier_dir.rglob("_manifest.json"):
                run_dir = manifest_file.parent
                ok, errors = verify_manifest(run_dir)
                run_count += 1
                if not ok:
                    errors_total += len(errors)
                    log.warning("FAILED: %s — %s", run_dir, "; ".join(errors))
        log.info("Verified %d runs, %d errors", run_count, errors_total)

    elif action == "status":
        from graphids.lake.catalog import catalog_status

        catalog_path = lake.catalog_path()
        status = catalog_status(catalog_path)
        if not status.get("exists"):
            log.info("Lake root: %s", lake.lake_root)
            log.info(
                "Catalog: not built yet. Run: python -m graphids.pipeline.cli lake --lake-action rebuild-catalog"
            )
            return
        log.info("Lake root: %s", lake.lake_root)
        log.info("Total runs: %d", status["total_runs"])
        log.info("By stage: %s", status["by_stage"])
        log.info("By dataset: %s", status["by_dataset"])


def _run_plan(args: argparse.Namespace, log: logging.Logger) -> None:
    """Build and save (or preview) an execution plan."""
    from graphids.config import parse_seeds

    from .orchestration.plan import build_plan

    dataset = args.dataset or DEFAULT_DATASET
    seeds = parse_seeds(args.seeds) if args.seeds else [42]
    variant = args.variant

    plan = build_plan(dataset=dataset, seeds=seeds, variant=variant)

    if args.dry_run:
        # Preview mode: print plan summary to stdout
        log.info(
            "Plan: %s | variant=%s | %d seeds | %d jobs | hash=%s",
            dataset,
            variant,
            len(seeds),
            len(plan.jobs),
            plan.plan_hash,
        )
        for job in plan.jobs:
            deps = ", ".join(d.job_id for d in job.depends_on) or "(none)"
            log.info(
                "  %s  [%s %s %s]  deps=[%s]  res=%s",
                job.id,
                job.model_type,
                job.scale,
                job.stage,
                deps,
                f"{job.resources.partition}/{job.resources.memory_gb}GB/{job.resources.gpus}gpu",
            )
        return

    # Save plan to file
    if args.plan_output:
        out_path = Path(args.plan_output)
    else:
        from graphids.config import EXPERIMENT_ROOT

        out_path = Path(EXPERIMENT_ROOT) / dataset / "plan.json"

    plan.save(out_path)
    log.info("Plan saved: %s (%d jobs, hash=%s)", out_path, len(plan.jobs), plan.plan_hash)


def _run_orchestrate(args: argparse.Namespace, log: logging.Logger) -> None:
    """Dispatch pipeline via Dagster fire-and-forget (SLURM dependency chains)."""
    from .orchestration.dagster_defs import fire_and_forget

    dataset = args.dataset or DEFAULT_DATASET
    seeds = None
    if args.seeds:
        from graphids.config import parse_seeds

        seeds = parse_seeds(args.seeds)

    log.info(
        "Orchestrate (fire-and-forget): dataset=%s, seeds=%s, dry_run=%s",
        dataset,
        seeds,
        args.dry_run,
    )

    job_ids = fire_and_forget(dataset=dataset, seeds=seeds, dry_run=args.dry_run)

    log.info("Submitted %d jobs:", len(job_ids))
    for name, jid in job_ids.items():
        log.info("  %s: %s", name, jid)


def _archive_previous(sdir: Path, log: logging.Logger) -> Path | None:
    """Archive a completed run directory before re-running. Returns archive path or None."""
    if not (sdir / "config.json").exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = sdir.parent / f"{sdir.name}.archive_{ts}"
    sdir.rename(archive)
    log.warning("Archived completed run → %s", archive)
    return archive


def _log_stage_artifacts(cfg: PipelineConfig, stage: str, sdir: Path) -> None:
    """Log stage artifacts to MLflow and populate the artifact cache."""
    from graphids.pipeline.artifacts import put_artifact

    for artifact_name in [
        "best_model.pt",
        "config.json",
        "embeddings.npz",
        "attention_weights.npz",
        "dqn_policy.json",
        "explanations.npz",
    ]:
        artifact_path = sdir / artifact_name
        if artifact_path.exists():
            put_artifact(cfg, stage, artifact_path)


def _write_lake_manifest(
    cfg: PipelineConfig,
    stage: str,
    sdir: Path,
    log: logging.Logger,
    metrics: dict | None = None,
) -> None:
    """Write _manifest.json for the ESS data lake."""
    try:
        from graphids.lake.manifest import write_manifest

        aux_type = cfg.auxiliaries[0].type if cfg.auxiliaries else "none"
        write_manifest(
            sdir,
            dataset=cfg.dataset,
            model_type=cfg.model_type,
            scale=cfg.scale,
            stage=stage,
            auxiliaries=aux_type,
            seed=cfg.seed,
            metrics=metrics,
        )
    except Exception as e:
        log.warning("Failed to write manifest: %s", e)


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
    archive = _archive_previous(sdir, log)

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

    # ---- Checkpoint resume (orchestrator TIMEOUT resubmit) ----
    ckpt_path = getattr(args, "ckpt_path", None)
    if ckpt_path:
        os.environ["KD_GAT_CKPT_PATH"] = str(ckpt_path)

    # ---- Dispatch ----
    t_start = time.monotonic()
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

            # Extract stage metrics from result dict (all stages now return {"metrics": {...}})
            stage_metrics = {}
            if isinstance(result, dict):
                stage_metrics = result.get("metrics", {})

            # Enrich with runtime info
            stage_metrics["duration_seconds"] = duration_seconds
            if peak_gpu_mb is not None:
                stage_metrics["peak_gpu_mb"] = peak_gpu_mb

            # Log scalar metrics to MLflow
            mlflow_metrics = {"duration_seconds": duration_seconds}
            if peak_gpu_mb is not None:
                mlflow_metrics["peak_gpu_mb"] = peak_gpu_mb
            for k, v in stage_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow_metrics[k] = v
            mlflow.log_metrics(mlflow_metrics)

            _log_stage_artifacts(cfg, stage, sdir)
            _write_lake_manifest(cfg, stage, sdir, log, metrics=stage_metrics)

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
    if args.stage == "lake":
        _run_lake(args, log)
        return

    if args.stage == "preprocess":
        _run_preprocess(args, log)
        return

    if args.stage == "tune":
        _run_tune(args, log)
        return

    if args.stage == "sweep-pipeline":
        _run_sweep_pipeline(args, log)
        return

    if args.stage == "plan":
        _run_plan(args, log)
        return

    if args.stage == "orchestrate":
        _run_orchestrate(args, log)
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
            from graphids.config.handler import _deep_merge

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
