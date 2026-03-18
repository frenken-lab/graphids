"""Single CLI entry point.

Training stages use Hydra override grammar (key=value):
    python -m graphids.pipeline.cli stage=autoencoder model=vgae_large dataset=hcrl_sa
    python -m graphids.pipeline.cli stage=curriculum model=gat_small auxiliary=kd_standard training.lr=0.001
    python -m graphids.pipeline.cli show-config model=vgae_large dataset=hcrl_sa

Non-training subcommands use argparse:
    python -m graphids.pipeline.cli orchestrate --dataset hcrl_sa --seeds 42,123,456 --dry-run
    python -m graphids.pipeline.cli lake --lake-action status
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
import sys
import time
from datetime import datetime
from pathlib import Path

from graphids.config import (
    DEFAULT_DATASET,
    MLFLOW_TRACKING_URI,
    STAGES,
    SWEEP_ID,
    PipelineConfig,
    config_path,
    run_id,
    run_metadata,
    stage_dir,
)

from .validate import validate

log = logging.getLogger("pipeline")

# Subcommands that bypass Hydra config composition
_SUBCOMMANDS = frozenset({
    "orchestrate",
    "preprocess",
    "tune",
    "sweep-pipeline",
    "plan",
    "lake",
    "show-config",
})


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Training entry point (Hydra Compose API)
# ---------------------------------------------------------------------------


def _prepare_overrides(raw_overrides: list[str]) -> list[str]:
    """Ensure nested-key overrides use Hydra's ++ (force-set) prefix.

    Config group selections (model=X, dataset=X, etc.) and top-level keys
    (stage=X, seed=X) pass through unchanged. Nested dot-path overrides
    like training.lr=0.001 need ++ because config.yaml has empty anchors.
    """
    _PASSTHROUGH_KEYS = {"model", "auxiliary", "dataset", "stage", "seed"}
    prepared = []
    for ov in raw_overrides:
        if "=" not in ov:
            prepared.append(ov)
            continue
        key = ov.split("=", 1)[0].lstrip("+~")
        if key in _PASSTHROUGH_KEYS:
            prepared.append(ov)
        elif not ov.startswith(("+", "~")):
            prepared.append(f"++{ov}")
        else:
            prepared.append(ov)
    return prepared


def _run_training(overrides: list[str]) -> None:
    """Compose config from Hydra overrides and dispatch a training stage."""
    from omegaconf import OmegaConf

    from graphids.config._hydra_bridge import _hydra_compose

    with _hydra_compose(_prepare_overrides(overrides)) as hydra_cfg:
        raw = OmegaConf.to_object(hydra_cfg)

    stage = raw.pop("stage")
    if stage not in STAGES:
        log.error("Unknown training stage: %s. Valid: %s", stage, list(STAGES.keys()))
        raise SystemExit(1)

    cfg = PipelineConfig.model_validate(raw)
    log.info("Resolved config: model=%s, scale=%s, dataset=%s", cfg.model_type, cfg.scale, cfg.dataset)
    _run_single_stage(cfg, stage)


def _show_config(overrides: list[str]) -> None:
    """Print resolved config as YAML without running."""
    from omegaconf import OmegaConf

    from graphids.config._hydra_bridge import _hydra_compose

    with _hydra_compose(_prepare_overrides(overrides)) as hydra_cfg:
        print(OmegaConf.to_yaml(hydra_cfg))


# ---------------------------------------------------------------------------
# Non-training subcommands
# ---------------------------------------------------------------------------


def _run_preprocess(argv: list[str]) -> None:
    """Build preprocessed graph cache for a dataset."""
    p = argparse.ArgumentParser(prog="pipeline preprocess")
    p.add_argument("--dataset", type=str, default=None)
    args = p.parse_args(argv)

    from graphids.config import resolve

    dataset = args.dataset or DEFAULT_DATASET
    cfg = resolve("vgae", "large", dataset=dataset)

    log.info("Preprocessing dataset: %s", dataset)
    from graphids.core.preprocessing import PreprocessingPipeline

    PreprocessingPipeline(cfg).load_dataset()
    log.info("Preprocessed cache ready for %s", dataset)


def _run_tune(argv: list[str]) -> None:
    """Dispatch HPO sweep via Ray Tune."""
    p = argparse.ArgumentParser(prog="pipeline tune")
    p.add_argument("--model", type=str, default="vgae")
    p.add_argument("--scale", type=str, default="large")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--grace-period", type=int, default=10)
    p.add_argument("--tune-epochs", type=int, default=50)
    p.add_argument("--tune-patience", type=int, default=15)
    p.add_argument("--local", action="store_true", default=False)
    p.add_argument("--warm-start-from", type=str, default=None)
    args = p.parse_args(argv)

    from .orchestration.tune_config import _STAGE_MODEL, run_tune

    _model_to_stage = {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}

    if args.model in _STAGE_MODEL:
        tune_stage = args.model
    elif args.model in _model_to_stage:
        tune_stage = _model_to_stage[args.model]
    else:
        log.error(
            "For 'tune', --model must be a stage name (autoencoder, curriculum, fusion) "
            "or model type (vgae, gat, dqn). Got: %s",
            args.model,
        )
        return

    log.info(
        "Starting tune: stage=%s, dataset=%s, scale=%s, samples=%d",
        tune_stage,
        args.dataset or DEFAULT_DATASET,
        args.scale,
        args.num_samples,
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
        warm_start_from=args.warm_start_from,
    )

    best = results.get_best_result(metric="val_loss", mode="min")
    log.info("Tune complete. Best val_loss=%.6f", best.metrics.get("val_loss", float("inf")))
    log.info("Best config: %s", best.config)


def _run_sweep_pipeline(argv: list[str]) -> None:
    """Dispatch full sweep pipeline DAG."""
    p = argparse.ArgumentParser(prog="pipeline sweep-pipeline")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--scale", type=str, default="large")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--tune-epochs", type=int, default=50)
    p.add_argument("--tune-patience", type=int, default=15)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args(argv)

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


def _run_lake(argv: list[str]) -> None:
    """Dispatch lake management commands."""
    p = argparse.ArgumentParser(prog="pipeline lake")
    p.add_argument(
        "--lake-action",
        type=str,
        default="status",
        choices=["rebuild-catalog", "verify", "status"],
    )
    args = p.parse_args(argv)

    from graphids.config import lake_catalog_path, lake_root_from_env

    lake_root = lake_root_from_env()
    if lake_root is None:
        log.error("KD_GAT_LAKE_ROOT not set. Run: export KD_GAT_LAKE_ROOT=/fs/ess/PAS1266/kd-gat")
        return

    action = args.lake_action

    if action == "rebuild-catalog":
        from graphids.pipeline.catalog import rebuild_catalog

        catalog_path = rebuild_catalog(lake_root)
        log.info("Catalog rebuilt: %s", catalog_path)

    elif action == "verify":
        from graphids.pipeline.manifest import verify_manifest

        errors_total = 0
        run_count = 0
        for tier_dir in [lake_root / "production", lake_root / "dev"]:
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
        from graphids.pipeline.catalog import catalog_status

        cat_path = lake_catalog_path(lake_root)
        status = catalog_status(cat_path)
        if not status.get("exists"):
            log.info("Lake root: %s", lake_root)
            log.info(
                "Catalog: not built yet. Run: python -m graphids.pipeline.cli lake --lake-action rebuild-catalog"
            )
            return
        log.info("Lake root: %s", lake_root)
        log.info("Total runs: %d", status["total_runs"])
        log.info("By stage: %s", status["by_stage"])
        log.info("By dataset: %s", status["by_dataset"])


def _run_plan(argv: list[str]) -> None:
    """Build and save (or preview) an execution plan."""
    p = argparse.ArgumentParser(prog="pipeline plan")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--variant", type=str, default="large")
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--plan-output", type=str, default=None)
    args = p.parse_args(argv)

    from graphids.config import parse_seeds

    from .orchestration.plan import build_plan

    dataset = args.dataset or DEFAULT_DATASET
    seeds = parse_seeds(args.seeds) if args.seeds else [42]
    variant = args.variant

    plan = build_plan(dataset=dataset, seeds=seeds, variant=variant)

    if args.dry_run:
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

    if args.plan_output:
        out_path = Path(args.plan_output)
    else:
        from graphids.config import lake_root_from_env

        lake = lake_root_from_env() or Path("experimentruns")
        out_path = lake / dataset / "plan.json"

    plan.save(out_path)
    log.info("Plan saved: %s (%d jobs, hash=%s)", out_path, len(plan.jobs), plan.plan_hash)


def _run_orchestrate(argv: list[str]) -> None:
    """Dispatch pipeline via Dagster fire-and-forget (SLURM dependency chains)."""
    p = argparse.ArgumentParser(prog="pipeline orchestrate")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--fire-and-forget", action="store_true", default=False)
    args = p.parse_args(argv)

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


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "show-config": _show_config,
    "preprocess": _run_preprocess,
    "tune": _run_tune,
    "sweep-pipeline": _run_sweep_pipeline,
    "lake": _run_lake,
    "plan": _run_plan,
    "orchestrate": _run_orchestrate,
}


# ---------------------------------------------------------------------------
# Stage lifecycle helpers
# ---------------------------------------------------------------------------


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
        from graphids.pipeline.manifest import write_manifest

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


# ---------------------------------------------------------------------------
# Single stage execution
# ---------------------------------------------------------------------------


def _run_single_stage(
    cfg: PipelineConfig,
    stage: str,
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
    # sweep_id and user_tags are now in run_metadata() via EnvironmentSettings
    if SWEEP_ID:
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
    if teacher_run_id_str:
        extra_tags["teacher_run_id"] = teacher_run_id_str

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    args = argv if argv is not None else sys.argv[1:]

    if args and args[0] in _SUBCOMMANDS:
        _DISPATCH[args[0]](args[1:])
    else:
        _run_training(args)


if __name__ == "__main__":
    main()
