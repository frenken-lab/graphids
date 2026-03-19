"""Single CLI entry point (Typer + Hydra).

Training uses Hydra override grammar via the default command:
    python -m graphids.pipeline.cli stage=autoencoder model=vgae_large dataset=hcrl_sa

All other subcommands use typed flags:
    python -m graphids.pipeline.cli sweep --stage autoencoder --num-samples 20
    python -m graphids.pipeline.cli orchestrate --dataset hcrl_sa --dry-run
    python -m graphids.pipeline.cli daemon --status
    python -m graphids.pipeline.cli lake --action status
    python -m graphids.pipeline.cli show-config model=vgae_large dataset=hcrl_sa
    python -m graphids.pipeline.cli preprocess --dataset hcrl_sa
"""

from __future__ import annotations

import torch.multiprocessing as mp

# Must be called before any CUDA or multiprocessing usage.
mp.set_start_method("spawn", force=True)

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

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

app = typer.Typer(
    name="graphids",
    help="KD-GAT pipeline CLI",
    add_completion=False,
    no_args_is_help=False,
)


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------


def _setup_mlflow(run_name: str, cfg: PipelineConfig, stage: str, tags: dict | None = None):
    """Set up MLflow tracking and start a run."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"kd-gat-{stage}")
    mlflow_tags = run_metadata(cfg, stage)
    if tags:
        mlflow_tags.update(tags)
    return mlflow.start_run(run_name=run_name, tags=mlflow_tags)


# ---------------------------------------------------------------------------
# Training (default command — Hydra override grammar)
# ---------------------------------------------------------------------------


def _run_training(overrides: list[str]) -> None:
    """Compose config from Hydra overrides and dispatch a training stage."""
    from graphids.config._hydra_bridge import compose_config
    from omegaconf import OmegaConf

    merged, stage = compose_config(overrides)
    if stage is None or stage not in STAGES:
        log.error("Unknown training stage: %s. Valid: %s", stage, list(STAGES.keys()))
        raise SystemExit(1)

    raw = OmegaConf.to_object(merged)
    raw.pop("stage", None)
    cfg = PipelineConfig.model_validate(raw)
    log.info("Resolved config: model=%s, scale=%s, dataset=%s", cfg.model_type, cfg.scale, cfg.dataset)
    _run_single_stage(cfg, stage)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("show-config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def show_config(ctx: typer.Context):
    """Print resolved config as YAML without running."""
    from graphids.config._hydra_bridge import compose_config
    from omegaconf import OmegaConf

    merged, _stage = compose_config(ctx.args)
    print(OmegaConf.to_yaml(merged))


@app.command()
def preprocess(dataset: Optional[str] = None):
    """Build preprocessed graph cache for a dataset."""
    from graphids.config import resolve
    from graphids.core.preprocessing import PreprocessingPipeline

    ds = dataset or DEFAULT_DATASET
    cfg = resolve("vgae", "large", dataset=ds)
    log.info("Preprocessing dataset: %s", ds)
    PreprocessingPipeline(cfg).load_dataset()
    log.info("Preprocessed cache ready for %s", ds)


@app.command()
def sweep(
    stage: Optional[str] = None,
    full_pipeline: bool = False,
    dataset: Optional[str] = None,
    scale: str = "large",
    num_samples: int = 20,
    max_epochs: int = 50,
    patience: int = 15,
    warm_start_from: Optional[str] = None,
    dry_run: bool = False,
    multi_seed: bool = False,
):
    """Run Optuna HPO sweep (single-stage or full pipeline)."""
    from graphids.config import STAGE_MODEL_MAP
    from .orchestration.optuna_sweep import run_sweep, run_sweep_pipeline

    ds = dataset or DEFAULT_DATASET

    if full_pipeline:
        run_sweep_pipeline(
            dataset=ds, scale=scale, num_samples=num_samples, max_epochs=max_epochs,
            patience=patience, warm_start_from=warm_start_from, dry_run=dry_run,
            multi_seed=multi_seed,
        )
    elif stage:
        _model_to_stage = {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}
        sweep_stage = _model_to_stage.get(stage, stage)
        if sweep_stage not in STAGE_MODEL_MAP:
            log.error("--stage must be autoencoder/curriculum/fusion or vgae/gat/dqn. Got: %s", stage)
            raise SystemExit(1)
        best = run_sweep(
            stage=sweep_stage, dataset=ds, scale=scale, num_samples=num_samples,
            max_epochs=max_epochs, patience=patience, warm_start_from=warm_start_from,
        )
        log.info("Sweep complete. Best params: %s", best)
    else:
        log.error("Must specify --stage <name> or --full-pipeline")
        raise SystemExit(1)


@app.command()
def lake(action: str = typer.Option("status", help="rebuild-catalog | verify | status")):
    """Data lake management."""
    from graphids.config import lake_catalog_path, lake_root_from_env

    lake_root = lake_root_from_env()
    if lake_root is None:
        log.error("KD_GAT_LAKE_ROOT not set. Run: export KD_GAT_LAKE_ROOT=/fs/ess/PAS1266/kd-gat")
        raise SystemExit(1)

    if action == "rebuild-catalog":
        from graphids.pipeline.catalog import rebuild_catalog
        log.info("Catalog rebuilt: %s", rebuild_catalog(lake_root))
    elif action == "verify":
        from graphids.pipeline.manifest import verify_manifest
        errors_total = run_count = 0
        for tier_dir in [lake_root / "production", lake_root / "dev"]:
            if not tier_dir.exists():
                continue
            for manifest_file in tier_dir.rglob("_manifest.json"):
                ok, errors = verify_manifest(manifest_file.parent)
                run_count += 1
                if not ok:
                    errors_total += len(errors)
                    log.warning("FAILED: %s — %s", manifest_file.parent, "; ".join(errors))
        log.info("Verified %d runs, %d errors", run_count, errors_total)
    elif action == "status":
        from graphids.pipeline.catalog import catalog_status
        status = catalog_status(lake_catalog_path(lake_root))
        if not status.get("exists"):
            log.info("Catalog not built. Run: python -m graphids.pipeline.cli lake --action rebuild-catalog")
            return
        log.info("Lake: %d runs | stages: %s | datasets: %s",
                 status["total_runs"], status["by_stage"], status["by_dataset"])


@app.command()
def daemon(
    status: bool = False,
    stop: bool = False,
    resubmit: bool = False,
    time_override: Optional[str] = typer.Option(None, "--time", help="Override walltime"),
):
    """Manage Dagster daemon SLURM job."""
    import subprocess

    project_root = Path(__file__).resolve().parent.parent.parent
    connection_file = project_root / ".dagster" / "connection_info.txt"

    if status:
        result = subprocess.run(
            ["squeue", "-u", os.environ["USER"], "-n", "dagster-daemon", "-h", "-o", "%i %T %N %M"],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            log.info("Dagster daemon: %s", result.stdout.strip())
            if connection_file.exists():
                log.info("Connection info:\n%s", connection_file.read_text())
        else:
            log.info("No dagster daemon job running.")
        return

    if stop:
        result = subprocess.run(
            ["squeue", "-u", os.environ["USER"], "-n", "dagster-daemon", "-h", "-o", "%i"],
            capture_output=True, text=True,
        )
        job_id = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
        if job_id:
            subprocess.run(["scancel", job_id], check=True)
            log.info("Cancelled dagster daemon job %s", job_id)
            if connection_file.exists():
                connection_file.unlink()
        else:
            log.info("No dagster daemon job to cancel.")
        return

    cmd = ["bash", str(project_root / "scripts" / "slurm" / "launch_dagster.sh")]
    if resubmit:
        cmd.append("--resubmit")
    if time_override:
        cmd.extend(["--time", time_override])
    subprocess.run(cmd, check=True)


@app.command()
def orchestrate(
    dataset: Optional[str] = None,
    seeds: Optional[str] = None,
    dry_run: bool = False,
):
    """Submit pipeline via fire-and-forget SLURM dependency chains."""
    from .orchestration.dagster_defs import fire_and_forget

    ds = dataset or DEFAULT_DATASET
    seed_list = None
    if seeds:
        from graphids.config import parse_seeds
        seed_list = parse_seeds(seeds)

    job_ids = fire_and_forget(dataset=ds, seeds=seed_list, dry_run=dry_run)
    log.info("Submitted %d jobs:", len(job_ids))
    for name, jid in job_ids.items():
        log.info("  %s: %s", name, jid)


# ---------------------------------------------------------------------------
# Stage lifecycle helpers
# ---------------------------------------------------------------------------


def _archive_previous(sdir: Path) -> Path | None:
    """Archive a completed run directory before re-running."""
    if not (sdir / "config.json").exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = sdir.parent / f"{sdir.name}.archive_{ts}"
    sdir.rename(archive)
    log.warning("Archived completed run → %s", archive)
    return archive


def _log_stage_artifacts(sdir: Path) -> None:
    """Log stage artifacts to MLflow."""
    import mlflow
    for name in ["best_model.pt", "config.json", "embeddings.npz",
                 "attention_weights.npz", "dqn_policy.json", "explanations.npz"]:
        p = sdir / name
        if p.exists():
            mlflow.log_artifact(str(p))


def _write_lake_manifest(cfg: PipelineConfig, stage: str, sdir: Path, metrics: dict | None = None) -> None:
    """Write _manifest.json for the ESS data lake."""
    try:
        from graphids.pipeline.manifest import write_manifest
        write_manifest(
            sdir, dataset=cfg.dataset, model_type=cfg.model_type, scale=cfg.scale,
            stage=stage, auxiliaries=cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
            seed=cfg.seed, metrics=metrics,
        )
    except Exception as e:
        log.warning("Failed to write manifest: %s", e)


# ---------------------------------------------------------------------------
# Single stage execution
# ---------------------------------------------------------------------------


def _run_single_stage(cfg: PipelineConfig, stage: str) -> None:
    """Execute a single training stage with MLflow tracking."""
    validate(cfg, stage)

    sdir = stage_dir(cfg, stage)
    archive = _archive_previous(sdir)

    cfg_out = config_path(cfg, stage)
    cfg.save(cfg_out)

    run_name = run_id(cfg, stage)
    log.info("Run started: %s (seed=%d)", run_name, cfg.seed)

    # Environment metadata
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

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

    import mlflow

    extra_tags = {"slurm_job_id": slurm_job_id or "", "gpu_name": gpu_name or "", "run_type": run_type}
    if teacher_run_id_str:
        extra_tags["teacher_run_id"] = teacher_run_id_str

    t_start = time.monotonic()
    with _setup_mlflow(run_name, cfg, stage, tags=extra_tags):
        try:
            mlflow.log_params({
                "dataset": cfg.dataset, "model_type": cfg.model_type, "scale": cfg.scale,
                "stage": stage, "has_kd": cfg.has_kd, "seed": cfg.seed,
                "batch_size": cfg.training.batch_size, "max_epochs": cfg.training.max_epochs,
                "lr": cfg.training.lr,
            })
            mlflow.log_artifact(str(cfg_out))

            from .stages import STAGE_FNS
            result = STAGE_FNS[stage](cfg)

            duration = time.monotonic() - t_start
            peak_gpu_mb = None
            try:
                import torch
                if torch.cuda.is_available():
                    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
            except Exception:
                pass

            stage_metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            stage_metrics["duration_seconds"] = duration
            if peak_gpu_mb is not None:
                stage_metrics["peak_gpu_mb"] = peak_gpu_mb

            mlflow_metrics = {k: v for k, v in stage_metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(mlflow_metrics)

            _log_stage_artifacts(sdir)
            _write_lake_manifest(cfg, stage, sdir, metrics=stage_metrics)

            if archive and archive.exists():
                import shutil
                shutil.rmtree(archive, ignore_errors=True)

            mlflow.set_tag("status", "success")
            log.info("Stage '%s' complete (%.1fs, peak_gpu=%.0fMB)", stage, duration, peak_gpu_mb or 0.0)

        except Exception as e:
            duration = time.monotonic() - t_start
            if archive and archive.exists():
                if sdir.exists():
                    import shutil
                    shutil.rmtree(sdir, ignore_errors=True)
                archive.rename(sdir)
                log.warning("Restored archive after failure: %s", sdir)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", str(e)[:250])
            mlflow.log_metrics({"duration_seconds": duration})
            log.error("Run failed: %s", str(e))
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SUBCOMMANDS = frozenset({
    "show-config", "preprocess", "sweep", "lake", "daemon", "orchestrate",
})


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )
    args = argv if argv is not None else sys.argv[1:]

    # If first arg is a subcommand or --help, let Typer handle it.
    # Otherwise treat all args as Hydra overrides for training.
    if not args or args[0] in _SUBCOMMANDS or args[0].startswith("-"):
        app(args)
    else:
        _run_training(args)


if __name__ == "__main__":
    main()
