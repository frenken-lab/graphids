"""Single CLI entry point (Typer + Hydra).

Training uses Hydra override grammar via the default command:
    python -m graphids.cli stage=autoencoder model=vgae_large dataset=hcrl_sa

All other subcommands use typed flags:
    python -m graphids.cli sweep --stage autoencoder --num-samples 20
    python -m graphids.cli orchestrate --dataset hcrl_sa --dry-run
    python -m graphids.cli lake --action status
    python -m graphids.cli show-config model=vgae_large dataset=hcrl_sa
"""

from __future__ import annotations

import torch.multiprocessing as mp

# Must be called before any CUDA or multiprocessing usage.
mp.set_start_method("spawn", force=True)

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
import typer

from graphids.config import DEFAULT_DATASET, STAGES, PipelineConfig, run_id
from graphids.storage import open_gateway

from graphids.pipeline.validate import validate

log = structlog.get_logger()

app = typer.Typer(name="graphids", help="KD-GAT pipeline CLI", add_completion=False, no_args_is_help=False)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("show-config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def show_config(ctx: typer.Context):
    """Print resolved config as YAML without running."""
    from graphids.config import compose_config
    from omegaconf import OmegaConf

    print(OmegaConf.to_yaml(compose_config(ctx.args)[0]))


@app.command()
def preprocess(dataset: Optional[str] = None):
    """Build preprocessed graph cache for a dataset."""
    from graphids.config import resolve
    from graphids.core.preprocessing import PreprocessingPipeline

    ds = dataset or DEFAULT_DATASET
    PreprocessingPipeline(resolve("vgae", "large", dataset=ds)).load_dataset()
    log.info("preprocessing_complete", dataset=ds)


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
    from graphids.pipeline.orchestration.optuna_sweep import run_sweep, run_sweep_pipeline

    ds = dataset or DEFAULT_DATASET
    if full_pipeline:
        run_sweep_pipeline(
            dataset=ds, scale=scale, num_samples=num_samples, max_epochs=max_epochs,
            patience=patience, warm_start_from=warm_start_from, dry_run=dry_run,
            multi_seed=multi_seed,
        )
    elif stage:
        from graphids.config import STAGE_MODEL_MAP
        _aliases = {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}
        sweep_stage = _aliases.get(stage, stage)
        if sweep_stage not in STAGE_MODEL_MAP:
            raise SystemExit(f"Invalid stage: {stage}")
        log.info("sweep_complete", best_params=run_sweep(
            stage=sweep_stage, dataset=ds, scale=scale, num_samples=num_samples,
            max_epochs=max_epochs, patience=patience, warm_start_from=warm_start_from,
        ))
    else:
        raise SystemExit("Must specify --stage <name> or --full-pipeline")


@app.command()
def lake(action: str = typer.Option("status", help="rebuild-catalog | verify | status")):
    """Data lake management."""
    from graphids.storage.paths import lake_catalog_path, lake_root_from_env

    lake_root = lake_root_from_env()
    if lake_root is None:
        raise SystemExit("KD_GAT_LAKE_ROOT not set")

    if action == "rebuild-catalog":
        from graphids.storage import rebuild_catalog
        log.info("catalog_rebuilt", result=rebuild_catalog(lake_root))
    elif action == "verify":
        from graphids.storage import verify_all
        runs, errors = verify_all(lake_root)
        log.info("verify_complete", runs=runs, errors=errors)
    elif action == "status":
        from graphids.storage import catalog_status
        status = catalog_status(lake_catalog_path(lake_root))
        log.info("lake_status", **{k: v for k, v in status.items() if k != "exists"}) if status.get("exists") else log.info("catalog_not_built")


@app.command()
def orchestrate(
    dataset: Optional[str] = None,
    seeds: Optional[str] = None,
    dry_run: bool = False,
):
    """Submit pipeline via fire-and-forget SLURM dependency chains."""
    from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

    ds = dataset or DEFAULT_DATASET
    seed_list = None
    if seeds:
        from graphids.config import parse_seeds
        seed_list = parse_seeds(seeds)
    log.info("jobs_submitted", jobs=fire_and_forget(dataset=ds, seeds=seed_list, dry_run=dry_run))


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _init_pipes_context() -> None:
    """Initialize Dagster Pipes context if env vars are present (zero-cost otherwise)."""
    try:
        from dagster_pipes import PipesContext, open_dagster_pipes
        if not PipesContext.is_initialized():
            open_dagster_pipes()
    except (ImportError, Exception):
        pass


def _run_single_stage(cfg: PipelineConfig, stage: str) -> None:
    """Execute a single training stage."""
    _init_pipes_context()
    validate(cfg, stage)
    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed, slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )

    gw, mapper = open_gateway(cfg)
    sdir = gw.resolve(stage)

    # Archive previous completed run
    archive = None
    if (sdir / "config.json").exists():
        archive = sdir.parent / f"{sdir.name}.archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sdir.rename(archive)
        log.warning("run_archived", path=str(archive))

    mapper.save_config(cfg, stage)
    log.info("run_started")

    t_start = time.monotonic()
    try:
        from graphids.pipeline import STAGE_FNS
        result = STAGE_FNS[stage](cfg)
        duration = time.monotonic() - t_start

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        metrics["duration_seconds"] = duration

        # Write manifest (authoritative record of this run)
        from graphids.storage import write_manifest
        try:
            write_manifest(
                sdir, dataset=cfg.dataset, model_type=cfg.model_type, scale=cfg.scale,
                stage=stage, auxiliaries=cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
                seed=cfg.seed, metrics=metrics,
            )
        except Exception as e:
            log.warning("manifest_write_failed", error=str(e))

        # Report to Dagster Pipes if running under orchestration
        try:
            from dagster_pipes import PipesContext, open_dagster_pipes
            if PipesContext.is_initialized():
                pipes = PipesContext.get()
                pipes.report_asset_materialization(
                    metadata={k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                )
        except ImportError:
            pass

        if archive and archive.exists():
            shutil.rmtree(archive, ignore_errors=True)
        log.info("stage_complete", **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    except Exception as e:
        duration = time.monotonic() - t_start
        if archive and archive.exists():
            if sdir.exists():
                shutil.rmtree(sdir, ignore_errors=True)
            archive.rename(sdir)
        log.error("stage_failed", error=str(e)[:250], duration_s=round(duration, 1))
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SUBCOMMANDS = frozenset({"show-config", "preprocess", "sweep", "lake", "orchestrate"})


def main(argv: list[str] | None = None) -> None:
    from graphids.logging import configure_logging

    args = argv if argv is not None else sys.argv[1:]
    json_logs = "--json-logs" in args
    if json_logs:
        args = [a for a in args if a != "--json-logs"]
    configure_logging(json=json_logs or None)

    if not args or args[0] in _SUBCOMMANDS or args[0].startswith("-"):
        app(args)
    else:
        # Hydra override grammar for training
        from graphids.config import compose_config
        from omegaconf import OmegaConf

        merged, stage = compose_config(args)
        if stage is None or stage not in STAGES:
            raise SystemExit(f"Unknown stage: {stage}. Valid: {list(STAGES.keys())}")
        raw = OmegaConf.to_object(merged)
        raw.pop("stage", None)
        _run_single_stage(PipelineConfig.model_validate(raw), stage)


if __name__ == "__main__":
    main()
