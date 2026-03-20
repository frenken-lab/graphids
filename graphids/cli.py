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

import sys
from typing import Optional

import structlog
import typer

from graphids.config import DEFAULT_DATASET, STAGES, PipelineConfig
from graphids.pipeline.executor import execute_stage

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
    from graphids.config import parse_seeds, resolve
    from graphids.pipeline.orchestration.dag import build_dag_topology, run_dag
    from graphids.pipeline.orchestration.slurm import make_slurm_executor

    ds = dataset or DEFAULT_DATASET
    seed_list = parse_seeds(seeds) if seeds else [resolve("vgae", "large").seed]
    dag = build_dag_topology()
    futures = run_dag(
        executor_factory=lambda r, deps: make_slurm_executor(r, dep_futures=deps),
        dag=dag, dataset=ds, seeds=seed_list, dry_run=dry_run,
    )
    log.info("jobs_submitted", count=len(futures))


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
        execute_stage(PipelineConfig.model_validate(raw), stage)


if __name__ == "__main__":
    main()
