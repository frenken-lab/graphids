"""Single CLI entry point: Hydra-as-framework for training/sweep, argparse for the rest.

Usage:
    python -m graphids stage=autoencoder model=vgae_large dataset=hcrl_sa
    python -m graphids --multirun stage=autoencoder model=vgae_large  # HPO sweep
    python -m graphids orchestrate --dataset hcrl_sa [--seeds 42,123] [--dry-run]
    python -m graphids lake --action status|rebuild-catalog|verify
    python -m graphids preprocess --dataset hcrl_sa
    python -m graphids --cfg job model=vgae_large  # show resolved config
"""

from __future__ import annotations

import sys

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

_SUBCOMMANDS = {"orchestrate", "lake", "preprocess"}


def main(argv: list[str] | None = None) -> None:
    from graphids.logging import configure_logging

    args = argv if argv is not None else sys.argv[1:]

    json_logs = "--json-logs" in args
    if json_logs:
        args = [a for a in args if a != "--json-logs"]
    configure_logging(json=json_logs or None)

    if args and args[0] in _SUBCOMMANDS:
        cmd, rest = args[0], args[1:]
        sys.argv = [sys.argv[0]] + rest
        if cmd == "orchestrate":
            _orchestrate()
        elif cmd == "lake":
            _lake()
        elif cmd == "preprocess":
            _preprocess()
    else:
        # Hydra path: training (default) or sweep (--multirun)
        sys.argv = [sys.argv[0]] + args
        _hydra_main()


# ---------------------------------------------------------------------------
# Hydra entry point (training + sweep share the same function)
# ---------------------------------------------------------------------------

def _hydra_main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from graphids.config import STAGES, PipelineConfig
    from graphids.pipeline.executor import execute_stage

    @hydra.main(config_path="config/conf", config_name="config", version_base="1.3")
    def run(cfg: DictConfig) -> float | None:
        stage = cfg.get("stage")
        if not stage or stage not in STAGES:
            raise SystemExit(f"stage= required. Valid: {list(STAGES.keys())}")

        raw = OmegaConf.to_object(cfg)
        raw.pop("stage", None)
        pcfg = PipelineConfig.model_validate(raw)

        result = execute_stage(pcfg, stage)
        return result.metrics.get("val_loss", float("inf"))

    run()


# ---------------------------------------------------------------------------
# Argparse subcommands (no Hydra)
# ---------------------------------------------------------------------------

def _orchestrate() -> None:
    import argparse

    import structlog

    from graphids.config import DEFAULT_DATASET, parse_seeds, resolve
    from graphids.pipeline.orchestration.dag import build_dag_topology, run_dag
    from graphids.pipeline.orchestration.slurm import make_slurm_executor

    log = structlog.get_logger()
    parser = argparse.ArgumentParser(description="Submit pipeline DAG to SLURM")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_list = parse_seeds(args.seeds) if args.seeds else [resolve("vgae", "large").seed]
    dag = build_dag_topology()
    futures = run_dag(
        executor_factory=lambda r, deps: make_slurm_executor(r, dep_futures=deps),
        dag=dag, dataset=args.dataset, seeds=seed_list, dry_run=args.dry_run,
    )
    log.info("jobs_submitted", count=len(futures))


def _lake() -> None:
    import argparse

    import structlog

    log = structlog.get_logger()
    parser = argparse.ArgumentParser(description="Data lake management")
    parser.add_argument("--action", default="status", choices=["status", "rebuild-catalog", "verify"])
    args = parser.parse_args()

    from graphids.storage.paths import lake_catalog_path, lake_root_from_env

    lake_root = lake_root_from_env()
    if lake_root is None:
        raise SystemExit("KD_GAT_LAKE_ROOT not set")

    if args.action == "rebuild-catalog":
        from graphids.storage import rebuild_catalog
        log.info("catalog_rebuilt", result=rebuild_catalog(lake_root))
    elif args.action == "verify":
        from graphids.storage import verify_all
        runs, errors = verify_all(lake_root)
        log.info("verify_complete", runs=runs, errors=errors)
    elif args.action == "status":
        from graphids.storage import catalog_status
        status = catalog_status(lake_catalog_path(lake_root))
        if status.get("exists"):
            log.info("lake_status", **{k: v for k, v in status.items() if k != "exists"})
        else:
            log.info("catalog_not_built")


def _preprocess() -> None:
    import argparse

    import structlog

    from graphids.config import DEFAULT_DATASET, resolve

    log = structlog.get_logger()
    parser = argparse.ArgumentParser(description="Build preprocessed graph cache")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    from graphids.core.preprocessing import PreprocessingPipeline

    PreprocessingPipeline(resolve("vgae", "large", dataset=args.dataset)).load_dataset()
    log.info("preprocessing_complete", dataset=args.dataset)


if __name__ == "__main__":
    main()
