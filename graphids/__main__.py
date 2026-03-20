"""Single CLI entry point: Hydra for training/sweep, argparse for orchestrate.

Usage:
    python -m graphids stage=autoencoder model=vgae_large dataset=hcrl_sa
    python -m graphids --multirun stage=autoencoder model=vgae_large
    python -m graphids orchestrate --dataset hcrl_sa [--seeds 42,123] [--dry-run]
    python -m graphids --cfg job model=vgae_large
"""

from __future__ import annotations

import sys

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


def main(argv: list[str] | None = None) -> None:
    from graphids.logging import configure_logging

    args = argv if argv is not None else sys.argv[1:]

    json_logs = "--json-logs" in args
    if json_logs:
        args = [a for a in args if a != "--json-logs"]
    configure_logging(json=json_logs or None)

    cmd = args[0] if args else None

    if cmd == "orchestrate":
        import argparse

        import structlog

        from graphids.config import DEFAULT_DATASET, parse_seeds, resolve
        from graphids.pipeline.dag import build_dag_topology, run_dag

        sys.argv = [sys.argv[0]] + args[1:]
        parser = argparse.ArgumentParser(description="Submit pipeline DAG to SLURM")
        parser.add_argument("--dataset", default=DEFAULT_DATASET)
        parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
        parser.add_argument("--dry-run", action="store_true")
        parsed = parser.parse_args()

        seed_list = parse_seeds(parsed.seeds) if parsed.seeds else [resolve("vgae", "large").seed]
        futures = run_dag(dag=build_dag_topology(), dataset=parsed.dataset, seeds=seed_list, dry_run=parsed.dry_run)
        structlog.get_logger().info("jobs_submitted", count=len(futures))

    else:
        # Hydra path: training (default) or sweep (--multirun)
        import hydra
        from omegaconf import DictConfig, OmegaConf

        from graphids.config import STAGES, PipelineConfig
        from graphids.pipeline.stages import run_stage

        sys.argv = [sys.argv[0]] + args

        @hydra.main(config_path="config/conf", config_name="config", version_base="1.3")
        def run(cfg: DictConfig) -> float | None:
            stage = cfg.get("stage")
            if not stage or stage not in STAGES:
                raise SystemExit(f"stage= required. Valid: {list(STAGES.keys())}")

            raw = OmegaConf.to_object(cfg)
            raw.pop("stage", None)
            pcfg = PipelineConfig.model_validate(raw)

            result = run_stage(pcfg, stage)
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            return metrics.get("val_loss", float("inf"))

        run()


if __name__ == "__main__":
    main()
