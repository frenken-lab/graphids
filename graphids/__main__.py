"""CLI entry point: Hydra-as-framework for training and sweep.

Usage:
    python -m graphids stage=autoencoder model_type=vgae scale=large dataset=hcrl_sa
    python -m graphids --multirun stage=autoencoder model_type=vgae scale=large training.lr=0.001,0.01
    python -m graphids stage=autoencoder model_type=vgae scale=large --cfg job

Orchestration (submit full DAG to SLURM):
    python -m graphids.pipeline.dag --dataset hcrl_sa --seeds 42,123 --dry-run
"""

from __future__ import annotations

import sys

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


def main(argv: list[str] | None = None) -> None:
    import hydra
    from omegaconf import DictConfig

    from graphids.config import STAGES, _merge_model_preset
    from graphids.logging import configure_logging
    from graphids.pipeline.stages import run_stage

    args = argv if argv is not None else sys.argv[1:]

    json_logs = "--json-logs" in args
    if json_logs:
        args = [a for a in args if a != "--json-logs"]
    configure_logging(json=json_logs or None)

    sys.argv = [sys.argv[0]] + args

    @hydra.main(config_path="config", config_name="config", version_base="1.3")
    def run(cfg: DictConfig) -> float | None:
        cfg = _merge_model_preset(cfg)

        stage = cfg.get("stage")
        if not stage or stage not in STAGES:
            raise SystemExit(f"stage= required. Valid: {list(STAGES.keys())}")

        result = run_stage(cfg, stage)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        return metrics.get("val_loss", float("inf"))

    run()


if __name__ == "__main__":
    main()
