"""Pipeline stages: dispatch and run.

Public API:
    from graphids.pipeline.stages import run_stage
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from .evaluation import evaluate
from .fusion import train_fusion
from .temporal import train_temporal
from .training import train_autoencoder, train_curriculum, train_normal

STAGE_FNS = {
    "autoencoder": train_autoencoder,
    "curriculum":  train_curriculum,
    "normal":      train_normal,
    "fusion":      train_fusion,
    "evaluation":  evaluate,
    "temporal":    train_temporal,
}


def run_stage(cfg, stage: str) -> dict:
    """Bind context, chdir to run directory, save config, run stage function.

    Output directory mirrors Hydra's run.dir pattern:
      {lake_root}/{tier}/{dataset}/{model_type}_{scale}_{stage}/seed_{seed}
    All Lightning outputs (checkpoints, logs, metrics) land in the data lake,
    not the project directory.
    """
    from omegaconf import OmegaConf

    from graphids.config import STAGES

    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    # Resolve output directory from config (same interpolation as hydra.run.dir)
    tier = f"dev/{os.environ.get('USER', 'unknown')}"
    production = cfg.get("production", False)
    if production:
        tier = "production"
    run_dir = Path(cfg.lake_root) / tier / cfg.dataset / f"{cfg.model_type}_{cfg.scale}_{stage}" / f"seed_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        run_dir=str(run_dir),
    )
    OmegaConf.save(cfg, run_dir / "config.yaml")
    return STAGE_FNS[stage](cfg)
