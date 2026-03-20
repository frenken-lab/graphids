"""Pipeline stages: dispatch + run.

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
    """Validate, bind context, save config, run stage function."""
    from graphids.pipeline.validate import validate

    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    validate(cfg, stage)
    cfg.save(Path.cwd() / "config.json")
    return STAGE_FNS[stage](cfg)
