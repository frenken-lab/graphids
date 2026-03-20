"""Stage executor: shared entry point for CLI and orchestrator.

Owns: validation, structlog context, config snapshot.
Logging is configured by the caller (__main__.py or SLURM preamble).
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from graphids.config import PipelineConfig
from graphids.pipeline.validate import validate

log = structlog.get_logger()


def execute_stage(cfg: PipelineConfig, stage: str) -> dict:
    """Validate, bind context, save config, run stage function.

    Returns the stage function's result dict (typically {"metrics": {...}}).
    """
    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    validate(cfg, stage)
    cfg.save(Path.cwd() / "config.json")

    from graphids.pipeline import STAGE_FNS

    return STAGE_FNS[stage](cfg)
