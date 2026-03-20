"""Stage executor: single entry point for ALL stage execution.

Every path through the pipeline (CLI, API, submitit, notebook) calls
execute_stage(). Cross-cutting concerns live here, nowhere else.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from graphids.config import PipelineConfig
from graphids.logging import configure_logging
from graphids.pipeline.validate import validate

log = structlog.get_logger()


@dataclass(frozen=True)
class StageResult:
    metrics: dict[str, float]
    duration_seconds: float
    checkpoint_path: Path | None
    manifest_path: Path


def execute_stage(cfg: PipelineConfig, stage: str) -> StageResult:
    """Execute a pipeline stage with full guarantees.

    Owns: logging setup, validation, structlog context, archive/restore,
    config snapshot, timing, manifest write. Stage functions do the ML work.
    """
    configure_logging()
    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    validate(cfg, stage)

    sdir = Path.cwd()
    cfg.save(sdir / "config.json")
    log.info("run_started")
    t0 = time.monotonic()

    try:
        from graphids.pipeline import STAGE_FNS

        result = STAGE_FNS[stage](cfg)
        duration = time.monotonic() - t0

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        metrics["duration_seconds"] = duration

        ckpt = sdir / "best_model.pt"
        log.info("stage_complete", **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        return StageResult(metrics, duration, ckpt if ckpt.exists() else None, sdir / "_manifest.json")

    except Exception:
        raise
