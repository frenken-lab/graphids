"""Stage executor: single entry point for ALL stage execution.

Every path through the pipeline (CLI, API, submitit, notebook) calls
execute_stage(). Cross-cutting concerns live here, nowhere else.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

from graphids.config import PipelineConfig
from graphids.logging import configure_logging
from graphids.pipeline.validate import validate
from graphids.storage import open_gateway, write_manifest

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

    gw, mapper = open_gateway(cfg)
    sdir = gw.resolve(stage)

    # Archive previous run (restore on failure)
    archive = None
    if (sdir / "config.json").exists():
        archive = sdir.parent / f"{sdir.name}.archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sdir.rename(archive)
        log.warning("run_archived", path=str(archive))

    mapper.save_config(cfg, stage)
    log.info("run_started")
    t0 = time.monotonic()

    try:
        from graphids.pipeline import STAGE_FNS

        result = STAGE_FNS[stage](cfg)
        duration = time.monotonic() - t0

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        metrics["duration_seconds"] = duration

        manifest_path = sdir / "_manifest.json"
        try:
            write_manifest(
                sdir, dataset=cfg.dataset, model_type=cfg.model_type,
                scale=cfg.scale, stage=stage,
                auxiliaries=cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
                seed=cfg.seed, metrics=metrics,
            )
        except Exception as e:
            log.warning("manifest_write_failed", error=str(e))

        if archive and archive.exists():
            shutil.rmtree(archive, ignore_errors=True)

        ckpt = sdir / "best_model.pt"
        log.info("stage_complete", **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        return StageResult(metrics, duration, ckpt if ckpt.exists() else None, manifest_path)

    except Exception:
        if archive and archive.exists():
            if sdir.exists():
                shutil.rmtree(sdir, ignore_errors=True)
            archive.rename(sdir)
        raise
