"""Pipeline orchestration — planning, resolution, execution.

Module layout (Layer 0 → 5, each layer depends only on layers below):

- ``config.py``        (Layer 0) — frozen data types: PipelineConfig,
                         StageConfig, TrainingRunConfig, KDEntry,
                         ResolvedConfig, InstantiatedRun, PipelineResult.
- ``planning.py``      (Layer 1) — build_pipeline_stages,
                         resolve_jsonnet_path. Pure data.
- ``resolve.py``       (Layer 2) — resolve_config (StageConfig → ResolvedConfig).
                         CLI callers go through ``ResolvedConfig.from_rendered``.
- ``instantiate.py``   (Layer 3) — build_run, build_model, build_datamodule,
                         build_trainer, build_callbacks, build_loggers.
- ``stage.py``         (Layer 4) — build, train, evaluate primitives.
- ``run.py``           (Layer 5) — run_pipeline, _run_one_stage.
"""

from __future__ import annotations

from graphids.orchestrate.config import (
    InstantiatedRun,
    KDEntry,
    PipelineConfig,
    PipelineResult,
    ResolvedConfig,
    StageConfig,
    TrainingRunConfig,
)
from graphids.orchestrate.instantiate import build_run
from graphids.orchestrate.planning import build_pipeline_stages
from graphids.orchestrate.resolve import resolve_config
from graphids.orchestrate.run import run_pipeline

__all__ = [
    "InstantiatedRun",
    "KDEntry",
    "PipelineConfig",
    "PipelineResult",
    "ResolvedConfig",
    "StageConfig",
    "TrainingRunConfig",
    "build_pipeline_stages",
    "build_run",
    "resolve_config",
    "run_pipeline",
]
