"""Config validation. Catches mistakes before they become 6-hour SLURM failures.

Most field-level checks are now handled by Pydantic Field() constraints.
This module handles filesystem checks and cross-stage prerequisite checks.
All artifact lookups go through the ArtifactResolver (cache → legacy → MLflow).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphids.config import PipelineConfig

from graphids.config import STAGE_DEPENDENCIES, STAGES, data_dir, get_datasets

_log = logging.getLogger(__name__)


def validate_datasets(datasets: list[str], scale: str) -> list[str]:
    """Validate that datasets resolve and have data directories.

    Returns a list of error strings (empty if all OK).
    """
    from graphids.config import resolve

    errors: list[str] = []
    for dataset in datasets:
        try:
            cfg = resolve("vgae", scale, dataset=dataset)
            ddir = data_dir(cfg)
            if not ddir.exists():
                errors.append(f"Data dir missing for {dataset}: {ddir}")
        except Exception as e:
            errors.append(f"Config resolution failed for {dataset}: {e}")
    return errors


def _artifact_exists(cfg: PipelineConfig, stage: str, name: str, model_type: str) -> bool:
    """Check if a stage artifact exists via the resolver (cache → legacy → filesystem)."""
    from graphids.pipeline.artifacts import artifact_exists

    return artifact_exists(cfg, stage, name, model_type=model_type)


def validate(cfg: PipelineConfig, stage: str) -> None:
    """Raise ValueError if config + stage combination is invalid."""
    errors: list[str] = []

    # --- basic checks ---
    if stage not in STAGES:
        errors.append(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    if cfg.dataset not in get_datasets():
        errors.append(f"Unknown dataset '{cfg.dataset}'. Choose from: {get_datasets()}")

    if not data_dir(cfg).exists():
        errors.append(f"Data directory not found: {data_dir(cfg)}")

    # --- KD consistency ---
    if cfg.scale == "small" and not cfg.has_kd:
        _log.warning("Small model without KD -- running as ablation baseline")

    if cfg.has_kd and cfg.kd.model_path:
        # Explicit teacher path — validate it exists
        tp = Path(cfg.kd.model_path)
        if not tp.exists():
            errors.append(f"Teacher checkpoint not found: {tp}")
        teacher_cfg = tp.parent / "config.json"
        if not teacher_cfg.exists():
            errors.append(f"Teacher config not found: {teacher_cfg}")
    elif cfg.has_kd and not cfg.kd.model_path and stage != "evaluation":
        # Auto-resolution via prepare_kd() at training time
        _log.info("KD teacher will auto-resolve from scale='%s'", cfg.kd.teacher_scale)

    # --- prerequisite checkpoints (via resolver, not hardcoded paths) ---
    if stage in STAGE_DEPENDENCIES:
        for model_type, prereq_stage in STAGE_DEPENDENCIES[stage]:
            if not _artifact_exists(cfg, prereq_stage, "best_model.pt", model_type):
                errors.append(
                    f"{stage} needs {prereq_stage}/{model_type} checkpoint (not found via resolver)"
                )
            if not _artifact_exists(cfg, prereq_stage, "config.json", model_type):
                errors.append(
                    f"{stage} needs {prereq_stage}/{model_type} config (not found via resolver)"
                )

    # Evaluation: needs at least one of GAT or VGAE (graceful degradation)
    if stage == "evaluation":
        has_vgae = _artifact_exists(cfg, "autoencoder", "best_model.pt", "vgae")
        has_gat = _artifact_exists(cfg, "curriculum", "best_model.pt", "gat")
        if not has_gat and not has_vgae:
            errors.append("Evaluation needs at least one checkpoint (GAT or VGAE)")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))
