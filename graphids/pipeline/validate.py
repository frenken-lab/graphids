"""Config validation. Catches mistakes before they become 6-hour SLURM failures.

Most field-level checks are now handled by Pydantic Field() constraints.
This module handles filesystem checks and cross-stage prerequisite checks.
All artifact lookups go through the ArtifactResolver (cache → legacy → MLflow).
"""

from __future__ import annotations

import structlog
from pathlib import Path


from graphids.config import STAGE_DEPENDENCIES, STAGES, data_dir, get_datasets, resolve

log = structlog.get_logger()


def validate_datasets(datasets: list[str], scale: str) -> list[str]:
    """Validate that datasets resolve and have data directories.

    Returns a list of error strings (empty if all OK).
    """

    errors: list[str] = []
    for dataset in datasets:
        try:
            cfg = resolve("model_type=vgae", f"scale={scale}", f"dataset={dataset}")
            ddir = data_dir(cfg)
            if not ddir.exists():
                errors.append(f"Data dir missing for {dataset}: {ddir}")
        except Exception as e:
            errors.append(f"Config resolution failed for {dataset}: {e}")
    return errors


def _artifact_exists(cfg, stage: str, name: str, model_type: str) -> bool:
    """Check if a stage artifact exists via checkpoint paths."""
    from pathlib import Path
    return Path(cfg.checkpoints[model_type]).exists()


def validate(cfg, stage: str) -> None:
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
    kd = next((a for a in cfg.get("auxiliaries", []) if a.type == "kd"), None)
    if cfg.scale == "small" and not kd:
        log.warning("Small model without KD -- running as ablation baseline")

    if kd and kd.model_path:
        tp = Path(kd.model_path)
        if not tp.exists():
            errors.append(f"Teacher checkpoint not found: {tp}")
        teacher_cfg = tp.parent / "config.yaml"
        if not teacher_cfg.exists():
            errors.append(f"Teacher config not found: {teacher_cfg}")
    elif kd and not kd.model_path and stage != "evaluation":
        log.info("kd_teacher_auto_resolve", teacher_scale=kd.teacher_scale)

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
