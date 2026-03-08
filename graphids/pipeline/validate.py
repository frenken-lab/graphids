"""Config validation. Catches mistakes before they become 6-hour SLURM failures.

Most field-level checks are now handled by Pydantic Field() constraints.
This module handles filesystem checks and cross-stage prerequisite checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphids.config import PipelineConfig

from graphids.config import STAGES, data_dir, get_datasets
from graphids.config.constants import STAGE_DEPENDENCIES

_log = logging.getLogger(__name__)


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

    if cfg.has_kd and not cfg.kd.model_path and stage != "evaluation":
        errors.append("KD auxiliary enabled but model_path is empty")

    if cfg.has_kd and cfg.kd.model_path:
        tp = Path(cfg.kd.model_path)
        if not tp.exists():
            errors.append(f"Teacher checkpoint not found: {tp}")
        teacher_cfg = tp.parent / "config.json"
        if not teacher_cfg.exists():
            errors.append(f"Teacher config not found: {teacher_cfg}")

    # --- prerequisite checkpoints + frozen configs ---
    if stage in STAGE_DEPENDENCIES:
        exp = Path(cfg.experiment_root) / cfg.dataset
        aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
        for model_type, prereq_stage in STAGE_DEPENDENCIES[stage]:
            base = exp / f"{model_type}_{cfg.scale}_{prereq_stage}{aux_suffix}"
            if not (base / "best_model.pt").exists():
                errors.append(f"{stage} needs {prereq_stage} checkpoint: {base / 'best_model.pt'}")
            if not (base / "config.json").exists():
                errors.append(f"{stage} needs {prereq_stage} config: {base / 'config.json'}")

    if stage == "evaluation":
        exp = Path(cfg.experiment_root) / cfg.dataset
        aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
        vgae_ckpt = exp / f"vgae_{cfg.scale}_autoencoder{aux_suffix}" / "best_model.pt"
        gat_ckpt = exp / f"gat_{cfg.scale}_curriculum{aux_suffix}" / "best_model.pt"
        if not gat_ckpt.exists() and not vgae_ckpt.exists():
            errors.append("Evaluation needs at least one checkpoint (GAT or VGAE)")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))
