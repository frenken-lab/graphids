"""Artifact store: ESS filesystem lookups for cross-stage artifact resolution.

Provides get/exists for cross-stage artifact reads (e.g. loading VGAE checkpoint
while training GAT). Same-stage writes use stage_dir() directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphids.config.schema import PipelineConfig


def _resolve_path(
    cfg: PipelineConfig, stage: str, artifact_name: str, model_type: str
) -> Path:
    """Build filesystem path, handling cross-model lookups."""
    from graphids.config import stage_dir

    if model_type != cfg.model_type:
        sd = stage_dir(cfg.model_copy(update={"model_type": model_type}), stage)
    else:
        sd = stage_dir(cfg, stage)
    return sd / artifact_name


def get_artifact(
    cfg: PipelineConfig,
    stage: str,
    artifact_name: str,
    model_type: str | None = None,
) -> Path:
    """Get artifact path from ESS filesystem. Raises FileNotFoundError if missing.

    For cross-model reads (e.g. loading VGAE from GAT config), pass
    model_type to override cfg.model_type.
    """
    mt = model_type or cfg.model_type
    path = _resolve_path(cfg, stage, artifact_name, mt)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return path


def artifact_exists(
    cfg: PipelineConfig,
    stage: str,
    artifact_name: str,
    model_type: str | None = None,
) -> bool:
    """Check if artifact exists on ESS filesystem."""
    mt = model_type or cfg.model_type
    return _resolve_path(cfg, stage, artifact_name, mt).exists()
