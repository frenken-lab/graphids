"""Centralized config constants.

Layer 1 — typed model for ``configs/matrix/axes.json`` plus project-wide
literal constants (filenames, subpaths).

Layer 2 — ``LAKE_ROOT`` alias (reads from ``get_settings().lake_root``).
All ``KD_GAT_*`` env vars live in ``graphids.config.settings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field

# ---------------------------------------------------------------------------
# Paths (no dependencies)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------------
# Pydantic model for axes.json
# ---------------------------------------------------------------------------


class PipelineAxes(BaseModel):
    """Typed view of the ``axes`` block in ``axes.json``."""

    model_config = ConfigDict(frozen=True)

    datasets: list[str]
    scales: list[str]
    model_families: list[str]
    model_types_by_family: dict[str, list[str]]
    fusion_methods: list[str]
    stages_by_family: dict[str, list[str]] = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid_model_types(self) -> frozenset[str]:
        """Architecture dispatch keys (excludes fusion family)."""
        return frozenset(
            t for fam, types in self.model_types_by_family.items() if fam != "fusion" for t in types
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def family_for_model_type(self) -> dict[str, str]:
        return {t: fam for fam, types in self.model_types_by_family.items() for t in types}


class AxesConfig(BaseModel):
    """Typed view of ``configs/matrix/axes.json``."""

    model_config = ConfigDict(frozen=True)

    pipeline_defaults: dict[str, Any] = {}
    axes: PipelineAxes


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------

AXES = AxesConfig.model_validate_json((CONFIG_DIR / "matrix" / "axes.json").read_bytes())

# ---------------------------------------------------------------------------
# Backward-compat module-level names
# ---------------------------------------------------------------------------

PIPELINE_DEFAULTS: dict[str, Any] = AXES.pipeline_defaults
VALID_SCALES: frozenset[str] = frozenset(AXES.axes.scales)
VALID_FUSION_METHODS: frozenset[str] = frozenset(AXES.axes.fusion_methods)
VALID_MODEL_FAMILIES: frozenset[str] = frozenset(AXES.axes.model_families)
VALID_MODEL_TYPES: frozenset[str] = AXES.axes.valid_model_types
FAMILY_FOR_MODEL_TYPE: dict[str, str] = AXES.axes.family_for_model_type

# Static Literal for type-checking; keep in sync with axes.json model_types.
ModelType = Literal["vgae", "dgi", "gat"]

# ---------------------------------------------------------------------------
# Layer 1 — project constants (no env vars)
# ---------------------------------------------------------------------------

PREPROCESSING_VERSION: str = "8.0.0"
CKPT_SUBPATH: str = "checkpoints/best_model.ckpt"
LAST_CKPT_SUBPATH: str = "checkpoints/last.ckpt"
COMPLETE_MARKER: str = ".complete"
PHASE_MARKERS: dict[str, str] = {
    "train": ".train_complete",
    "test": ".test_complete",
    "analyze": ".analyze_complete",
}
DAGSTER_IO_DIR_TEMPLATE: str = "{lake_root}/.dagster/io"
CATALOG_SUBPATH: str = "catalog/kd_gat.duckdb"
DATASET_REGISTRY_PATH: Path = PROJECT_ROOT / "configs" / "datasets" / "dataset_registry.json"

# ---------------------------------------------------------------------------
# Layer 2 — lake root alias (source of truth: graphids.config.settings)
# ---------------------------------------------------------------------------

from graphids.config.settings import get_settings  # noqa: E402

LAKE_ROOT: str = get_settings().lake_root
