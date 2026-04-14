"""Centralized config constants.

Layer 1 — typed model for ``configs/matrix/axes.json`` plus project-wide
literal constants (filenames, subpaths).

Layer 2 — ``LAKE_ROOT`` alias (reads from ``get_settings().lake_root``).
All ``GRAPHIDS_*`` env vars live in ``graphids.config.settings``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, get_args

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
    """Typed view of ``configs/matrix/axes.json``.

    The JSON top-level has ``axes`` (this model's fields) and an optional
    ``pipeline_defaults`` block alongside it; we load both directly below.
    """

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


# ---------------------------------------------------------------------------
# Load axes.json directly — no wrapper class
# ---------------------------------------------------------------------------

_axes_raw = json.loads((CONFIG_DIR / "matrix" / "axes.json").read_bytes())
AXES: PipelineAxes = PipelineAxes.model_validate(_axes_raw["axes"])
PIPELINE_DEFAULTS: dict[str, Any] = _axes_raw.get("pipeline_defaults", {})

VALID_SCALES: frozenset[str] = frozenset(AXES.scales)
VALID_FUSION_METHODS: frozenset[str] = frozenset(AXES.fusion_methods)
VALID_MODEL_FAMILIES: frozenset[str] = frozenset(AXES.model_families)
VALID_MODEL_TYPES: frozenset[str] = AXES.valid_model_types
FAMILY_FOR_MODEL_TYPE: dict[str, str] = AXES.family_for_model_type

# Static Literal for type-checking (Pydantic needs a concrete Literal for
# field validation). Enforced against axes.json at import: if someone adds
# a model_type to the JSON without updating this line, package import fails.
ModelType = Literal["vgae", "dgi", "gat"]
assert set(get_args(ModelType)) == VALID_MODEL_TYPES, (
    f"ModelType Literal {set(get_args(ModelType))} drifted from axes.json "
    f"model_types {VALID_MODEL_TYPES}; update both to keep them in sync"
)

# ---------------------------------------------------------------------------
# Layer 1 — project constants (no env vars)
# ---------------------------------------------------------------------------

PREPROCESSING_VERSION: str = "8.0.0"
CKPT_SUBPATH: str = "checkpoints/best_model.ckpt"
LAST_CKPT_SUBPATH: str = "checkpoints/last.ckpt"
PHASE_MARKERS: dict[str, str] = {
    "train": ".train_complete",
    "test": ".test_complete",
}
CATALOG_SUBPATH: str = "catalog/graphids.duckdb"
DATASET_REGISTRY_PATH: Path = PROJECT_ROOT / "configs" / "datasets" / "dataset_registry.json"

# ---------------------------------------------------------------------------
# Layer 2 — lake root alias (source of truth: graphids.config.settings)
# ---------------------------------------------------------------------------

from graphids.config.settings import get_settings  # noqa: E402

LAKE_ROOT: str = get_settings().lake_root
