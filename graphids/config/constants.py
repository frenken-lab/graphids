"""Project-wide path constants, filename literals, and the ModelType drift guard.

``LAKE_ROOT`` is an alias over ``get_settings().lake_root``; all other
``GRAPHIDS_*`` env vars live in :mod:`graphids.config.settings`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, get_args

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATASET_REGISTRY_PATH: Path = CONFIG_DIR / "datasets" / "dataset_registry.json"

# ---------------------------------------------------------------------------
# Filename / subpath literals
# ---------------------------------------------------------------------------

PREPROCESSING_VERSION: str = "10.0.0"
CKPT_SUBPATH: str = "checkpoints/best_model.ckpt"
LAST_CKPT_SUBPATH: str = "checkpoints/last.ckpt"
PHASE_MARKERS: dict[str, str] = {
    "train": ".train_complete",
    "test": ".test_complete",
}

# ---------------------------------------------------------------------------
# ModelType — Pydantic needs a concrete Literal for field validation, so we
# can't derive it from JSON. Drift guard below fails at import if the Literal
# falls out of sync with axes.json's model_types_by_family (excluding fusion).
# ---------------------------------------------------------------------------

ModelType = Literal["vgae", "dgi", "gat"]

_axes_types = frozenset(
    t
    for fam, types in json.loads((CONFIG_DIR / "matrix" / "axes.json").read_bytes())["axes"][
        "model_types_by_family"
    ].items()
    if fam != "fusion"
    for t in types
)
assert set(get_args(ModelType)) == _axes_types, (
    f"ModelType Literal {set(get_args(ModelType))} drifted from axes.json "
    f"{_axes_types}; update both to keep them in sync"
)

# ---------------------------------------------------------------------------
# Path-root aliases (source of truth: graphids.config.settings)
# ---------------------------------------------------------------------------

from graphids.config.settings import get_settings  # noqa: E402

LAKE_ROOT: str = get_settings().lake_root
RUN_ROOT: str = get_settings().run_root
