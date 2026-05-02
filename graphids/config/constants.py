"""Project-wide path constants and filename literals.

`GRAPHIDS_*` env vars are read directly from `os.environ` at the call
sites that need them. `lake_root()` and `_run_root()` in
:mod:`graphids.config.catalog` are the two shared readers. This module
is import-safe with no external deps so it can be loaded from anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Concrete Literal — Pydantic needs this for field validation. Each model
# module annotates its `model_type` arg with this so the rendered_config's
# value is checked at instantiation. (Fusion methods aren't model_types in
# this sense; their identity is `model_type='fusion'` + a `method` field.)
ModelType = Literal["vgae", "dgi", "gat", "fusion"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATASET_REGISTRY_PATH: Path = CONFIG_DIR / "data" / "datasets.json"

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
