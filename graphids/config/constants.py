"""Centralized config constants.

Layer 1 — project-wide literal constants (filenames, subpaths, topology
keys loaded from ``configs/matrix/axes.json``).

Layer 2 — the single lake-root env var. Every other env var that used
to live here has moved to the module that reads it:

- ``SLURM_ACCOUNT`` / ``SLURM_LOG_DIR`` → ``graphids.slurm.env``
- ``BUDGET_*``                          → ``graphids.core.data.budget``
- ``WANDB_WRITE_DIR``                   → ``graphids.core.instantiate``

``LAKE_ROOT`` stays because dozens of call sites read it; moving it is
its own refactor pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

_axes_file = json.loads((CONFIG_DIR / "matrix" / "axes.json").read_text())
_axes = _axes_file.get("axes", {})

# Pipeline-wide defaults — single source of truth for CLI, PipelineConfig, jsonnet TLAs.
PIPELINE_DEFAULTS: dict[str, object] = _axes_file.get("pipeline_defaults", {})
VALID_SCALES: frozenset[str] = frozenset(_axes.get("scales", ["small", "large"]))
VALID_FUSION_METHODS: frozenset[str] = frozenset(_axes.get("fusion_methods", []))

# Model families = organizational units (unsupervised, supervised, fusion)
VALID_MODEL_FAMILIES: frozenset[str] = frozenset(_axes.get("model_families", []))

# Model types = architecture dispatch keys (vgae, dgi, gat)
_types_by_family: dict[str, list[str]] = _axes.get("model_types_by_family", {})
VALID_MODEL_TYPES: frozenset[str] = frozenset(
    t for fam, types in _types_by_family.items() if fam != "fusion" for t in types
)
FAMILY_FOR_MODEL_TYPE: dict[str, str] = {
    t: fam for fam, types in _types_by_family.items() for t in types
}

# ---------------------------------------------------------------------------
# Layer 1 — project constants (no env vars)
# ---------------------------------------------------------------------------

PREPROCESSING_VERSION: str = "8.0.0"
MAX_DATA_BYTES: int = 8
CKPT_SUBPATH: str = "checkpoints/best_model.ckpt"
LAST_CKPT_SUBPATH: str = "checkpoints/last.ckpt"
COMPLETE_MARKER: str = ".complete"
PHASE_MARKERS: dict[str, str] = {
    "train": ".train_complete",
    "test": ".test_complete",
    "analyze": ".analyze_complete",
}
RUN_RECORD_FILENAME: str = "run_record.json"
DAGSTER_IO_DIR_TEMPLATE: str = "{lake_root}/.dagster/io"
CATALOG_SUBPATH: str = "catalog/kd_gat.duckdb"
DATASET_REGISTRY_PATH: Path = PROJECT_ROOT / "configs" / "datasets" / "dataset_registry.json"

# ---------------------------------------------------------------------------
# Layer 2 — lake root (single remaining env var — broad reach, kept here)
# ---------------------------------------------------------------------------

LAKE_ROOT: str = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
