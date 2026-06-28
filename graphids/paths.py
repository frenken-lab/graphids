"""Project-wide path helpers + dataset registry.

This module still owns:
- dataset registry lookup (``load_catalog``, ``dataset_names``)
- raw / cache paths under ``$GRAPHIDS_LAKE_ROOT`` (``data_dir``, ``cache_dir``)
- run-root helpers for local and SLURM launches

Import-safe: no torch imports, so this stays usable from login-node code
paths and config composition.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Concrete Literal — Pydantic needs this for field validation.
ModelType = Literal["temporal_event_classifier", "temporal_gat", "temporal_vgae"]


# ---------------------------------------------------------------------------
# Static path roots / filename literals
# ---------------------------------------------------------------------------

# This file lives at <project_root>/graphids/paths.py — one parent up from
# the package, two from this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATASET_REGISTRY_PATH: Path = CONFIG_DIR / "data" / "datasets.json"

PREPROCESSING_VERSION: str = "10.0.0"
CKPT_SUBPATH: str = "checkpoints/best_model.ckpt"

# Phase markers are kept for filesystem probes and diagnostics.
LAST_CKPT_SUBPATH: str = "checkpoints/last.ckpt"
PHASE_MARKERS: dict[str, str] = {
    "train": ".train_complete",
    "test": ".test_complete",
}


# ---------------------------------------------------------------------------
# Lake-root paths (raw CSVs, preprocessed caches)
# ---------------------------------------------------------------------------


def lake_root() -> str:
    """Resolve `$GRAPHIDS_LAKE_ROOT`, fail-fast if unset.

    Cross-user shared root: holds mlflow.db, cache/, mlartifacts/, slurm/.
    """
    lr = os.environ.get("GRAPHIDS_LAKE_ROOT")
    if not lr:
        raise RuntimeError("GRAPHIDS_LAKE_ROOT unset — set it to the shared data lake root")
    return lr


def data_dir(lake_root: str, dataset: str) -> Path:
    """Path to raw CSVs for a dataset: ``{lake_root}/raw/{dataset}``."""
    return Path(lake_root) / "raw" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    """Path to preprocessed tensor cache. Pinned to
    :data:`PREPROCESSING_VERSION` so a bump of the version forces rebuild
    without deleting the old tree.
    """
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


# ---------------------------------------------------------------------------
# Run-root helpers
# ---------------------------------------------------------------------------


def trial_dir() -> Path:
    """Root for GraphIDS run directories.

    Override with ``GRAPHIDS_RUN_ROOT``. Otherwise use ``$GRAPHIDS_LAKE_ROOT/runs``
    when available, falling back to ``<project>/runs`` for local development.
    """
    if override := os.environ.get("GRAPHIDS_RUN_ROOT"):
        return Path(override)
    if lake := os.environ.get("GRAPHIDS_LAKE_ROOT"):
        return Path(lake) / "runs"
    return PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, Any]]:
    """Flat view of ``configs/data/datasets.json`` — ``{dataset: {name, **entry}}``.

    Cached once per process. The on-disk format is already flat; the `name`
    field is added so callers can iterate values without losing the key.
    """
    if not DATASET_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Dataset registry missing: {DATASET_REGISTRY_PATH}")
    registry = json.loads(DATASET_REGISTRY_PATH.read_text())
    return {name: {"name": name, **entry} for name, entry in registry.items()}


def dataset_names() -> list[str]:
    """Public dataset names — entries starting with `_` are excluded."""
    return [k for k in load_catalog() if not k.startswith("_")]
