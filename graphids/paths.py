"""Project-wide path helpers + dataset registry.

The custom per-run root has been retired in favor of Ray storage/trial
primitives. The remaining helpers are thin naming wrappers that sit on top
of Ray's experiment directory, checkpoint directory, and storage context.

This module still owns:
- dataset registry lookup (``load_catalog``, ``dataset_names``)
- raw / cache paths under ``$GRAPHIDS_LAKE_ROOT`` (``data_dir``, ``cache_dir``)
- Ray-backed trial / checkpoint / artifact helpers

Import-safe: no torch, Ray imports are lazy, so this stays usable from
login-node code paths and config composition.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Concrete Literal — Pydantic needs this for field validation. Each model
# module annotates its `model_type` arg with this so the rendered_config's
# value is checked at instantiation. (Fusion methods aren't model_types in
# this sense; their identity is `model_type='fusion'` + a `method` field.)
ModelType = Literal["vgae", "dgi", "gat", "fusion"]


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

    Cross-user shared root: holds mlflow.db, cache/, mlartifacts/,
    slurm_logs/. Distinct from Ray's run storage root (per-run writes).
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
# Ray-backed run / trial helpers
# ---------------------------------------------------------------------------


def _ray_context() -> Any | None:
    """Return the active Ray Tune/Train context if one exists."""
    with suppress(ImportError):
        from ray import tune

        with suppress(RuntimeError):
            return tune.get_context()
    with suppress(ImportError):
        from ray import train

        with suppress(RuntimeError):
            return train.get_context()
    return None


def trial_dir() -> Path:
    """Ray-backed experiment directory.

    Inside Tune/Train this returns the native trial directory. Outside Ray it
    falls back to a deterministic path under the default Ray results root.
    """
    ctx = _ray_context()
    if ctx is not None:
        try:
            return Path(ctx.get_trial_dir())
        except Exception:
            pass
    try:
        storage = ctx.get_storage() if ctx is not None else None
        path = getattr(storage, "storage_path", None) if storage is not None else None
        if path:
            return Path(path)
    except Exception:
        pass
    return Path.home() / "ray_results"


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
