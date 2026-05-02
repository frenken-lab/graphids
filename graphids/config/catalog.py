"""Dataset catalog helpers and filesystem path scheme.

One module owns three related concerns:
- dataset registry lookup (`load_catalog`, `dataset_names`)
- raw/cache paths under `LAKE_ROOT` (`data_dir`, `cache_dir`)
- run paths under `RUN_ROOT` (`run_dir`, `best_ckpt`, `states_dir`)

The run-path functions are also exposed to jsonnet as `std.native('paths.*')`
by :mod:`graphids.config.jsonnet`. Keeping the schemes here means jsonnet and
Python share one source — no parallel jsonnet implementation that can drift.

`run_root` is read from `$GRAPHIDS_RUN_ROOT` lazily; the path scheme is
`{run_root}/{dataset}/ablations/{group}/{variant}/seed_{N}` (and
`{run_root}/{dataset}/cached_states/seed_{N}` for fusion).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import DATASET_REGISTRY_PATH, PREPROCESSING_VERSION

# ---------------------------------------------------------------------------
# Lake-root paths (raw CSVs, preprocessed caches)
# ---------------------------------------------------------------------------


def lake_root() -> str:
    """Resolve `$GRAPHIDS_LAKE_ROOT`, fail-fast if unset.

    Cross-user shared root: holds mlflow.db, cache/, mlartifacts/,
    slurm_logs/. Distinct from `$GRAPHIDS_RUN_ROOT` (per-user run writes).
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
    :data:`graphids.config.constants.PREPROCESSING_VERSION` so a bump
    of the version forces rebuild without deleting the old tree.
    """
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


# ---------------------------------------------------------------------------
# Run-root paths (per-user experiment writes)
# ---------------------------------------------------------------------------


def _run_root() -> str:
    """Resolve `$GRAPHIDS_RUN_ROOT`, fail-fast if unset."""
    rr = os.environ.get("GRAPHIDS_RUN_ROOT")
    if not rr:
        raise RuntimeError("GRAPHIDS_RUN_ROOT unset — set it to the per-user experiment root")
    return rr


def run_dir(dataset: str, group: str, variant: str, seed: int) -> str:
    """Per-(variant, seed) run directory under `$GRAPHIDS_RUN_ROOT`."""
    return str(Path(_run_root()) / dataset / "ablations" / group / variant / f"seed_{int(seed)}")


def best_ckpt(dataset: str, group: str, variant: str, seed: int) -> str:
    """Best-model checkpoint path. Suffix lives here so callers don't string-concat."""
    return f"{run_dir(dataset, group, variant, seed)}/checkpoints/best_model.ckpt"


def states_dir(dataset: str, seed: int) -> str:
    """Fusion-states directory shared across the 4 fusion methods for a seed."""
    return str(Path(_run_root()) / dataset / "cached_states" / f"seed_{int(seed)}")


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
