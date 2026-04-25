"""Path/catalog helpers and stage-name registry."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR, DATASET_REGISTRY_PATH, PREPROCESSING_VERSION

STAGES: list[str] = json.loads((CONFIG_DIR / "matrix" / "topology.json").read_bytes())["stages"]


def data_dir(lake_root: str, dataset: str) -> Path:
    """Path to raw CSVs for a dataset: ``{lake_root}/raw/{dataset}``."""
    return Path(lake_root) / "raw" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    """Path to preprocessed tensor cache. Pinned to
    :data:`graphids.config.constants.PREPROCESSING_VERSION` so a bump
    of the version forces rebuild without deleting the old tree.
    """
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, Any]]:
    """Flat view of ``configs/datasets/dataset_registry.json`` —
    ``{dataset_name: {name, domain, **entry}}``. Cached once per
    process; domains collapse into a ``domain`` field on each entry.
    """
    if not DATASET_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Dataset registry missing: {DATASET_REGISTRY_PATH}")
    registry = json.loads(DATASET_REGISTRY_PATH.read_text())
    return {
        name: {"name": name, "domain": domain, **entry}
        for domain, datasets in registry.items()
        if isinstance(datasets, dict)
        for name, entry in datasets.items()
    }


def dataset_names() -> list[str]:
    """Public dataset names — entries starting with ``_`` are internal
    placeholders (test fixtures, retired datasets) and excluded.
    """
    return [k for k in load_catalog() if not k.startswith("_")]
