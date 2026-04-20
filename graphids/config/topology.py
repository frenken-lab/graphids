"""Config-tree validation + path/catalog helpers.

Import-time check that the jsonnet tree (model families, fusion methods,
stage files) is coherent with the static axes, plus a small set of path
helpers and the dataset catalog loader.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import FilePath, TypeAdapter

from .constants import (
    CONFIG_DIR,
    DATASET_REGISTRY_PATH,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    VALID_FUSION_METHODS,
    VALID_MODEL_FAMILIES,
)

_CONFIGS_DIR = PROJECT_ROOT / "configs"
_file_check = TypeAdapter(FilePath)


def _validate_config_tree(stage_names: list[str]) -> None:
    required = [
        *(_CONFIGS_DIR / "models" / f"{f}.libsonnet" for f in VALID_MODEL_FAMILIES),
        _CONFIGS_DIR / "models" / "fusion" / "base.libsonnet",
        *(
            _CONFIGS_DIR / "models" / "fusion" / "methods" / f"{m}.libsonnet"
            for m in VALID_FUSION_METHODS
        ),
        *(_CONFIGS_DIR / "stages" / f"{s}.jsonnet" for s in stage_names),
    ]
    for p in required:
        _file_check.validate_python(p)
    _validate_submit_profiles()


def _validate_submit_profiles() -> None:
    """Import-time shape check for ``configs/resources/submit_profiles.json``.

    Enforces the two-profile (gpu, cpu) invariant — catches accidental
    profile proliferation at package load instead of at sbatch time.
    """
    path = _CONFIGS_DIR / "resources" / "submit_profiles.json"
    cfg = json.loads(path.read_text())
    profiles = cfg.get("submit_profiles") or {}
    expected = {"gpu", "cpu"}
    actual = set(profiles)
    if actual != expected:
        raise ValueError(
            f"submit_profiles.json must have exactly {sorted(expected)} entries, "
            f"got {sorted(actual)}. Per-job profiles belong in scripts/run flags."
        )
    for name, p in profiles.items():
        for key in ("mode", "cpus", "mem", "partitions", "times"):
            if key not in p:
                raise ValueError(f"submit_profiles[{name!r}] missing required key {key!r}")


STAGES: list[str] = json.loads((CONFIG_DIR / "matrix" / "topology.json").read_bytes())["stages"]
_validate_config_tree(STAGES)


def data_dir(lake_root: str, dataset: str) -> Path:
    return Path(lake_root) / "raw" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, dict[str, Any]]:
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
    return [k for k in load_catalog() if not k.startswith("_")]
