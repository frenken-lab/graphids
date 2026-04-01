"""Path and identity helpers derived from runtime config and topology."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import CONFIG_DIR
from .runtime import CKPT_SUBPATH, PREPROCESSING_VERSION
from .topology import PIPELINE_YAML
from .yaml_utils import read_yaml

_DATASETS_DIR: Path = CONFIG_DIR / "datasets"
DEFAULT_DATASET: str = "set_01"

_catalog_cache: dict[str, dict[str, Any]] | None = None


def load_catalog() -> dict[str, dict[str, Any]]:
    """Load dataset catalog from per-file configs in config/datasets/.

    Returns ``{dataset_name: {metadata_dict}}`` — same shape as the old
    monolithic ``datasets.yaml``, so consumers need only an import change.
    """
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    if not _DATASETS_DIR.is_dir():
        raise FileNotFoundError(f"Dataset config directory missing: {_DATASETS_DIR}")
    catalog: dict[str, dict[str, Any]] = {}
    for p in sorted(_DATASETS_DIR.glob("*.yaml")):
        entry = read_yaml(p)
        # Unwrap if nested under 'dataset' key (backwards compat with skeleton format)
        if "dataset" in entry and isinstance(entry["dataset"], dict):
            entry = entry["dataset"]
        name = entry.get("name", p.stem)
        catalog[name] = entry
    _catalog_cache = catalog
    return catalog


def dataset_names() -> list[str]:
    """Return list of dataset names (for dagster partitions, etc.)."""
    return [k for k in load_catalog() if not k.startswith("_")]


def run_dir(
    lake_root: str,
    user: str,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    identity: str,
    kd_tag: str,
    seed: int,
) -> str:
    return (
        f"{lake_root}/dev/{user}/{dataset}/"
        f"{model_type}_{scale}_{stage}{identity}{kd_tag}/seed_{seed}"
    )


def compute_preprocessing_hash() -> str:
    import hashlib

    from graphids.core.preprocessing.features import N_EDGE_FEATURES, N_NODE_FEATURES

    components = [
        PREPROCESSING_VERSION,
        str(N_NODE_FEATURES),
        str(N_EDGE_FEATURES),
        "100",
        "100",
        "0.8",
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]


def data_dir(lake_root: str, dataset: str) -> Path:
    candidate = Path(lake_root) / "raw" / dataset
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


def compute_identity_hash(stage: str, cfg: Any) -> str:
    import hashlib

    stage_def = PIPELINE_YAML.get("stages", {}).get(stage, {})
    keys = stage_def.get("identity_keys", [])
    if not keys:
        return ""

    def _get(dotted_key: str, default=None):
        cur = cfg
        for part in dotted_key.split("."):
            if cur is None:
                return default
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        return cur if cur is not None else default

    unresolved = [k for k in keys if _get(k) is None]
    if unresolved:
        raise KeyError(
            f"Identity keys {unresolved} not found in config for stage '{stage}'. "
            "These keys must be set for stable checkpoint identity."
        )

    pairs = [f"{k}={_get(k, '_default_')}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]


def checkpoint_path(
    lake_root: str,
    dataset: str,
    model_type: str,
    scale: str,
    seed: int,
    cfg: Any,
    *,
    gat_stage: str = "curriculum",
) -> Path:
    user = os.environ.get("USER", "unknown")
    output_base = f"{lake_root}/dev/{user}/{dataset}"
    stage = PIPELINE_YAML.get("ckpt_stages", {}).get(model_type, model_type)
    if model_type == "gat":
        stage = gat_stage
    identity = compute_identity_hash(stage, cfg)
    return Path(f"{output_base}/{model_type}_{scale}_{stage}{identity}/seed_{seed}/{CKPT_SUBPATH}")
