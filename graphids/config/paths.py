"""Path and identity helpers derived from runtime config and topology."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict

from .base import CONFIG_DIR
from .runtime import (
    CKPT_SUBPATH,
    COMPLETE_MARKER,
    LAST_CKPT_SUBPATH,
    PREPROCESSING_VERSION,
)
from .topology import PIPELINE_YAML
from .yaml_utils import read_yaml


class LakeWriteError(PermissionError):
    """Raised when lake write is attempted without KD_GAT_LAKE_WRITE=1."""


def require_lake_write() -> None:
    """Gate all data-lake writes behind KD_GAT_LAKE_WRITE=1.

    SLURM jobs get this via _preamble.sh → .env. Direct CLI invocations
    on login nodes or interactive sessions are blocked by default.
    """
    if os.environ.get("KD_GAT_LAKE_WRITE") != "1":
        raise LakeWriteError(
            "Lake write blocked: set KD_GAT_LAKE_WRITE=1 in environment "
            "(SLURM jobs get this from .env via _preamble.sh). "
            "For read-only runs, use --dry-run."
        )


class PathContext(BaseModel):
    """Frozen path model — single source for all run-related paths."""

    model_config = ConfigDict(frozen=True)

    lake_root: str
    user: str
    dataset: str
    model_type: str
    scale: str
    stage: str
    identity: str
    kd_tag: str
    seed: int

    @property
    def run_dir(self) -> Path:
        return Path(
            f"{self.lake_root}/dev/{self.user}/{self.dataset}/"
            f"{self.model_type}_{self.scale}_{self.stage}"
            f"{self.identity}{self.kd_tag}/seed_{self.seed}"
        )

    @property
    def ckpt_file(self) -> Path:
        return self.run_dir / CKPT_SUBPATH

    @property
    def complete_marker(self) -> Path:
        return self.run_dir / COMPLETE_MARKER

    @property
    def last_ckpt_file(self) -> Path:
        return self.run_dir / LAST_CKPT_SUBPATH

    @property
    def resolved_ckpt(self) -> Path:
        """Best-available checkpoint: ``ckpt_file`` if present, else ``last_ckpt_file``.

        Fusion RL (DQN/bandit) never writes a ``best_model.ckpt`` because they
        don't track a validation-loss minimum — they only produce ``last.ckpt``.
        """
        return self.ckpt_file if self.ckpt_file.exists() else self.last_ckpt_file

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / PurePosixPath(CKPT_SUBPATH).parent

_DATASETS_DIR: Path = CONFIG_DIR / "datasets"
DEFAULT_DATASET: str = "set_01"

CATALOG_SUBPATH: str = "catalog/kd_gat.duckdb"


def catalog_path(lake_root: str) -> Path:
    """Return the DuckDB experiment catalog path for a given lake root."""
    return Path(lake_root) / CATALOG_SUBPATH

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


def compute_preprocessing_hash() -> str:
    import hashlib

    from graphids.core.preprocessing.datasets.can_bus import N_EDGE_FEATURES, N_NODE_FEATURES

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
    """Resolve checkpoint path for a given model. Delegates to PathContext."""
    stage = PIPELINE_YAML.get("ckpt_stages", {}).get(model_type, model_type)
    if model_type == "gat":
        stage = gat_stage
    identity = compute_identity_hash(stage, cfg)
    return PathContext(
        lake_root=lake_root,
        user=os.environ.get("USER", "unknown"),
        dataset=dataset,
        model_type=model_type,
        scale=scale,
        stage=stage,
        identity=identity,
        kd_tag="",
        seed=seed,
    ).ckpt_file
