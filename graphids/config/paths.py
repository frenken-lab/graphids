"""Path derivation and environment constants."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .constants import (
    CATALOG_PATH,
    PREPROCESSING_VERSION,
)


# ---------------------------------------------------------------------------
# Environment (KD_GAT_* env vars with defaults)
# ---------------------------------------------------------------------------

SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", "v100")
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")


# ---------------------------------------------------------------------------
# Lake paths
# ---------------------------------------------------------------------------

def lake_root_from_env() -> Path | None:
    root = os.environ.get("KD_GAT_LAKE_ROOT")
    return Path(root) if root else None


def lake_cache_dir(lake_root, dataset: str, version: str | None = None) -> Path:
    return Path(lake_root) / "cache" / f"v{version or PREPROCESSING_VERSION}" / dataset


def lake_raw_dir(lake_root, dataset: str) -> Path:
    return Path(lake_root) / "raw" / dataset


def lake_catalog_path(lake_root) -> Path:
    return Path(lake_root) / "catalog" / "kd_gat.duckdb"


def lake_exports_dir(lake_root) -> Path:
    return Path(lake_root) / "exports"


def data_dir(cfg) -> Path:
    candidate = lake_raw_dir(cfg.lake_root, cfg.dataset)
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / cfg.dataset


def cache_dir(cfg) -> Path:
    return lake_cache_dir(cfg.lake_root, cfg.dataset)


# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------

_datasets_cache: list[str] | None = None


def get_datasets() -> list[str]:
    global _datasets_cache
    if _datasets_cache is None:
        _datasets_cache = list(yaml.safe_load(CATALOG_PATH.read_text()).keys())
    return _datasets_cache


def load_catalog() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text())


def parse_seeds(value: str) -> list[int]:
    if value is None:
        return []
    return [int(s.strip()) for s in value.split(",")]
