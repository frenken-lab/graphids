"""Path derivation and environment settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    CATALOG_PATH,
    PREPROCESSING_VERSION,
)

if TYPE_CHECKING:
    from .schema import PipelineConfig


# ---------------------------------------------------------------------------
# Environment settings (SLURM + MLflow only — path vars are in PipelineConfig)
# ---------------------------------------------------------------------------


class EnvironmentSettings(BaseSettings):
    """KD_GAT_* env vars for infrastructure outside config composition."""

    model_config = SettingsConfigDict(env_prefix="KD_GAT_")

    slurm_account: str = "PAS1266"
    slurm_partition: str = "gpu"
    gpu_type: str = "v100"

    # Run metadata (not config identity — never on PipelineConfig)
    sweep_id: str = ""
    tags: str = ""
    ckpt_path: str = ""


# Module-level singleton (read once at import)
_env = EnvironmentSettings()

# Derived constants from env
SLURM_ACCOUNT: str = _env.slurm_account
SLURM_PARTITION: str = _env.slurm_partition
SLURM_GPU_TYPE: str = _env.gpu_type
SWEEP_ID: str = _env.sweep_id
USER_TAGS: str = _env.tags
CKPT_PATH: str = _env.ckpt_path


# ---------------------------------------------------------------------------
# Lake path primitives (preprocessing cache + raw data — not per-run)
# ---------------------------------------------------------------------------

_DEFAULT_PREPROCESSING_VERSION = "4"


def lake_cache_dir(lake_root: str | Path, dataset: str, version: str | None = None) -> Path:
    """Path: {lake_root}/cache/v{version}/{dataset}"""
    if version is None:
        version = _DEFAULT_PREPROCESSING_VERSION
    return Path(lake_root) / "cache" / f"v{version}" / dataset


def lake_raw_dir(lake_root: str | Path, dataset: str) -> Path:
    """Path: {lake_root}/raw/{dataset}"""
    return Path(lake_root) / "raw" / dataset


def lake_root_from_env() -> Path | None:
    """Read KD_GAT_LAKE_ROOT from the environment."""
    root = os.environ.get("KD_GAT_LAKE_ROOT")
    return Path(root) if root else None


def lake_catalog_path(lake_root: str | Path) -> Path:
    """Path: {lake_root}/catalog/kd_gat.duckdb"""
    return Path(lake_root) / "catalog" / "kd_gat.duckdb"


# ---------------------------------------------------------------------------
# Path derivation (PipelineConfig-based)
# ---------------------------------------------------------------------------


def data_dir(cfg: PipelineConfig) -> Path:
    """Raw data directory for a dataset."""
    candidate = lake_raw_dir(cfg.lake_root, cfg.dataset)
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / cfg.dataset


def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory."""
    return lake_cache_dir(cfg.lake_root, cfg.dataset, version=PREPROCESSING_VERSION)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_datasets_cache: list[str] | None = None


def get_datasets() -> list[str]:
    """Dataset names from datasets.yaml (cached)."""
    global _datasets_cache
    if _datasets_cache is None:
        _datasets_cache = list(load_catalog().keys())
    return _datasets_cache


def load_catalog() -> dict:
    """Load and validate all dataset entries from datasets.yaml."""
    from .schema import DatasetEntry

    raw = yaml.safe_load(CATALOG_PATH.read_text())
    return {name: DatasetEntry.model_validate(entry) for name, entry in raw.items()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def parse_seeds(value: str) -> list[int]:
    """Parse seeds: comma-separated ints."""
    if value is None:
        return []
    try:
        return [int(s.strip()) for s in value.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid seeds value '{value}': {e}") from e
