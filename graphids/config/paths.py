"""Path derivation and environment constants."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .constants import CATALOG_PATH, PREPROCESSING_VERSION


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
# Config-based paths (called with Hydra cfg)
# ---------------------------------------------------------------------------

def data_dir(cfg) -> Path:
    """Raw data directory. Tries lake, falls back to local."""
    candidate = Path(cfg.lake_root) / "raw" / cfg.dataset
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / cfg.dataset


def cache_dir(cfg) -> Path:
    """Processed-graph cache directory."""
    return Path(cfg.lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / cfg.dataset


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
