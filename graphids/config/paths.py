"""Path derivation, run identity, and environment settings.

PipelineConfig-based path helpers and environment settings.
Lake path primitives (lake_run_dir, etc.) live in graphids.storage.paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    CATALOG_PATH,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    SWEEP_RESULTS_DIR,
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
# Run identity
# ---------------------------------------------------------------------------


def _id_parts(cfg: PipelineConfig, stage: str) -> tuple[str, str, str]:
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    model = "eval" if stage == "evaluation" else cfg.model_type
    return model, cfg.scale, aux_suffix


def run_id(cfg: PipelineConfig, stage: str) -> str:
    """Deterministic run ID: {dataset}/{model}_{scale}_{stage}[_{aux}]."""
    model, scale, aux_suffix = _id_parts(cfg, stage)
    return f"{cfg.dataset}/{model}_{scale}_{stage}{aux_suffix}"


def run_id_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Run ID from raw strings (no PipelineConfig needed)."""
    suffix = f"_{aux}" if aux else ""
    model = "eval" if stage == "evaluation" else model_type
    return f"{dataset}/{model}_{scale}_{stage}{suffix}"




def _config_hash(cfg: PipelineConfig) -> str:
    return hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Path derivation (PipelineConfig-based wrappers over storage layer)
# ---------------------------------------------------------------------------


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical output directory for a stage (always lake layout)."""
    aux = cfg.auxiliaries[0].type if cfg.auxiliaries else ""
    return lake_run_dir(
        lake_root=cfg.lake_root,
        dataset=cfg.dataset,
        model_type=cfg.model_type,
        scale=cfg.scale,
        stage=stage,
        aux=aux,
        seed=cfg.seed,
        production=cfg.production,
    )


def checkpoint_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the best model checkpoint is saved."""
    return stage_dir(cfg, stage) / "best_model.pt"


def config_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the frozen config JSON is saved alongside the model."""
    return stage_dir(cfg, stage) / "config.json"


def data_dir(cfg: PipelineConfig) -> Path:
    """Raw data directory for a dataset."""
    candidate = lake_raw_dir(cfg.lake_root, cfg.dataset)
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / cfg.dataset


def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory."""
    return lake_cache_dir(cfg.lake_root, cfg.dataset, version=PREPROCESSING_VERSION)


def sweep_result_path(stage: str, dataset: str, scale: str) -> Path:
    """Path for a sweep's best-config YAML.

    Standalone (no PipelineConfig) because sweep orchestration iterates
    over steps without a full config. Reads KD_GAT_LAKE_ROOT directly.
    """
    lake_root = os.environ.get("KD_GAT_LAKE_ROOT")
    if lake_root:
        return Path(lake_root) / "sweeps" / dataset / f"{stage}_{scale}_best.yaml"
    return PROJECT_ROOT / SWEEP_RESULTS_DIR / f"{stage}_{dataset}_{scale}_best.yaml"


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
