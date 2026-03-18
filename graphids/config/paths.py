"""Path derivation, run identity, and environment settings.

All path functions take a PipelineConfig and derive filesystem paths.
Path-related env vars (KD_GAT_LAKE_ROOT, etc.) flow through Hydra → PipelineConfig.
SLURM/MLflow env vars are read via pydantic-settings (outside config composition).
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

    # MLFLOW_TRACKING_URI doesn't have KD_GAT_ prefix — handled separately
    mlflow_tracking_uri: str | None = Field(None, validation_alias="MLFLOW_TRACKING_URI")

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
MLFLOW_TRACKING_URI: str = (
    _env.mlflow_tracking_uri or f"sqlite:///{PROJECT_ROOT / 'data' / 'mlflow' / 'mlflow.db'}"
)
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


def run_metadata(cfg: PipelineConfig, stage: str) -> dict[str, str]:
    """MLflow tags for a run."""
    tags = {
        "dataset": cfg.dataset,
        "model_type": cfg.model_type,
        "scale": cfg.scale,
        "stage": stage,
        "auxiliaries": cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
        "seed": str(cfg.seed),
        "run_group": run_id(cfg, stage),
        "config_hash": _config_hash(cfg),
    }
    if SWEEP_ID:
        tags["sweep_id"] = SWEEP_ID
    if USER_TAGS:
        tags["user_tags"] = USER_TAGS
    return tags


def _config_hash(cfg: PipelineConfig) -> str:
    return hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Path derivation (all reads from cfg, no hidden singleton)
# ---------------------------------------------------------------------------


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical output directory for a stage (always lake layout)."""
    aux = cfg.auxiliaries[0].type if cfg.auxiliaries else ""
    return _lake_run_dir(cfg, stage, aux=aux)


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
    return lake_cache_dir(cfg.lake_root, cfg.dataset)


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


# ---------------------------------------------------------------------------
# Lake path primitives (single source of truth — LakeConfig delegates here)
# ---------------------------------------------------------------------------


def lake_run_dir(
    lake_root: str | Path,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    aux: str = "",
    seed: int = 42,
    production: bool = True,
) -> Path:
    """Derive a lake run directory from raw identity dimensions.

    Path: {lake_root}/{production|dev/user}/{dataset}/{model}_{scale}_{stage}[_{aux}]/seed_{seed}

    This is the single source of truth for lake run-dir layout.
    Both PipelineConfig-based callers and LakeConfig delegate here.
    """
    import getpass

    tier = "production" if production else f"dev/{getpass.getuser()}"
    model = "eval" if stage == "evaluation" else model_type
    suffix = f"_{aux}" if aux else ""
    return Path(lake_root) / tier / dataset / f"{model}_{scale}_{stage}{suffix}" / f"seed_{seed}"


def lake_cache_dir(
    lake_root: str | Path,
    dataset: str,
    version: str | None = None,
) -> Path:
    """Derive a lake cache directory for preprocessed graphs.

    Path: {lake_root}/cache/v{version}/{dataset}

    Single source of truth for lake cache-dir layout.
    """
    if version is None:
        version = PREPROCESSING_VERSION
    return Path(lake_root) / "cache" / f"v{version}" / dataset


def lake_raw_dir(lake_root: str | Path, dataset: str) -> Path:
    """Derive a lake raw-data directory.

    Path: {lake_root}/raw/{dataset}

    Single source of truth for lake raw-dir layout.
    """
    return Path(lake_root) / "raw" / dataset


def lake_root_from_env() -> Path | None:
    """Read KD_GAT_LAKE_ROOT from the environment.

    Returns Path if set, None otherwise.
    """
    root = os.environ.get("KD_GAT_LAKE_ROOT")
    if not root:
        return None
    return Path(root)


def lake_sweep_dir(lake_root: str | Path, dataset: str) -> Path:
    """Sweep results directory for a dataset.

    Path: {lake_root}/sweeps/{dataset}
    """
    return Path(lake_root) / "sweeps" / dataset


def lake_catalog_path(lake_root: str | Path) -> Path:
    """DuckDB catalog file path.

    Path: {lake_root}/catalog/kd_gat.duckdb
    """
    return Path(lake_root) / "catalog" / "kd_gat.duckdb"


def lake_exports_dir(lake_root: str | Path) -> Path:
    """Exports directory for parquet files.

    Path: {lake_root}/exports
    """
    return Path(lake_root) / "exports"


def _lake_run_dir(cfg: PipelineConfig, stage: str, aux: str = "") -> Path:
    """PipelineConfig wrapper around lake_run_dir()."""
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
