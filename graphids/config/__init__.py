"""Configuration layer: constants, paths, Hydra YAML."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .constants import (  # noqa: F401
    CATALOG_PATH,
    CONFIG_DIR,
    DEFAULT_DATASET,
    DEFAULT_LAKE_ROOT,
    DEFAULT_SEEDS,
    EDGE_FEATURE_COUNT,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    MMAP_TENSOR_LIMIT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_DEFAULTS,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    SWEEP_RESULTS_DIR,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    compute_preprocessing_hash,
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


# ---------------------------------------------------------------------------
# Hydra config resolution
# ---------------------------------------------------------------------------

def resolve(*overrides: str):
    """Compose config: Hydra base + model preset + CLI overrides.

    Merge order: config.yaml defaults → model preset → CLI overrides.
    This ensures CLI overrides always win over model presets.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        base = compose(config_name="config", overrides=[])
        cfg = compose(config_name="config", overrides=list(overrides))

    # Merge order: base → preset → CLI overrides
    # CLI overrides = keys where cfg differs from base
    preset = _load_model_preset(cfg.model_type, cfg.scale)
    if preset:
        cli_diff = _extract_overrides(base, cfg)
        cfg = OmegaConf.merge(cfg, preset, cli_diff)
    return cfg


def _merge_model_preset(cfg):
    """Merge model preset into cfg (for @hydra.main where CLI overrides are already applied).

    Called from __main__.py where Hydra has already merged CLI overrides.
    We re-apply the overrides on top of the preset so CLI always wins.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # Get the base config (no overrides) to compute what the user changed
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        base = compose(config_name="config", overrides=[])

    preset = _load_model_preset(cfg.model_type, cfg.scale)
    if preset:
        cli_diff = _extract_overrides(base, cfg)
        cfg = OmegaConf.merge(cfg, preset, cli_diff)
    return cfg


def _load_model_preset(model_type: str, scale: str):
    """Load model preset from models.yaml, or None if not found."""
    from omegaconf import OmegaConf

    models = OmegaConf.load(CONFIG_DIR / "models.yaml")
    key = f"{model_type}_{scale}"
    if key in models and models[key]:
        return models[key]
    return None


def _extract_overrides(base, cfg):
    """Extract the subset of cfg that differs from base (i.e. CLI overrides)."""
    from omegaconf import DictConfig, OmegaConf

    diff = {}
    for key in cfg:
        if key not in base:
            diff[key] = cfg[key]
        elif isinstance(cfg[key], DictConfig) and isinstance(base.get(key), DictConfig):
            sub = _extract_overrides(base[key], cfg[key])
            if sub:
                diff[key] = sub
        elif OmegaConf.is_missing(base, key) or cfg[key] != base[key]:
            diff[key] = cfg[key]
    return OmegaConf.create(diff)
