"""Configuration layer: schema, resolution, constants."""

from jsonargparse import Namespace

from .constants import (  # noqa: F401
    CATALOG_PATH,
    CKPT_PATH,
    CONFIG_DIR,
    DEFAULT_DATASET,
    DEFAULT_MODEL_TYPE,
    DEFAULT_SCALE,
    DEFAULT_STAGE,
    DEFAULTS_DIR,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    PIPELINE_YAML,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    SLURM_ACCOUNT,
    SLURM_GPU_TYPE,
    SLURM_PARTITION,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    SWEEP_ID,
    USER_TAGS,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    cache_dir,
    compute_identity_hash,
    compute_preprocessing_hash,
    data_dir,
)
from .defaults.schema import Config  # noqa: F401
from .resolve import resolve  # noqa: F401


def to_namespace(cfg):
    """Convert dict (from checkpoint reload) to jsonargparse Namespace.

    Recursively converts nested dicts and list items. No-op on Namespace.
    """
    if isinstance(cfg, Namespace):
        return cfg
    if isinstance(cfg, dict):
        return Namespace(**{k: to_namespace(v) for k, v in cfg.items()})
    if isinstance(cfg, list):
        return [to_namespace(v) for v in cfg]
    return cfg
