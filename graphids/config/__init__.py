"""Configuration layer: schema and constants."""

import dataclasses


def coerce_config(val, cls):
    """Coerce jsonargparse Namespace or dict to a dataclass. Passthrough if already correct type.

    LightningCLI passes Namespace (live CLI), checkpoint reload passes dict.
    Both must become the typed dataclass so downstream code can use attribute access
    with full defaults.
    """
    if isinstance(val, cls):
        return val
    if hasattr(val, "as_dict"):  # jsonargparse Namespace
        return cls(**val.as_dict())
    if isinstance(val, dict):
        return cls(**val)
    if hasattr(val, "__dict__"):  # plain Namespace
        return cls(**{k: v for k, v in vars(val).items() if k in {f.name for f in dataclasses.fields(cls)}})
    raise TypeError(f"Cannot coerce {type(val).__name__} to {cls.__name__}")

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
    checkpoint_path,
    compute_identity_hash,
    compute_preprocessing_hash,
    data_dir,
)
