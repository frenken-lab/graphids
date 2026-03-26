"""Configuration layer: schema, resolution, constants."""

from .constants import (  # noqa: F401
    CATALOG_PATH,
    CONFIG_DIR,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    PIPELINE_YAML,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    compute_preprocessing_hash,
)
from .resolve import (  # noqa: F401
    cache_dir,
    compute_identity_hash,
    data_dir,
    resolve,
    to_namespace,
)
from .schema import Config  # noqa: F401

import os  # noqa: E402

SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", "v100")
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")
