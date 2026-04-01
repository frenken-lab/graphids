"""Public config API facade with stable imports across refactors."""

from .base import CONFIG_DIR, PROJECT_ROOT
from .contracts import KDEntry, TrainingRunConfig, expand_recipe_configs
from .paths import (
    CATALOG_PATH,
    DEFAULT_DATASET,
    cache_dir,
    checkpoint_path,
    compute_identity_hash,
    compute_preprocessing_hash,
    data_dir,
    run_dir,
)
from .runtime import (
    CKPT_PATH,
    CKPT_SUBPATH,
    COMPLETE_MARKER,
    DAGSTER_HOME_DEFAULT,
    DAGSTER_IO_DIR_TEMPLATE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_SCALE,
    DEFAULT_STAGE,
    EXCLUDED_ATTACK_TYPES,
    LAKE_ROOT,
    LAST_CKPT_SUBPATH,
    MAX_DATA_BYTES,
    PREPROCESSING_VERSION,
    SLURM_ACCOUNT,
    SLURM_GPU_TYPE,
    SLURM_LOG_DIR,
    SLURM_PARTITION,
    SWEEP_ID,
    USER_TAGS,
    WANDB_WRITE_DIR,
)
from .topology import (
    PIPELINE_YAML,
    STAGES,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    VALID_FUSION_METHODS,
    VALID_MODEL_TYPES,
    VALID_SCALES,
)
