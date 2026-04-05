"""Public config API facade with stable imports across refactors."""

from .base import CONFIG_DIR, PROJECT_ROOT
from .contracts import KDEntry, TrainingRunConfig, expand_recipe_configs
from .paths import (
    CATALOG_SUBPATH,
    DEFAULT_DATASET,
    LakeWriteError,
    PathContext,
    cache_dir,
    catalog_path,
    checkpoint_path,
    compute_identity_hash,
    compute_preprocessing_hash,
    data_dir,
    dataset_names,
    load_catalog,
    require_lake_write,
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
    LAKE_ROOT,
    LAST_CKPT_SUBPATH,
    MAX_DATA_BYTES,
    PHASE_MARKERS,
    PREPROCESSING_VERSION,
    RUN_RECORD_FILENAME,
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
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    VALID_FUSION_METHODS,
    VALID_MODEL_TYPES,
    VALID_SCALES,
)
from .validated_config import (
    ConfigValidationError,
    ValidatedConfig,
    validate_config,
)
