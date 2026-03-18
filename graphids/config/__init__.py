"""Configuration layer: inert, declarative, no imports from pipeline/ or core/.

Usage:
    from graphids.config import resolve, PipelineConfig, stage_dir, STAGES
"""

from ._hydra_bridge import resolve  # noqa: F401
from .constants import (  # noqa: F401
    CATALOG_PATH,
    CONFIG_DIR,
    DEFAULT_DATASET,
    DEFAULT_SEEDS,
    EDGE_FEATURE_COUNT,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    MMAP_TENSOR_LIMIT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    SWEEP_RESULTS_DIR,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    load_pipeline_yaml,
)
from .paths import (  # noqa: F401
    CKPT_PATH,
    MLFLOW_TRACKING_URI,
    SLURM_ACCOUNT,
    SLURM_GPU_TYPE,
    SLURM_PARTITION,
    SWEEP_ID,
    USER_TAGS,
    cache_dir,
    checkpoint_path,
    config_path,
    data_dir,
    get_datasets,
    lake_cache_dir,
    lake_catalog_path,
    lake_exports_dir,
    lake_raw_dir,
    lake_root_from_env,
    lake_run_dir,
    lake_sweep_dir,
    load_catalog,
    parse_seeds,
    run_id,
    run_id_str,
    run_metadata,
    stage_dir,
    sweep_result_path,
)
from .schema import (  # noqa: F401
    AuxiliaryConfig,
    DatasetEntry,
    EvaluationArtifact,
    PipelineConfig,
    PreprocessingArtifact,
    PreprocessingConfig,
    TrainingArtifact,
    TrainingConfig,
    compute_preprocessing_hash,
)
