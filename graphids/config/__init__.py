"""Configuration layer: inert, declarative, no imports from pipeline/ or src/.

Usage:
    from graphids.config import PipelineConfig, STAGES, checkpoint_path
    from graphids.config.resolver import resolve, list_models, list_auxiliaries
    from graphids.config.constants import NODE_FEATURE_COUNT, MMAP_TENSOR_LIMIT
"""

from .constants import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    EDGE_FEATURE_COUNT,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    MMAP_TENSOR_LIMIT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_VERSION,
    SLURM_ACCOUNT,
    SLURM_GPU_TYPE,
    SLURM_PARTITION,
)
from .paths import (
    CATALOG_PATH,
    EXPERIMENT_ROOT,
    MLFLOW_TRACKING_URI,
    STAGES,
    benchmark_path_str,
    cache_dir,
    checkpoint_path,
    checkpoint_path_str,
    config_path,
    data_dir,
    get_datasets,
    log_dir,
    metrics_path,
    metrics_path_str,
    run_id,
    run_id_str,
    stage_dir,
)
from .resolver import (
    list_auxiliaries,
    list_models,
    resolve,
)
from .schema import (
    AuxiliaryConfig,
    DQNArchitecture,
    FusionConfig,
    GATArchitecture,
    PipelineConfig,
    PreprocessingConfig,
    TrainingConfig,
    VariantConfig,
    VGAEArchitecture,
)
