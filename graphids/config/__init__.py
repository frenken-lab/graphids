"""Configuration layer: inert, declarative, no imports from pipeline/ or src/.

Usage:
    from graphids.config import PipelineConfig, STAGES, checkpoint_path, resolve
    from graphids.config.constants import NODE_FEATURE_COUNT, MMAP_TENSOR_LIMIT
"""

from .paths import (
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
from .resolver import resolve
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
