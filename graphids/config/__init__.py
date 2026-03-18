"""Configuration layer: inert, declarative, no imports from pipeline/ or core/.

Usage:
    from graphids.config import resolve, PipelineConfig, stage_dir, STAGES
"""

from .handler import ConfigHandler, load_pipeline_yaml  # noqa: F401
from .schema import (
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

# Singleton
_api = ConfigHandler()

# --- Methods ---
resolve = _api.resolve
stage_dir = _api.stage_dir
checkpoint_path = _api.checkpoint_path
config_path = _api.config_path
metrics_path = _api.metrics_path
data_dir = _api.data_dir
cache_dir = _api.cache_dir
run_id = _api.run_id
run_id_str = _api.run_id_str
run_metadata = _api.run_metadata
get_datasets = _api.get_datasets
sweep_result_path = _api.sweep_result_path
sweep_searcher_path = _api.sweep_searcher_path
list_models = _api.list_models
list_auxiliaries = _api.list_auxiliaries
parse_seeds = _api.parse_seeds
load_catalog = _api.load_catalog

# --- Constants ---
STAGES = _api.STAGES
STAGE_MODEL_MAP = _api.STAGE_MODEL_MAP
STAGE_DEPENDENCIES = _api.STAGE_DEPENDENCIES
VALID_MODEL_TYPES = _api.VALID_MODEL_TYPES
VALID_SCALES = _api.VALID_SCALES
EXPERIMENT_ROOT = _api.EXPERIMENT_ROOT
MLFLOW_TRACKING_URI = _api.MLFLOW_TRACKING_URI
DEFAULT_DATASET = _api.DEFAULT_DATASET
DEFAULT_SEEDS = _api.DEFAULT_SEEDS
PROJECT_ROOT = _api.PROJECT_ROOT
CATALOG_PATH = _api.CATALOG_PATH
PREPROCESSING_VERSION = _api.PREPROCESSING_VERSION
NODE_FEATURE_COUNT = _api.NODE_FEATURE_COUNT
EDGE_FEATURE_COUNT = _api.EDGE_FEATURE_COUNT
MAX_DATA_BYTES = _api.MAX_DATA_BYTES
EXCLUDED_ATTACK_TYPES = _api.EXCLUDED_ATTACK_TYPES
MMAP_TENSOR_LIMIT = _api.MMAP_TENSOR_LIMIT
SLURM_ACCOUNT = _api.SLURM_ACCOUNT
SLURM_PARTITION = _api.SLURM_PARTITION
SLURM_GPU_TYPE = _api.SLURM_GPU_TYPE
SWEEP_RESULTS_DIR = _api.SWEEP_RESULTS_DIR
