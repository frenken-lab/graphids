"""Configuration layer: constants, paths, Hydra YAML."""

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
from .paths import (  # noqa: F401
    CKPT_PATH,
    SLURM_ACCOUNT,
    SLURM_GPU_TYPE,
    SLURM_PARTITION,
    SWEEP_ID,
    USER_TAGS,
    cache_dir,
    data_dir,
    get_datasets,
    load_catalog,
    parse_seeds,
)

CONF_DIR = str((CONFIG_DIR / "conf").resolve())


def resolve(*overrides: str):
    """Compose a Hydra config. Usage: resolve("model=vgae_large", "dataset=hcrl_sa")"""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        return compose(config_name="config", overrides=list(overrides))
