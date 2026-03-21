"""Configuration layer: constants, paths, Hydra YAML.

Usage:
    from graphids.config import STAGES, STAGE_MODEL_MAP, data_dir, cache_dir
    # Config resolution: use hydra.compose() directly
    # Config save/load: use OmegaConf.save() / OmegaConf.load()
"""

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
    load_pipeline_yaml,
)
CONF_DIR = str((CONFIG_DIR / "conf").resolve())


def resolve(*overrides: str):
    """Compose a Hydra config. Usage: resolve("model=vgae_large", "dataset=hcrl_sa")"""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        return compose(config_name="config", overrides=list(overrides))


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
    lake_catalog_path,
    lake_exports_dir,
    lake_root_from_env,
    load_catalog,
    parse_seeds,
)
