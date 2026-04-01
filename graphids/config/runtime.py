"""Runtime constants and environment-backed defaults."""

from __future__ import annotations

import os

from .base import CONFIG_DIR
from .topology import VALID_MODEL_TYPES, VALID_SCALES
from .yaml_utils import read_yaml

_global_defaults = read_yaml(CONFIG_DIR / "defaults" / "global.yaml")
_io_defaults = read_yaml(CONFIG_DIR / "defaults" / "io.yaml")

PREPROCESSING_VERSION: str = os.environ.get("KD_GAT_PREPROCESSING_VERSION", "7.0.0")
MAX_DATA_BYTES: int = int(os.environ.get("KD_GAT_MAX_DATA_BYTES", "8"))
EXCLUDED_ATTACK_TYPES: list[str] = ["suppress", "masquerade"]

LAKE_ROOT: str = os.environ.get(
    "KD_GAT_LAKE_ROOT",
    str(_global_defaults.get("paths", {}).get("lake_root", "experimentruns")),
)

_io = _io_defaults.get("io", {})
CKPT_SUBPATH: str = _io.get("checkpoint_subpath", "checkpoints/best_model.ckpt")
LAST_CKPT_SUBPATH: str = _io.get("last_checkpoint_subpath", "checkpoints/last.ckpt")
COMPLETE_MARKER: str = _io.get("complete_marker", ".complete")
DAGSTER_IO_DIR_TEMPLATE: str = "{lake_root}/dagster_io"
DAGSTER_HOME_DEFAULT: str = os.environ.get("KD_GAT_DAGSTER_HOME", f"{LAKE_ROOT}/dagster")
WANDB_WRITE_DIR: str = os.environ.get("WANDB_DIR", "/fs/scratch/PAS1266/wandb")

SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_LOG_DIR: str = os.environ.get("KD_GAT_SLURM_LOG_DIR", f"{LAKE_ROOT}/slurm")
SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", "v100")
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")

DEFAULT_MODEL_TYPE: str = next(iter(VALID_MODEL_TYPES)) if VALID_MODEL_TYPES else "vgae"
DEFAULT_SCALE: str = next(iter(VALID_SCALES)) if VALID_SCALES else "small"
DEFAULT_STAGE: str = "autoencoder"
