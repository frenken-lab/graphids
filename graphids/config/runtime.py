"""Runtime constants and environment-backed defaults.

Layer 1 (constants): project invariants that change via commits, not config.
Layer 2 (env vars): machine/job-specific values read from os.environ at import.
"""

from __future__ import annotations

import os

from .topology import VALID_MODEL_TYPES, VALID_SCALES

# ---------------------------------------------------------------------------
# Layer 1 — project constants (no YAML, no env vars)
# ---------------------------------------------------------------------------

PREPROCESSING_VERSION: str = "7.0.0"
MAX_DATA_BYTES: int = 8
EXCLUDED_ATTACK_TYPES: list[str] = ["suppress", "masquerade"]
CKPT_SUBPATH: str = "checkpoints/best_model.ckpt"
LAST_CKPT_SUBPATH: str = "checkpoints/last.ckpt"
COMPLETE_MARKER: str = ".complete"
PHASE_MARKERS: dict[str, str] = {
    "train": ".train_complete",
    "test": ".test_complete",
    "analyze": ".analyze_complete",
}
DAGSTER_IO_DIR_TEMPLATE: str = "{lake_root}/dagster_io"
DEFAULT_MODEL_TYPE: str = next(iter(VALID_MODEL_TYPES)) if VALID_MODEL_TYPES else "vgae"
DEFAULT_SCALE: str = next(iter(VALID_SCALES)) if VALID_SCALES else "small"
DEFAULT_STAGE: str = "autoencoder"

# ---------------------------------------------------------------------------
# Layer 2 — environment config (machine/job-specific, read from os.environ)
# ---------------------------------------------------------------------------

LAKE_ROOT: str = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
DAGSTER_HOME_DEFAULT: str = os.environ.get("KD_GAT_DAGSTER_HOME", f"{LAKE_ROOT}/dagster")
WANDB_WRITE_DIR: str = os.environ.get("WANDB_DIR", "/fs/scratch/PAS1266/wandb")

SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_LOG_DIR: str = os.environ.get("KD_GAT_SLURM_LOG_DIR", f"{LAKE_ROOT}/slurm")
SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", "v100")
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")
