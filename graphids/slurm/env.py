"""SLURM-scoped environment variables.

``KD_GAT_SLURM_*`` env vars are read from ``graphids.config.settings``.
SLURM-injected vars (``SLURM_JOB_ID`` etc.) stay as direct ``os.environ``
reads because they are only set inside a running job.
"""

from __future__ import annotations

import os

from graphids.config.constants import PROJECT_ROOT
from graphids.config.settings import get_settings

_s = get_settings()
SLURM_ACCOUNT: str = _s.slurm_account
SLURM_LOG_DIR: str = _s.slurm_log_dir

# Shell script paths sourced by generated sbatch scripts
PREAMBLE_PATH: str = str(PROJECT_ROOT / "scripts" / "slurm" / "_preamble.sh")
EPILOG_PATH: str = str(PROJECT_ROOT / "scripts" / "slurm" / "_epilog.sh")


def slurm_job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID")


def slurm_job_partition() -> str | None:
    return os.environ.get("SLURM_JOB_PARTITION")


def slurm_cpus_per_task() -> int | None:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
