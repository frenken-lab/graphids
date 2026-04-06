"""SLURM-scoped environment variables.

Kept out of ``graphids.config.constants`` because SLURM account / log
directory are SLURM infrastructure concerns, not config composition.
Callers (``slurm/slurm.py``, ``commands/pipeline_status.py``) import
from here.
"""

from __future__ import annotations

import os

from graphids.config.constants import LAKE_ROOT, PROJECT_ROOT

SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_LOG_DIR: str = os.environ.get("KD_GAT_SLURM_LOG_DIR", f"{LAKE_ROOT}/slurm")

# Shell script paths sourced by generated sbatch scripts
PREAMBLE_PATH: str = str(PROJECT_ROOT / "scripts" / "slurm" / "_preamble.sh")
EPILOG_PATH: str = str(PROJECT_ROOT / "scripts" / "slurm" / "_epilog.sh")
