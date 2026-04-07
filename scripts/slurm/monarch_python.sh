#!/bin/bash
# Monarch worker wrapper — sets up the env before Python starts.
#
# Used as `python_exe` in SlurmJob. Monarch generates:
#   srun /path/to/monarch_python.sh -c 'from monarch.actor import ...'
# This script sources .env + CUDA config, then exec's the venv Python
# with all original args ($@). The worker process inherits the full
# environment that _preamble.sh would normally provide.
#
# Why: Monarch spawns workers via a bare srun, bypassing _preamble.sh.
# Workers need KD_GAT_LAKE_WRITE, SLURM account, CUDA alloc config, etc.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Module system (OSC requires this for shared libs linked by venv Python)
module load python/3.12 2>/dev/null || true

# Project env vars (.env has KD_GAT_LAKE_WRITE, KD_GAT_SLURM_ACCOUNT, etc.)
set -a; source "$PROJECT_ROOT/.env"; set +a

# CUDA memory config (same as _preamble.sh)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# Unbuffered output for SLURM log visibility
export PYTHONUNBUFFERED=1

exec "$PROJECT_ROOT/.venv/bin/python" "$@"
