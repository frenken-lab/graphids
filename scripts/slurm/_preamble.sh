#!/usr/bin/env bash
# scripts/slurm/_preamble.sh — sourced by SLURM job scripts.
# Sets up Python environment, env vars, and data staging.
#
# Lightning handles: GPU monitoring (DeviceStatsMonitor callback),
# USR1/timeout (SLURMEnvironment plugin), checkpointing (ModelCheckpoint).
#
# Override before sourcing:
#   SKIP_STAGE_DATA=1  — skip data staging (tests, fusion jobs)
#   SKIP_CUDA_CONF=1   — skip CUDA alloc config (CPU-only jobs)

set -euo pipefail

# Without this, Python fully buffers stdout under SLURM (no TTY).
# turm and tail -f see nothing until the process exits.
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

module load python/3.12
source .venv/bin/activate
set -a; source ./.env; set +a

SLURM_LOG_DIR="${KD_GAT_SLURM_LOG_DIR:-${KD_GAT_LAKE_ROOT:-experimentruns}/slurm}"
mkdir -p "$SLURM_LOG_DIR"

# Group-writable umask for shared ESS data lake
umask 002

# wandb: scratch for I/O-heavy run data, skip git probing on NFS, reduce SLURM log noise
export WANDB_DIR="${WANDB_DIR:-/fs/scratch/PAS1266/wandb}"
mkdir -p "$WANDB_DIR"
export WANDB_DISABLE_GIT=true
export WANDB_SILENT=true

if [[ "${SKIP_CUDA_CONF:-0}" != "1" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
fi

# Data staging: NFS → scratch → TMPDIR (node-local SSD)
if [[ "${SKIP_STAGE_DATA:-0}" != "1" ]]; then
    eval "$(python -m graphids stage-data ${STAGE_DATA_ARGS:---cache} | grep '^export ')"
fi

if [[ -n "${TMPDIR:-}" ]]; then
    export KD_GAT_STAGE_DIR="$TMPDIR/kd-gat-stage"
    mkdir -p "$KD_GAT_STAGE_DIR"
fi
