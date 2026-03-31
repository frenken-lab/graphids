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

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

module load python/3.12
source .venv/bin/activate
set -a; source ./.env; set +a

# Group-writable umask for shared ESS data lake
umask 002

# wandb: scratch for I/O-heavy run data, skip git probing on NFS, reduce SLURM log noise
# Path sourced from write_paths.yaml via config/__init__.py (single source of truth)
export WANDB_DIR=$(python -c "from graphids.config import WANDB_WRITE_DIR; print(WANDB_WRITE_DIR)")
mkdir -p "$WANDB_DIR"
export WANDB_DISABLE_GIT=true
export WANDB_SILENT=true

if [[ "${SKIP_CUDA_CONF:-0}" != "1" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
fi

# Data staging: NFS → scratch → TMPDIR (node-local SSD)
if [[ "${SKIP_STAGE_DATA:-0}" != "1" ]]; then
    source scripts/data/stage_data.sh ${STAGE_DATA_ARGS:---cache}
fi

if [[ -n "${TMPDIR:-}" ]]; then
    export KD_GAT_STAGE_DIR="$TMPDIR/kd-gat-stage"
    mkdir -p "$KD_GAT_STAGE_DIR"
fi
