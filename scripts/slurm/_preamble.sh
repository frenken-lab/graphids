#!/usr/bin/env bash
# scripts/slurm/_preamble.sh — sourced by all SLURM job scripts.
# Sets up environment, activates venv, sources .env, stages data.
#
# Override before sourcing:
#   STAGE_DATA_ARGS="--raw"  — for preprocessing jobs (default: --cache)
#   SKIP_STAGE_DATA=1        — skip data staging entirely (e.g. CPU-only tests)
#   SKIP_CUDA_CONF=1         — skip PYTORCH_CUDA_ALLOC_CONF (e.g. CPU jobs)

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

module load python/3.12
source .venv/bin/activate

set -a; source .env; set +a

# MLflow tracking URI (sourced from .env, but ensure it's set for all jobs)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$PROJECT_ROOT/data/mlflow/mlflow.db}"

if [[ "${SKIP_CUDA_CONF:-0}" != "1" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

if [[ "${SKIP_STAGE_DATA:-0}" != "1" ]]; then
    source scripts/data/stage_data.sh ${STAGE_DATA_ARGS:---cache}
fi

# Redirect training artifacts to node-local SSD when running under SLURM
if [[ -n "${TMPDIR:-}" ]]; then
    export KD_GAT_STAGE_DIR="$TMPDIR/kd-gat-stage"
    mkdir -p "$KD_GAT_STAGE_DIR"
fi

# Shared job header/footer for consistent log formatting
log_job_header() {
    echo "=== $1 ==="
    echo "Job ID:    ${SLURM_JOB_ID:-interactive}"
    echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
    echo "Started:   $(date)"
    echo "Python:    $(which python)"
    echo ""
}

log_job_footer() {
    local exit_code=$1
    echo ""
    echo "=== $([ "$exit_code" -eq 0 ] && echo 'COMPLETE' || echo "FAILED (exit $exit_code)") ==="
}
