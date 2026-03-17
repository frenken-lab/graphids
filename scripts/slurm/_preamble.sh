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

# Group-writable umask for shared ESS data lake
umask 002

# Launch shared PostgreSQL if launcher exists (sets KD_GAT_DB_URI + MLFLOW_TRACKING_URI)
if [[ -f "$PROJECT_ROOT/scripts/lab-db/ensure_pg.sh" ]]; then
    source "$PROJECT_ROOT/scripts/lab-db/ensure_pg.sh" 2>/dev/null || true
fi

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

# SIGUSR1 trap for graceful timeout (SLURM sends USR1 before killing)
# When coordinator submits with --signal=B:USR1@180, this fires 180s before timeout.
# The trap forwards USR1 to the child process (Python training), which lets
# Lightning's SLURMEnvironment save a checkpoint before exit.
_KD_CHILD_PID=""
handle_timeout() {
    echo "[$(date)] SIGUSR1 received — forwarding to training process (PID=$_KD_CHILD_PID)..."
    if [[ -n "$_KD_CHILD_PID" ]]; then
        kill -USR1 "$_KD_CHILD_PID" 2>/dev/null || true
        wait "$_KD_CHILD_PID" 2>/dev/null
        EXIT_CODE=$?
        echo "[$(date)] Training process exited with code $EXIT_CODE"
        exit $EXIT_CODE
    fi
}
trap handle_timeout USR1

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
