#!/usr/bin/env bash
# Full sweep pipeline DAG on SLURM (7 steps: 3 sweeps + 3 trains + 1 eval).
#
# Usage:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/sweep_pipeline.sh <dataset> <scale> [--num-samples 20]
#
# Examples:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/sweep_pipeline.sh set_01 large --num-samples 20
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/sweep_pipeline.sh hcrl_ch large --num-samples 1 --tune-epochs 2

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=kd-sweep-pipe
#SBATCH --output=slurm_logs/sweep_pipeline_%j.out
#SBATCH --error=slurm_logs/sweep_pipeline_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

# --- Environment ---
module load python/3.12
source .venv/bin/activate

# Source project env vars
set -a
source .env
set +a

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage data to fast storage
source scripts/data/stage_data.sh --cache

# Extract dataset and scale from positional args, forward the rest
DATASET="${1:?Usage: sweep_pipeline.sh <dataset> <scale> [--num-samples N] [--tune-epochs N]}"
SCALE="${2:?Usage: sweep_pipeline.sh <dataset> <scale> [--num-samples N] [--tune-epochs N]}"
shift 2

# SIGUSR1 handler: log warning, Python orchestrator handles state persistence
trap 'echo "SIGUSR1 received — wall time approaching. State file is safe for resume on resubmit."' USR1

echo "=== Sweep Pipeline DAG ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Dataset:   ${DATASET}"
echo "Scale:     ${SCALE}"
echo "Extra args: $*"
echo "Python:    $(which python)"
echo ""

# Run the sweep pipeline (--resume is default)
python -m graphids.pipeline.cli sweep-pipeline \
    --dataset "${DATASET}" \
    --scale "${SCALE}" \
    "$@"

EXIT_CODE=$?

# --- Post-job ---
# Sync datalake Parquet to S3
if [[ -d data/datalake ]] && command -v aws &>/dev/null; then
    echo "Syncing datalake to S3..."
    aws s3 sync data/datalake/ "s3://${KD_GAT_S3_BUCKET:-kd-gat}/datalake/" \
        --exclude "analytics.duckdb" 2>/dev/null || true
fi

# Sync sweep results to S3
if [[ -d data/sweep_results ]] && command -v aws &>/dev/null; then
    echo "Syncing sweep results to S3..."
    aws s3 sync data/sweep_results/ "s3://${KD_GAT_S3_BUCKET:-kd-gat}/sweep_results/" \
        2>/dev/null || true
fi

# Sync sweep state to S3
if [[ -d data/sweep_state ]] && command -v aws &>/dev/null; then
    echo "Syncing sweep state to S3..."
    aws s3 sync data/sweep_state/ "s3://${KD_GAT_S3_BUCKET:-kd-gat}/sweep_state/" \
        2>/dev/null || true
fi

# Sync offline W&B runs
if command -v wandb &>/dev/null; then
    echo "Syncing offline W&B runs..."
    find wandb/ -maxdepth 1 -name "run-*" -newer slurm_logs/sweep_pipeline_${SLURM_JOB_ID}.out \
        -exec wandb sync {} \; 2>/dev/null || true
fi

# GPU utilization report
source scripts/slurm/job_epilog.sh

exit $EXIT_CODE
