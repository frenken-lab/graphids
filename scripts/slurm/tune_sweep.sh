#!/usr/bin/env bash
# Ray Tune HPO sweep on SLURM.
#
# Usage:
#   sbatch scripts/slurm/tune_sweep.sh autoencoder --dataset set_01 --scale large --num-samples 20
#   sbatch scripts/slurm/tune_sweep.sh curriculum  --dataset set_01 --scale large --num-samples 20
#   sbatch scripts/slurm/tune_sweep.sh fusion      --dataset set_01 --scale large --num-samples 15
#
# The first positional arg is the stage name (autoencoder, curriculum, fusion).
# All subsequent args are forwarded to `python -m graphids.pipeline.cli tune`.
#
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/tune_sweep.sh ...

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=kd-gat-tune
#SBATCH --output=slurm_logs/tune_%j.out
#SBATCH --error=slurm_logs/tune_%j.err
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

# Stage data to fast storage (NFS → scratch → TMPDIR)
source scripts/data/stage_data.sh --cache

# Extract stage from first positional arg, forward the rest
TUNE_STAGE="${1:?Usage: tune_sweep.sh <stage> [--dataset ...] [--scale ...] [--num-samples ...]}"
shift

echo "=== Ray Tune Sweep ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Stage:     ${TUNE_STAGE}"
echo "Args:      $*"
echo "Python:    $(which python)"
echo "Ray:       $(python -c 'import ray; print(ray.__version__)')"
echo ""

# Map stage name to model type for --model flag
case "${TUNE_STAGE}" in
    autoencoder) MODEL="vgae" ;;
    curriculum|normal) MODEL="gat" ;;
    fusion) MODEL="dqn" ;;
    *) echo "Unknown stage: ${TUNE_STAGE}"; exit 1 ;;
esac

# Run tune via CLI
python -m graphids.pipeline.cli tune \
    --model "${TUNE_STAGE}" \
    "$@"

EXIT_CODE=$?

# --- Post-job ---
# Sync datalake Parquet to S3 (backup)
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

# Sync offline W&B runs (only from this job, not all historical runs)
if command -v wandb &>/dev/null; then
    echo "Syncing offline W&B runs..."
    find wandb/ -maxdepth 1 -name "run-*" -newer slurm_logs/tune_${SLURM_JOB_ID}.out \
        -exec wandb sync {} \; 2>/dev/null || true
fi

# GPU utilization report
source scripts/slurm/job_epilog.sh

exit $EXIT_CODE
