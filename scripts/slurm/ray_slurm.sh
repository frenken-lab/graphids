#!/usr/bin/env bash
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/ray_slurm.sh ...
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=kd-gat-ray
#SBATCH --output=slurm_logs/ray_%j.out
#SBATCH --error=slurm_logs/ray_%j.err
#SBATCH --signal=B:USR1@300

# Ray-based pipeline execution on SLURM.
#
# For single-node (default):
#   sbatch scripts/slurm/ray_slurm.sh flow --dataset hcrl_sa
#   sbatch scripts/slurm/ray_slurm.sh autoencoder --model vgae --scale large --dataset hcrl_sa
#
# For multi-node (pass extra #SBATCH --nodes=N):
#   sbatch --nodes=2 scripts/slurm/ray_slurm.sh flow --dataset hcrl_sa
#
# The script uses `ray start` on single-node jobs and `ray symmetric-run`
# for multi-node (Ray 2.49+).

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

echo "=== Ray Pipeline ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_NNODES:-1}"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "Python:    $(which python)"
echo "Ray:       $(python -c 'import ray; print(ray.__version__)')"
echo ""

# --- Launch ---
ENTRYPOINT_ARGS=("$@")

if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    # Multi-node: use ray symmetric-run (Ray 2.49+)
    # Each node starts a Ray worker; the first node becomes the head.
    ray symmetric-run \
        --num-nodes="${SLURM_NNODES}" \
        -- python -m graphids.pipeline.cli "${ENTRYPOINT_ARGS[@]}"
else
    # Single-node: start Ray locally, run entrypoint directly
    python -m graphids.pipeline.cli "${ENTRYPOINT_ARGS[@]}"
fi

EXIT_CODE=$?

# --- Post-job ---
# Sync datalake Parquet to S3 (backup)
if [[ -d data/datalake ]] && command -v aws &>/dev/null; then
    echo "Syncing datalake to S3..."
    aws s3 sync data/datalake/ "s3://${KD_GAT_S3_BUCKET:-kd-gat}/datalake/" \
        --exclude "analytics.duckdb" 2>/dev/null || true
fi

# Sync offline W&B runs (only from this job, not all historical runs)
if command -v wandb &>/dev/null; then
    echo "Syncing offline W&B runs..."
    find wandb/ -maxdepth 1 -name "run-*" -newer slurm_logs/ray_${SLURM_JOB_ID}.out \
        -exec wandb sync {} \; 2>/dev/null || true
fi

# Report orphaned/failed runs (informational only)
echo "Checking for orphaned runs..."
bash scripts/data/cleanup_orphans.sh --dry-run || true

# GPU utilization report
source scripts/slurm/job_epilog.sh

exit $EXIT_CODE
