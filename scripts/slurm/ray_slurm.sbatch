#!/usr/bin/env bash
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/ray_slurm.sbatch ...
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --job-name=kd-gat-ray
#SBATCH --output=slurm_logs/ray_%j.out
#SBATCH --error=slurm_logs/ray_%j.err
#SBATCH --signal=B:USR1@300

# Ray-based pipeline execution on SLURM.
#
# For single-node (default):
#   sbatch scripts/slurm/ray_slurm.sbatch flow --dataset hcrl_sa
#   sbatch scripts/slurm/ray_slurm.sbatch autoencoder --model vgae --scale large --dataset hcrl_sa
#
# For multi-node (pass extra #SBATCH --nodes=N):
#   sbatch --nodes=2 scripts/slurm/ray_slurm.sbatch flow --dataset hcrl_sa

SCRIPT_DIR="/users/PAS2022/rf15/KD-GAT/scripts/slurm"
source "$SCRIPT_DIR/_preamble.sh"

log_job_header "Ray Pipeline"
echo "Nodes:     ${SLURM_NNODES:-1}"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "Ray:       $(python -c 'import ray; print(ray.__version__)')"

# --- Launch ---
ENTRYPOINT_ARGS=("$@")

if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    ray symmetric-run \
        --num-nodes="${SLURM_NNODES}" \
        -- python -m graphids.pipeline.cli "${ENTRYPOINT_ARGS[@]}"
else
    python -m graphids.pipeline.cli "${ENTRYPOINT_ARGS[@]}"
fi

EXIT_CODE=$?

# Report orphaned/failed runs (informational only)
echo "Checking for orphaned runs..."
bash scripts/data/cleanup_orphans.sh --dry-run || true

# --- Post-job ---
source "$SCRIPT_DIR/_epilog.sh"

exit $EXIT_CODE
