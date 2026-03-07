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

source "$(dirname "$0")/_preamble.sh"

# Extract dataset and scale from positional args, forward the rest
DATASET="${1:?Usage: sweep_pipeline.sh <dataset> <scale> [--num-samples N] [--tune-epochs N]}"
SCALE="${2:?Usage: sweep_pipeline.sh <dataset> <scale> [--num-samples N] [--tune-epochs N]}"
shift 2

# SIGUSR1 handler: log warning, Python orchestrator handles state persistence
trap 'echo "SIGUSR1 received — wall time approaching. State file is safe for resume on resubmit."' USR1

log_job_header "Sweep Pipeline DAG"
echo "Dataset:   ${DATASET}"
echo "Scale:     ${SCALE}"
echo "Extra args: $*"

# Run the sweep pipeline (--resume is default)
python -m graphids.pipeline.cli sweep-pipeline \
    --dataset "${DATASET}" \
    --scale "${SCALE}" \
    "$@"

EXIT_CODE=$?

# --- Post-job ---
source "$(dirname "$0")/_epilog.sh"

exit $EXIT_CODE
