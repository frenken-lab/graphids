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

source "$(dirname "$0")/_preamble.sh"

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

# Run tune via CLI
python -m graphids.pipeline.cli tune \
    --model "${TUNE_STAGE}" \
    "$@"

EXIT_CODE=$?

# --- Post-job ---
source "$(dirname "$0")/_epilog.sh"

exit $EXIT_CODE
