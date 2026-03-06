#!/usr/bin/env bash
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT --partition=gpu --gres=gpu:v100:1 --mem=32G scripts/slurm/loss_landscape.sh ...
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=kd-gat-landscape
#SBATCH --output=slurm_logs/landscape_%j.out
#SBATCH --error=slurm_logs/landscape_%j.err

# Compute 2D loss landscapes for trained models.
#
# VGAE/GAT need GPU (graph neural network forward passes):
#   sbatch --partition=gpu --gres=gpu:v100:1 --mem=32G scripts/slurm/loss_landscape.sh vgae set_01
#   sbatch --partition=gpu --gres=gpu:v100:1 --mem=32G scripts/slurm/loss_landscape.sh gat set_01
#
# DQN is a small MLP — CPU-only, minimal memory:
#   sbatch --partition=serial --mem=8G scripts/slurm/loss_landscape.sh dqn set_01
#
# All three sequentially on GPU (simplest):
#   sbatch --partition=gpu --gres=gpu:v100:1 --mem=32G scripts/slurm/loss_landscape.sh all set_01

source "$(dirname "$0")/_preamble.sh"

MODEL="${1:?Usage: loss_landscape.sh <model|all> <dataset>}"
DATASET="${2:?Usage: loss_landscape.sh <model|all> <dataset>}"
RESOLUTION="${3:-51}"
SCALE="${4:-1.0}"

echo "=== Loss Landscape Computation ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Model:      ${MODEL}"
echo "Dataset:    ${DATASET}"
echo "Resolution: ${RESOLUTION}"
echo "Scale:      ${SCALE}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Memory:     ${SLURM_MEM_PER_NODE:-unknown}MB"
echo "Python:     $(which python)"
echo ""

MODELS=()
if [[ "$MODEL" == "all" ]]; then
    # Job array support: map SLURM_ARRAY_TASK_ID to model
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        ALL_MODELS=(vgae gat dqn)
        MODELS=("${ALL_MODELS[$SLURM_ARRAY_TASK_ID]}")
    else
        MODELS=(vgae gat dqn)
    fi
else
    MODELS=("$MODEL")
fi

FAILURES=0
for m in "${MODELS[@]}"; do
    echo "--- Computing loss landscape for ${m} on ${DATASET} ---"
    if python -m graphids.pipeline.stages.loss_landscape \
        --model "$m" \
        --dataset "$DATASET" \
        --resolution "$RESOLUTION" \
        --scale "$SCALE"; then
        echo "--- Done: ${m} ---"
    else
        echo "--- FAILED: ${m} (continuing with remaining models) ---" >&2
        FAILURES=$((FAILURES + 1))
    fi
    echo ""
done

echo "=== Loss landscape complete (${FAILURES} failures) ==="
exit "$FAILURES"
