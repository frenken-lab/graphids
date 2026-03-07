#!/usr/bin/env bash
# Layer 2: GPU smoke test on gpudebug partition.
# Runs 1 tune trial with 2 epochs on hcrl_ch (smallest dataset).
# Validates GPU access, data loading, model construction, and training loop.
#
# Usage:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/smoke_test.sh autoencoder
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/smoke_test.sh curriculum
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/smoke_test.sh fusion
#
# Expected: starts within minutes (priority scheduling), completes in <10 min.
# If this fails, do NOT submit to gpu partition — fix the error first.

#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=kd-gat-smoke
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err

source "$(dirname "$0")/_preamble.sh"

# Extract stage from first positional arg
SMOKE_STAGE="${1:?Usage: smoke_test.sh <stage>}"

log_job_header "GPU Smoke Test"
echo "Stage:     ${SMOKE_STAGE}"
echo "Partition: gpudebug (1hr max)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Run 1 trial, 2 epochs, smallest dataset
python -m graphids.pipeline.cli tune \
    --model "${SMOKE_STAGE}" \
    --dataset hcrl_ch \
    --scale large \
    --num-samples 1 \
    --tune-epochs 2 \
    --tune-patience 2

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "=== SMOKE TEST PASSED ==="
    echo "Safe to submit production job to gpu partition."
else
    echo ""
    echo "=== SMOKE TEST FAILED (exit code: $EXIT_CODE) ==="
    echo "Fix errors above before submitting to gpu partition."
fi

exit $EXIT_CODE
