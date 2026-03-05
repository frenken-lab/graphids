#!/usr/bin/env bash
# Minimal standalone v2 training validation.
# Tests that the base training pipeline works with v2 preprocessed graphs
# BEFORE adding tune/sweep complexity on top.
#
# Usage:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/standalone_v2_test.sh

#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=kd-gat-v2test
#SBATCH --output=slurm_logs/v2test_%j.out
#SBATCH --error=slurm_logs/v2test_%j.err

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

module load python/3.12
source .venv/bin/activate

set -a; source .env; set +a

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/data/stage_data.sh --cache

echo "=== Standalone V2 Training Test ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Test 1: VGAE autoencoder on hcrl_ch (smallest), 5 epochs
echo "--- Test 1: VGAE autoencoder (hcrl_ch, 5 epochs) ---"
python -m graphids.pipeline.cli autoencoder \
    --model vgae --scale large --dataset hcrl_ch \
    -O training.max_epochs 5 \
    -O training.patience 5
echo "VGAE autoencoder: EXIT CODE $?"

# Test 2: GAT curriculum on hcrl_ch (needs VGAE checkpoint from test 1)
echo "--- Test 2: GAT curriculum (hcrl_ch, 5 epochs) ---"
python -m graphids.pipeline.cli curriculum \
    --model gat --scale large --dataset hcrl_ch \
    -O training.max_epochs 5 \
    -O training.patience 5
echo "GAT curriculum: EXIT CODE $?"

echo ""
echo "=== Standalone V2 Training Test Complete ==="

# Check what was produced
echo "--- Outputs ---"
find experimentruns/hcrl_ch/ -name "best_model.pt" -newer scripts/slurm/standalone_v2_test.sh -ls 2>/dev/null
find experimentruns/hcrl_ch/ -name "metrics.json" -newer scripts/slurm/standalone_v2_test.sh -exec echo {} \; -exec cat {} \; 2>/dev/null
