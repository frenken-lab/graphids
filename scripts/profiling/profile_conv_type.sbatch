#!/usr/bin/env bash
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/profiling/profile_conv_type.sbatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=02:00:00
#SBATCH --job-name=kd-gat-profile-conv
#SBATCH --output=slurm_logs/profile_conv_%j.out
#SBATCH --error=slurm_logs/profile_conv_%j.err

# Profile GAT (no edge_attr) vs TransformerConv (with edge_attr) to decide
# whether cuGraph acceleration is worthwhile.
#
# Usage: sbatch scripts/profiling/profile_conv_type.sbatch [dataset] [scale]
# Then:  python scripts/profiling/analyze_profile.py

set -euo pipefail

DATASET="${1:-hcrl_sa}"
SCALE="${2:-large}"
EPOCHS=5

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Load environment
module load python/3.12
source .venv/bin/activate

mkdir -p slurm_logs

echo "=== Profiling conv_type comparison: ${DATASET} / ${SCALE} ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Run 1: GAT (no edge_attr)
echo ""
echo "--- Run 1: conv_type=gat (no edge_attr) ---"
python -m graphids.pipeline.cli curriculum \
    --model gat --scale "${SCALE}" --dataset "${DATASET}" \
    -O training.max_epochs "${EPOCHS}" \
    -O training.profile true \
    -O training.profile_steps 5 \
    -O gat.conv_type gat \
    -O training.patience 999

echo ""
echo "--- Run 2: conv_type=transformer (with edge_attr) ---"
python -m graphids.pipeline.cli curriculum \
    --model gat --scale "${SCALE}" --dataset "${DATASET}" \
    -O training.max_epochs "${EPOCHS}" \
    -O training.profile true \
    -O training.profile_steps 5 \
    -O gat.conv_type transformer \
    -O training.patience 999

echo ""
echo "=== Profiling complete. Run: python scripts/profiling/analyze_profile.py ==="
echo "Date: $(date)"
