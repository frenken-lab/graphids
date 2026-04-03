#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=0:30:00
#SBATCH --account=PAS1266
#SBATCH --job-name=extract-fusion-states
#SBATCH --output=/fs/ess/PAS1266/kd-gat/slurm_logs/extract_fusion_%j.out
#SBATCH --error=/fs/ess/PAS1266/kd-gat/slurm_logs/extract_fusion_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_preamble.sh"

LAKE=/fs/ess/PAS1266/kd-gat/dev/rf15/set_01

# --- Small scale: vgae_ff9f9014 + gat_bf2a5575 ---
echo "=== Extracting small-scale fusion states ==="
python -m graphids extract-fusion-states \
    --vgae-ckpt "$LAKE/vgae_small_autoencoder_ff9f9014/seed_42/checkpoints/best_model.ckpt" \
    --gat-ckpt "$LAKE/gat_small_curriculum_bf2a5575/seed_42/checkpoints/best_model.ckpt" \
    --dataset set_01 \
    --output-dir "$LAKE/fusion_states/small_ff9f9014_bf2a5575"

# --- Large scale: vgae_9ffb88b1 + gat_e9354ccd ---
echo "=== Extracting large-scale fusion states ==="
python -m graphids extract-fusion-states \
    --vgae-ckpt "$LAKE/vgae_large_autoencoder_9ffb88b1/seed_42/checkpoints/best_model.ckpt" \
    --gat-ckpt "$LAKE/gat_large_curriculum_e9354ccd/seed_42/checkpoints/best_model.ckpt" \
    --dataset set_01 \
    --output-dir "$LAKE/fusion_states/large_9ffb88b1_e9354ccd"

echo "=== Done ==="
source "$SCRIPT_DIR/_epilog.sh"
