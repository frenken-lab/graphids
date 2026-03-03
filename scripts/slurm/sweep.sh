#!/bin/bash
# Account comes from .env (KD_GAT_SLURM_ACCOUNT). Submit with:
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/sweep.sh
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=5
#SBATCH --time=08:00:00
#SBATCH --job-name=kd-gat-sweep
#SBATCH --output=experimentruns/sweep_%j.out

# Hyperparameter sweep via OSC's parallel-command-processor.
# Each worker gets 1 GPU; the manager coordinates distribution.
#
# Usage:
#   # Generate commands then submit
#   python scripts/dev/generate_sweep.py \
#     --stage autoencoder --model vgae --scale large --dataset hcrl_sa \
#     --sweep "training.lr=0.001,0.0005" "vgae.latent_dim=8,16,32" \
#     --output /tmp/sweep_commands.txt
#   sbatch scripts/slurm/sweep.sh /tmp/sweep_commands.txt
#
#   # Or inline (generates + runs)
#   sbatch scripts/slurm/sweep.sh <(python scripts/dev/generate_sweep.py \
#     --stage autoencoder --model vgae --scale large --dataset hcrl_sa \
#     --sweep "training.lr=0.001,0.0005" "vgae.latent_dim=8,16,32")

set -euo pipefail

COMMANDS_FILE="${1:?Usage: sbatch scripts/slurm/sweep.sh <commands_file>}"

if [[ ! -f "$COMMANDS_FILE" ]]; then
    echo "ERROR: Commands file not found: $COMMANDS_FILE"
    exit 1
fi

echo "=== KD-GAT Hyperparameter Sweep ==="
echo "Commands file: $COMMANDS_FILE"
echo "Number of configs: $(wc -l < "$COMMANDS_FILE")"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

srun parallel-command-processor "$COMMANDS_FILE"

echo "=== Sweep complete ==="
