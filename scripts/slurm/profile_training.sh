#!/bin/bash
#SBATCH --job-name=kd-gat-profile
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=36G
#SBATCH --cpus-per-task=4
#SBATCH --account=PAS1266
#SBATCH --signal=B:USR1@60
#SBATCH --output=slurm_logs/profile_%j.out
#SBATCH --error=slurm_logs/profile_%j.err

# Profiling job: 5 epochs with PyTorchProfiler + DeviceStatsMonitor.
# Uses profile overlay on top of any stage+scale config.
#
# Default: small VGAE on hcrl_ch.
# Override via env vars:
#   PROFILE_STAGE=normal PROFILE_SCALE=small_gat PROFILE_DATASET=set_01 \
#     sbatch scripts/slurm/profile_training.sh

source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

STAGE="${PROFILE_STAGE:-autoencoder}"
SCALE="${PROFILE_SCALE:-small_vgae}"
DATASET="${PROFILE_DATASET:-hcrl_ch}"

STAGES_DIR="graphids/config/stages"
OVERLAYS_DIR="graphids/config/overlays"

echo "=== Profile Run ==="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Stage:    $STAGE"
echo "Scale:    $SCALE"
echo "Dataset:  $DATASET"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "==================="

python -m graphids fit \
    --config "$STAGES_DIR/${STAGE}.yaml" \
    --config "$OVERLAYS_DIR/${SCALE}.yaml" \
    --config "$OVERLAYS_DIR/profile.yaml" \
    --data.init_args.dataset="$DATASET"

EXIT=$?
echo "=== Exit: $EXIT ==="

JOB_LOG_PREFIX="profile" source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh"
