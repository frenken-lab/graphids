#!/usr/bin/env bash
# Profile GAT vs TransformerConv to decide whether cuGraph acceleration is worthwhile.
#
# Usage: sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/profiling/profile_conv_type.sbatch [dataset] [scale]
# Then:  python scripts/profiling/analyze_profile.py
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=kd-gat-profile-conv
#SBATCH --output=slurm_logs/profile_conv_%j.out
#SBATCH --error=slurm_logs/profile_conv_%j.err
#SBATCH --signal=B:USR1@300

DATASET="${1:-hcrl_sa}"
SCALE="${2:-large}"
EPOCHS=5

source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh

log_job_header "Conv type profiling: ${DATASET} / ${SCALE}"

echo "--- Run 1: conv_type=gat (no edge_attr) ---"
python -m graphids stage=curriculum model_type=gat scale=${SCALE} dataset=${DATASET} \
    training.max_epochs=${EPOCHS} \
    training.profile=true \
    training.profile_steps=5 \
    gat.conv_type=gat \
    training.patience=999

echo ""
echo "--- Run 2: conv_type=transformer (with edge_attr) ---"
python -m graphids stage=curriculum model_type=gat scale=${SCALE} dataset=${DATASET} \
    training.max_epochs=${EPOCHS} \
    training.profile=true \
    training.profile_steps=5 \
    gat.conv_type=transformer \
    training.patience=999

echo ""
echo "=== Profiling complete. Run: python scripts/profiling/analyze_profile.py ==="

JOB_LOG_PREFIX="profile_conv" source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh
