#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --job-name=kd-profile-4w
#SBATCH --output=slurm_logs/%j/%j_0_log.out
#SBATCH --error=slurm_logs/%j/%j_0_log.err
#SBATCH --signal=B:USR1@300
#SBATCH --account=PAS1266
set -euo pipefail

# Test: 4 workers with mmap-based _FastCollate. Baseline 2w was 21.5GB.
# Expect ~24-25GB. 36GB allocation should hold.

export STAGE_DATA_ARGS="--cache --dataset set_02"
source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

python -m graphids \
    stage=autoencoder \
    model_type=vgae \
    scale=small \
    dataset=set_02 \
    training.max_epochs=10 \
    num_workers=4 \
    seed=42

JOB_LOG_PREFIX="profile-4w" source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh"
