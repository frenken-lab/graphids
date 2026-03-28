#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --job-name=kd-profile-fastcollate
#SBATCH --output=slurm_logs/%j/%j_0_log.out
#SBATCH --error=slurm_logs/%j/%j_0_log.err
#SBATCH --signal=B:USR1@300
#SBATCH --account=PAS1266
set -euo pipefail

# Same config as baseline profile job 45984077, but with fast_collate wired in.
# Baseline: 30.4% mean GPU util, 20 epochs, set_02, vgae small.
# This run: 10 epochs (enough to measure steady-state GPU util).

export STAGE_DATA_ARGS="--cache --dataset set_02"
source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh"

python -m graphids \
    stage=autoencoder \
    model_type=vgae \
    scale=small \
    dataset=set_02 \
    training.max_epochs=10 \
    seed=42

JOB_LOG_PREFIX="profile-fastcollate" source "/users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh"
