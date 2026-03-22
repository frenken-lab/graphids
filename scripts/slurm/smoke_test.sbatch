#!/usr/bin/env bash
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=kd-gat-smoke
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err
#SBATCH --signal=B:USR1@300

# Layer 2 smoke test: 1 stage, 2 epochs, smallest dataset.
# Usage: sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/smoke_test.sbatch [stage]
# Default stage: autoencoder

STAGE="${1:-autoencoder}"
source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh

log_job_header "Smoke test: $STAGE"

case "$STAGE" in
    autoencoder)
        python -m graphids stage=autoencoder model_type=vgae scale=small dataset=hcrl_ch \
            training.max_epochs=2 training.patience=5
        ;;
    curriculum)
        python -m graphids stage=curriculum model_type=gat scale=small dataset=hcrl_ch \
            training.max_epochs=2 training.patience=5
        ;;
    fusion)
        python -m graphids stage=fusion model_type=dqn scale=small dataset=hcrl_ch \
            fusion.episodes=10
        ;;
    evaluation)
        python -m graphids stage=evaluation model_type=vgae scale=small dataset=hcrl_ch
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid: autoencoder, curriculum, fusion, evaluation"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "Stage '$STAGE' exit code: $EXIT_CODE"

JOB_LOG_PREFIX="smoke" source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_epilog.sh
exit $EXIT_CODE
