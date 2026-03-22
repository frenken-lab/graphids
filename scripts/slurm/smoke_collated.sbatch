#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=smoke-collated
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err

# Smoke test: verify collated tensor storage enables num_workers=2
# and check GPU utilization with 10% of default epochs.

SCRIPT_DIR="/users/PAS2022/rf15/KD-GAT/scripts/slurm"
source "$SCRIPT_DIR/_preamble.sh"

log_job_header "Smoke Test: collated storage + num_workers"

echo "=== Stage 1: VGAE autoencoder ==="
python -m graphids stage=autoencoder model_type=vgae scale=large dataset=hcrl_sa \
    training.max_epochs=30 training.patience=10
VGAE_EXIT=$?
echo "VGAE exit code: $VGAE_EXIT"

if [[ $VGAE_EXIT -ne 0 ]]; then
    echo "VGAE failed, skipping GAT"
    source "$SCRIPT_DIR/_epilog.sh"
    exit $VGAE_EXIT
fi

echo ""
echo "=== Stage 2: GAT curriculum ==="
python -m graphids stage=curriculum model_type=gat scale=large dataset=hcrl_sa \
    training.max_epochs=30 training.patience=10
GAT_EXIT=$?
echo "GAT exit code: $GAT_EXIT"

source "$SCRIPT_DIR/_epilog.sh"
exit $GAT_EXIT
