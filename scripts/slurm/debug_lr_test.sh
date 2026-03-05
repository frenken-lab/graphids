#!/usr/bin/env bash
# Test VGAE with different learning rates to find stable range
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=kd-gat-lr
#SBATCH --output=slurm_logs/lr_%j.out
#SBATCH --error=slurm_logs/lr_%j.err

set -euo pipefail
cd "/users/PAS2022/rf15/KD-GAT"
mkdir -p slurm_logs
module load python/3.12
source .venv/bin/activate
set -a; source .env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/data/stage_data.sh --cache

for lr in 0.0001 0.0005 0.001 0.002; do
    echo ""
    echo "=== LR=$lr fp32 (3 epochs) ==="
    python -m graphids.pipeline.cli autoencoder \
        --model vgae --scale large --dataset hcrl_ch \
        -O training.max_epochs 3 \
        -O training.patience 5 \
        -O training.log_every_n_steps 1 \
        -O training.precision 32-true \
        -O training.lr $lr
    echo "EXIT CODE: $?"
    echo "--- metrics.json ---"
    cat experimentruns/hcrl_ch/vgae_large_autoencoder/metrics.json 2>/dev/null
done

echo ""
echo "=== LR=0.0005 fp16-mixed (3 epochs) ==="
python -m graphids.pipeline.cli autoencoder \
    --model vgae --scale large --dataset hcrl_ch \
    -O training.max_epochs 3 \
    -O training.patience 5 \
    -O training.log_every_n_steps 1 \
    -O training.precision 16-mixed \
    -O training.lr 0.0005
echo "EXIT CODE: $?"
echo "--- metrics.json ---"
cat experimentruns/hcrl_ch/vgae_large_autoencoder/metrics.json 2>/dev/null
