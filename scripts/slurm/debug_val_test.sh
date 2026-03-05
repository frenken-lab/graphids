#!/usr/bin/env bash
# Quick debug test: validation + NaN diagnosis
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=kd-gat-dbg
#SBATCH --output=slurm_logs/dbg_%j.out
#SBATCH --error=slurm_logs/dbg_%j.err

set -euo pipefail
cd "/users/PAS2022/rf15/KD-GAT"
mkdir -p slurm_logs
module load python/3.12
source .venv/bin/activate
set -a; source .env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/data/stage_data.sh --cache

echo "=== Test 1: VGAE fp32 (2 epochs) ==="
python -m graphids.pipeline.cli autoencoder \
    --model vgae --scale large --dataset hcrl_ch \
    -O training.max_epochs 2 \
    -O training.patience 5 \
    -O training.log_every_n_steps 1 \
    -O training.precision 32-true

echo "EXIT CODE: $?"
echo "--- metrics.json ---"
cat experimentruns/hcrl_ch/vgae_large_autoencoder/metrics.json 2>/dev/null

echo ""
echo "=== Test 2: VGAE fp16-mixed (2 epochs) ==="
python -m graphids.pipeline.cli autoencoder \
    --model vgae --scale large --dataset hcrl_ch \
    -O training.max_epochs 2 \
    -O training.patience 5 \
    -O training.log_every_n_steps 1 \
    -O training.precision 16-mixed

echo "EXIT CODE: $?"
echo "--- metrics.json ---"
cat experimentruns/hcrl_ch/vgae_large_autoencoder/metrics.json 2>/dev/null

echo ""
echo "--- CSV logs (latest) ---"
find experimentruns/hcrl_ch/vgae_large_autoencoder/csv_logs/ -name "*.csv" -newer scripts/slurm/debug_val_test.sh -exec echo {} \; -exec head -5 {} \; 2>/dev/null
