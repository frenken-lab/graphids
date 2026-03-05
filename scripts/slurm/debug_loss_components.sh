#!/usr/bin/env bash
# Debug: track per-component losses during training
#SBATCH --partition=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=kd-gat-ldbg
#SBATCH --output=slurm_logs/ldbg_%j.out
#SBATCH --error=slurm_logs/ldbg_%j.err

set -euo pipefail
cd "/users/PAS2022/rf15/KD-GAT"
mkdir -p slurm_logs
module load python/3.12
source .venv/bin/activate
set -a; source .env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/data/stage_data.sh --cache

echo "=== VGAE fp32 with component logging (5 epochs) ==="
python -m graphids.pipeline.cli autoencoder \
    --model vgae --scale large --dataset hcrl_ch \
    -O training.max_epochs 5 \
    -O training.patience 10 \
    -O training.log_every_n_steps 1 \
    -O training.precision 32-true \
    -O training.dynamic_batching false \
    -O training.batch_size 32

echo "EXIT CODE: $?"
echo "--- metrics.json ---"
cat experimentruns/hcrl_ch/vgae_large_autoencoder/metrics.json 2>/dev/null

echo ""
echo "--- CSV logs ---"
for f in experimentruns/hcrl_ch/vgae_large_autoencoder/csv_logs/version_0/*.csv; do
    echo "=== $f ==="
    head -10 "$f"
done
