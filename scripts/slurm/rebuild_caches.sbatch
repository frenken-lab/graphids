#!/usr/bin/env bash
# Rebuild preprocessed graph caches for all datasets after preprocessing changes.
# CPU-only job — no GPU needed. Rebuilds sequentially to avoid memory pressure.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=kd-gat-cache
#SBATCH --output=slurm_logs/cache_%j.out
#SBATCH --error=slurm_logs/cache_%j.err

SCRIPT_DIR="/users/PAS2022/rf15/KD-GAT/scripts/slurm"
SKIP_CUDA_CONF=1 source "$SCRIPT_DIR/_preamble.sh"

DATASETS=(hcrl_ch hcrl_sa set_01 set_02 set_03 set_04)

echo "=== Cache Rebuild: PREPROCESSING_VERSION=$(python -c 'from graphids.config.constants import PREPROCESSING_VERSION; print(PREPROCESSING_VERSION)') ==="

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== Rebuilding cache: $ds ==="
    start_time=$SECONDS

    python -c "
from graphids.core.training.datamodules import load_dataset
from graphids.config import data_dir, cache_dir, resolve

cfg = resolve('vgae', 'large', dataset='$ds')
train, val, num_ids = load_dataset(
    '$ds',
    dataset_path=data_dir(cfg),
    cache_dir_path=cache_dir(cfg),
    force_rebuild_cache=True,
)
print(f'  $ds: {len(train)+len(val)} graphs, {num_ids} unique IDs')
print(f'  Features: {train[0].x.shape[1]} dims')

# Quick feature range check — all should be [0, 1] after normalization
import torch
all_x = torch.cat([train[i].x for i in range(min(500, len(train)))])
for idx, name in [(17, 'Entropy'), (20, 'Skewness'), (21, 'Kurtosis'), (23, 'SplitHalf')]:
    vals = all_x[:, idx]
    print(f'  {name} (feat {idx}): min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}')
"
    elapsed=$((SECONDS - start_time))
    echo "  Completed in ${elapsed}s"
done

echo ""
echo "=== All caches rebuilt ==="

# Invalidate scratch staging marker so next GPU job re-stages
if [ -f "/fs/scratch/PAS1266/kd-gat-data/cache/.staged_marker" ]; then
    rm "/fs/scratch/PAS1266/kd-gat-data/cache/.staged_marker"
    echo "Invalidated scratch staging marker (will re-stage on next GPU job)"
fi
