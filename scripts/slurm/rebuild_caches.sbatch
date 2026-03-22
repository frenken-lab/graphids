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

PREPROCESSING_VERSION=$(python -c 'from graphids.config.constants import PREPROCESSING_VERSION; print(PREPROCESSING_VERSION)')
echo "=== Cache Rebuild: PREPROCESSING_VERSION=${PREPROCESSING_VERSION} ==="

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== Rebuilding cache: $ds ==="
    start_time=$SECONDS

    # Delete existing cache to force rebuild (InMemoryDataset skips if .complete marker exists)
    CACHE_DIR=$(python -c "from graphids.config import cache_dir; print(cache_dir('${KD_GAT_LAKE_ROOT:-/fs/ess/PAS1266/kd-gat}', '$ds'))")
    if [ -d "$CACHE_DIR" ]; then
        echo "  Removing stale cache: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
    fi

    python -c "
from graphids.core.preprocessing.datamodule import CANBusDataModule
from graphids.config import resolve

cfg = resolve('model_type=vgae', 'scale=large', 'dataset=$ds')
dm = CANBusDataModule.from_cfg(cfg)
dm.setup('fit')

train = dm.train_dataset
val = dm.val_dataset
print(f'  $ds: {len(train)+len(val)} graphs, {dm.num_ids} unique IDs')
print(f'  Features: {dm.in_channels} node dims, {dm.edge_dim} edge dims')

# Quick feature range check
import torch
all_x = torch.cat([train[i].x for i in range(min(500, len(train)))])
for idx, name in [(17, 'Entropy'), (20, 'Skewness'), (21, 'Kurtosis'), (23, 'SplitHalf')]:
    if idx < all_x.shape[1]:
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
