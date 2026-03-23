#!/usr/bin/env bash
# Rebuild preprocessed graph caches for all datasets after preprocessing changes.
# CPU-only job array — one task per dataset, all run in parallel.
#
# Usage:
#   JOBID=$(sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/rebuild_caches.sbatch | awk '{print $4}')
#   sbatch --account=$KD_GAT_SLURM_ACCOUNT --dependency=afterany:$JOBID scripts/slurm/invalidate_cache_marker.sh
#
# Or as a one-liner:
#   source .env && JID=$(sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/rebuild_caches.sbatch | awk '{print $4}') && \
#     sbatch --account=$KD_GAT_SLURM_ACCOUNT --dependency=afterany:$JID scripts/slurm/invalidate_cache_marker.sh

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=kd-gat-cache
#SBATCH --output=slurm_logs/cache_%A_%a.out
#SBATCH --error=slurm_logs/cache_%A_%a.err

SCRIPT_DIR="/users/PAS2022/rf15/KD-GAT/scripts/slurm"
SKIP_CUDA_CONF=1 STAGE_DATA_ARGS="--raw" source "$SCRIPT_DIR/_preamble.sh"

DATASETS=(hcrl_ch hcrl_sa set_01 set_02 set_03 set_04)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

PREPROCESSING_VERSION=$(python -c 'from graphids.config.constants import PREPROCESSING_VERSION; print(PREPROCESSING_VERSION)')

log_job_header "Cache rebuild: $DS (task $SLURM_ARRAY_TASK_ID, version $PREPROCESSING_VERSION)"

# Delete existing cache to force rebuild (InMemoryDataset skips if .complete marker exists)
CACHE_DIR=$(python -c "from graphids.config import cache_dir; print(cache_dir('${KD_GAT_LAKE_ROOT:-/fs/ess/PAS1266/kd-gat}', '$DS'))")
if [ -d "$CACHE_DIR" ]; then
    echo "Removing stale cache: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
fi

python -c "
from graphids.core.preprocessing.datamodule import CANBusDataModule
from graphids.config import resolve

cfg = resolve('model_type=vgae', 'scale=large', 'dataset=$DS')
dm = CANBusDataModule.from_cfg(cfg)
dm.setup('fit')

train = dm.train_dataset
val = dm.val_dataset
print(f'  $DS: {len(train)+len(val)} graphs, {dm.num_ids} unique IDs')
print(f'  Features: {dm.in_channels} node dims, {dm.edge_dim} edge dims')

# Quick feature range check
import torch
all_x = torch.cat([train[i].x for i in range(min(500, len(train)))])
for idx, name in [(17, 'Entropy'), (20, 'Skewness'), (21, 'Kurtosis'), (23, 'SplitHalf')]:
    if idx < all_x.shape[1]:
        vals = all_x[:, idx]
        print(f'  {name} (feat {idx}): min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}')
"

EXIT_CODE=$?
log_job_footer $EXIT_CODE
exit $EXIT_CODE
