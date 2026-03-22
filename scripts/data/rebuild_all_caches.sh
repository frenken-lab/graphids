#!/bin/bash
# Rebuild ALL graph caches (train + test) for specified datasets via SLURM.
# Each dataset gets its own SLURM job for parallelism.
#
# Usage: bash scripts/data/rebuild_all_caches.sh                          # All 6 datasets
#        bash scripts/data/rebuild_all_caches.sh hcrl_ch hcrl_sa          # Specific datasets
#        bash scripts/data/rebuild_all_caches.sh --dry-run                # Show what would run

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

# Source .env for KD_GAT_SLURM_ACCOUNT and KD_GAT_LAKE_ROOT
set -a; source "$PROJECT_DIR/.env" 2>/dev/null; set +a

DRY_RUN=false
DATASETS=""

for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    else
        DATASETS="$DATASETS $arg"
    fi
done

if [ -z "$DATASETS" ]; then
    DATASETS="hcrl_ch hcrl_sa set_01 set_02 set_03 set_04"
fi

ACCOUNT="${KD_GAT_SLURM_ACCOUNT:?Set KD_GAT_SLURM_ACCOUNT in .env}"
LAKE_ROOT="${KD_GAT_LAKE_ROOT:-/fs/ess/PAS1266/kd-gat}"

for ds in $DATASETS; do
    CACHE_DIR=$("$PROJECT_DIR/.venv/bin/python" -c "from graphids.config import cache_dir; print(cache_dir('${LAKE_ROOT}', '$ds'))")

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] Would delete $CACHE_DIR and submit rebuild job for $ds"
        continue
    fi

    # Delete existing cache to force rebuild
    if [ -d "$CACHE_DIR" ]; then
        echo "Deleting stale cache: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
    fi

    sbatch --account="$ACCOUNT" --partition=cpu \
      --time=02:00:00 --mem=48G --cpus-per-task=8 \
      --job-name="cache-${ds}" \
      --output="$PROJECT_DIR/slurm_logs/%j-cache-${ds}.out" \
      --error="$PROJECT_DIR/slurm_logs/%j-cache-${ds}.err" \
      --wrap="SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source $PROJECT_DIR/scripts/slurm/_preamble.sh && python -c \"
from graphids.core.preprocessing.datamodule import CANBusDataModule
from graphids.config import resolve

ds = '${ds}'
cfg = resolve('model_type=vgae', 'scale=large', f'dataset={ds}')
dm = CANBusDataModule.from_cfg(cfg)

print(f'=== Rebuilding train/val cache for {ds} ===', flush=True)
dm.setup('fit')
print(f'  Train: {len(dm.train_dataset)}, Val: {len(dm.val_dataset)}, IDs: {dm.num_ids}', flush=True)
print(f'  Features: {dm.in_channels} node dims, {dm.edge_dim} edge dims', flush=True)

print(f'=== Rebuilding test caches for {ds} ===', flush=True)
dm.setup('test')
for name, test_ds in dm.test_datasets.items():
    print(f'  {name}: {len(test_ds)} graphs', flush=True)
print(f'=== Done: {ds} ===', flush=True)
\""
    echo "Submitted cache rebuild job for ${ds}"
done
