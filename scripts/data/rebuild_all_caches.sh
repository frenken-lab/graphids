#!/bin/bash
# Rebuild ALL graph caches (train + test) for specified datasets via SLURM.
# Deletes existing caches first to force a clean rebuild with current preprocessing version.
#
# Usage: bash scripts/data/rebuild_all_caches.sh                          # All 6 datasets
#        bash scripts/data/rebuild_all_caches.sh hcrl_ch hcrl_sa          # Specific datasets
#        bash scripts/data/rebuild_all_caches.sh --dry-run                # Show what would run

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$PROJECT_DIR/slurm_logs"

# Source .env for KD_GAT_SLURM_ACCOUNT
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

for ds in $DATASETS; do
    CACHE_DIR="$PROJECT_DIR/data/cache/$ds"

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] Would delete $CACHE_DIR and submit rebuild job for $ds"
        continue
    fi

    # Delete existing cache to force rebuild
    if [ -d "$CACHE_DIR" ]; then
        echo "Deleting stale cache: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
    fi

    # Submit train cache rebuild
    sbatch --account="$ACCOUNT" --partition=cpu \
      --time=360 --mem=85G --cpus-per-task=8 \
      --job-name="cache-${ds}" \
      --output="$PROJECT_DIR/slurm_logs/%j-cache-${ds}.out" \
      --error="$PROJECT_DIR/slurm_logs/%j-cache-${ds}.err" \
      --wrap="source $PROJECT_DIR/.venv/bin/activate && cd $PROJECT_DIR && python -c \"
from graphids.core.training.datamodules import load_dataset, load_test_scenarios
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
ds = '${ds}'
ds_path = Path(f'data/automotive/{ds}')
cache_path = Path(f'data/cache/{ds}')

print(f'=== Rebuilding train cache for {ds} ===', flush=True)
train, val, num_ids = load_dataset(ds, ds_path, cache_path, force_rebuild_cache=True)
print(f'  Train: {len(train)}, Val: {len(val)}, IDs: {num_ids}', flush=True)

print(f'=== Rebuilding test caches for {ds} ===', flush=True)
scenarios = load_test_scenarios(ds, ds_path, cache_path, force_rebuild_cache=True)
for name, graphs in scenarios.items():
    print(f'  {name}: {len(graphs)} graphs', flush=True)
print(f'=== Done: {ds} ===', flush=True)
\""
    echo "Submitted cache rebuild job for ${ds}"
done
