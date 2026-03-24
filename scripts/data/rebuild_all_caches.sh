#!/bin/bash
# Rebuild ALL graph caches for specified datasets via SLURM.
# Usage: bash scripts/data/rebuild_all_caches.sh [--dry-run] [dataset ...]
set -euo pipefail

source "$(dirname "$0")/../lib/datasets.sh"
source "$(dirname "$0")/../lib/dryrun.sh"
source "$(dirname "$0")/../lib/slurm.sh"

kd_parse_dry_run "$@"
read -ra DATASETS <<< "$(kd_parse_datasets "$@")"
LAKE_ROOT="${KD_GAT_LAKE_ROOT:-/fs/ess/PAS1266/kd-gat}"

rebuild_one() {
    local ds="$1"
    local cache_dir
    cache_dir=$("${KD_PROJECT_ROOT}/.venv/bin/python" -c \
        "from graphids.config import cache_dir; print(cache_dir('${LAKE_ROOT}', '$ds'))")

    if [[ -d "$cache_dir" ]]; then
        kd_log INFO "Deleting stale cache" dataset="$ds" path="$cache_dir"
        kd_exec rm -rf "$cache_dir"
    fi

    kd_submit cpu "cache-${ds}" \
        "SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1 source ${KD_PROJECT_ROOT}/scripts/slurm/_preamble.sh && python -c \"
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
\"" \
        "--time=02:00:00 --mem=48G --cpus-per-task=8"

    kd_log INFO "Submitted cache rebuild" dataset="$ds"
}

kd_each_dataset rebuild_one "${DATASETS[@]}"
