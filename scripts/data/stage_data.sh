#!/usr/bin/env bash
# Stage data from permanent storage to scratch / $TMPDIR for fast job I/O.
#
# Data flow:  ESS (permanent NFS) → Scratch (GPFS, 90-day purge) → TMPDIR (per-job local SSD)
#
# Smart caching: skips ESS→scratch copy if a marker file exists and the source
# hasn't changed (based on file count). Scratch has a 90-day purge policy, so
# the marker disappears after purge and triggers a fresh sync.
#
# Usage:
#   source scripts/data/stage_data.sh                          # stage raw + cache (all datasets)
#   source scripts/data/stage_data.sh --cache                  # cache only (training jobs)
#   source scripts/data/stage_data.sh --raw                    # raw only (preprocessing jobs)
#   source scripts/data/stage_data.sh --dataset set_01         # only stage one dataset's cache
#   source scripts/data/stage_data.sh --skip-tmpdir            # read from scratch, skip TMPDIR copy
#   source scripts/data/stage_data.sh --cache --dataset set_01 --skip-tmpdir  # combine flags
#
# After sourcing, KD_GAT_DATA_ROOT and KD_GAT_CACHE_ROOT point to the fastest
# available copy. The calling SLURM script can use these env vars directly.

set -euo pipefail

# --- Config ---
LAKE_ROOT="${KD_GAT_LAKE_ROOT:-}"
DATA_ROOT="${KD_GAT_DATA_ROOT:-/users/PAS2022/rf15/kd-gat-data}"
SCRATCH="${KD_GAT_SCRATCH:-/fs/scratch/PAS1266}"
SCRATCH_DATA="${SCRATCH}/kd-gat-data"

# If lake root is set and has data, prefer it as source
if [[ -n "$LAKE_ROOT" && -d "${LAKE_ROOT}/raw" ]]; then
    DATA_ROOT="${LAKE_ROOT}"
    echo "Using ESS lake as data source: ${DATA_ROOT}"
fi

STAGE_RAW=true
STAGE_CACHE=true
DATASET=""
SKIP_TMPDIR=false

for arg in "$@"; do
    case "$arg" in
        --cache) STAGE_RAW=false ;;
        --raw)   STAGE_CACHE=false ;;
        --skip-tmpdir) SKIP_TMPDIR=true ;;
        --dataset=*) DATASET="${arg#--dataset=}" ;;
        --dataset) ;; # next arg handled below
        *) [[ "${prev_arg:-}" == "--dataset" ]] && DATASET="$arg" ;;
    esac
    prev_arg="$arg"
done

echo "=== Data staging ==="
echo "Source:  ${DATA_ROOT}"
echo "Scratch: ${SCRATCH_DATA}"
echo "TMPDIR:  ${TMPDIR:-<not set>}"
[[ -n "$DATASET" ]] && echo "Dataset: ${DATASET}" || echo "Dataset: all"
$SKIP_TMPDIR && echo "Mode:    scratch-only (skip TMPDIR)"

# --- Helper: check if scratch copy is fresh ---
_needs_sync() {
    local src_dir="$1"
    local dst_dir="$2"
    local marker="${dst_dir}/.staged_marker"

    if [[ ! -d "$dst_dir" ]] || [[ ! -f "$marker" ]]; then
        return 0
    fi

    local src_count dst_count
    src_count=$(find "$src_dir" -type f 2>/dev/null | wc -l)
    dst_count=$(cat "$marker" 2>/dev/null || echo "0")

    if [[ "$src_count" != "$dst_count" ]]; then
        echo "  File count changed (source=$src_count, staged=$dst_count) — re-syncing"
        return 0
    fi

    return 1
}

_write_marker() {
    local src_dir="$1"
    local dst_dir="$2"
    find "$src_dir" -type f 2>/dev/null | wc -l > "${dst_dir}/.staged_marker"
}

# --- Step 1: ESS → Scratch (rsync, incremental, skip if fresh) ---
if $STAGE_RAW && [[ -d "${DATA_ROOT}/raw" ]]; then
    if _needs_sync "${DATA_ROOT}/raw" "${SCRATCH_DATA}/raw"; then
        echo "Staging raw data to scratch..."
        mkdir -p "${SCRATCH_DATA}/raw"
        rsync -a --info=progress2 "${DATA_ROOT}/raw/" "${SCRATCH_DATA}/raw/"
        _write_marker "${DATA_ROOT}/raw" "${SCRATCH_DATA}/raw"
    else
        echo "Scratch raw data is fresh — skipping copy"
    fi
fi

if $STAGE_CACHE && [[ -d "${DATA_ROOT}/cache" ]]; then
    # Scope to single dataset if --dataset is set
    local_src="${DATA_ROOT}/cache"
    local_dst="${SCRATCH_DATA}/cache"
    if [[ -n "$DATASET" ]]; then
        local_src="${DATA_ROOT}/cache/${DATASET}"
        local_dst="${SCRATCH_DATA}/cache/${DATASET}"
    fi

    if [[ -d "$local_src" ]]; then
        if _needs_sync "$local_src" "$local_dst"; then
            echo "Staging cache to scratch ($([[ -n "$DATASET" ]] && echo "$DATASET" || echo "all"))..."
            mkdir -p "$local_dst"
            rsync -a --info=progress2 "${local_src}/" "${local_dst}/"
            _write_marker "$local_src" "$local_dst"
        else
            echo "Scratch cache is fresh — skipping copy"
        fi
    fi
fi

# --- Step 2: Scratch → TMPDIR (per-job local SSD) ---
if ! $SKIP_TMPDIR && [[ -n "${TMPDIR:-}" ]]; then
    TMPDIR_DATA="${TMPDIR}/kd-gat-data"
    mkdir -p "${TMPDIR_DATA}"

    if $STAGE_CACHE; then
        # Scope TMPDIR copy to single dataset if --dataset is set
        if [[ -n "$DATASET" ]]; then
            scratch_cache="${SCRATCH_DATA}/cache/${DATASET}"
            tmpdir_cache="${TMPDIR_DATA}/cache/${DATASET}"
        else
            scratch_cache="${SCRATCH_DATA}/cache"
            tmpdir_cache="${TMPDIR_DATA}/cache"
        fi

        if [[ -d "$scratch_cache" ]] && [[ ! -d "$tmpdir_cache" ]]; then
            echo "Staging cache to TMPDIR ($([[ -n "$DATASET" ]] && echo "$DATASET — $(du -sh "$scratch_cache" 2>/dev/null | cut -f1)" || echo "all"))..."
            mkdir -p "$(dirname "$tmpdir_cache")"
            cp -r "$scratch_cache" "$tmpdir_cache"
        else
            echo "TMPDIR cache already exists — skipping copy"
        fi
        export KD_GAT_CACHE_ROOT="${tmpdir_cache}"
        echo "KD_GAT_CACHE_ROOT=${KD_GAT_CACHE_ROOT}"
    fi

    if $STAGE_RAW && [[ -d "${SCRATCH_DATA}/raw" ]]; then
        if [[ ! -d "${TMPDIR_DATA}/raw" ]]; then
            echo "Staging raw data to TMPDIR..."
            cp -r "${SCRATCH_DATA}/raw" "${TMPDIR_DATA}/"
        else
            echo "TMPDIR raw data already exists — skipping copy"
        fi
        export KD_GAT_DATA_ROOT="${TMPDIR_DATA}"
        echo "KD_GAT_DATA_ROOT=${KD_GAT_DATA_ROOT}"
    fi
else
    # Skip TMPDIR: use scratch as fastest available tier
    if [[ -d "${SCRATCH_DATA}" ]]; then
        export KD_GAT_DATA_ROOT="${SCRATCH_DATA}"
        if [[ -n "$DATASET" ]]; then
            export KD_GAT_CACHE_ROOT="${SCRATCH_DATA}/cache/${DATASET}"
        else
            export KD_GAT_CACHE_ROOT="${SCRATCH_DATA}/cache"
        fi
    fi
fi

echo "=== Staging complete ==="
echo "KD_GAT_DATA_ROOT=${KD_GAT_DATA_ROOT}"
echo "KD_GAT_CACHE_ROOT=${KD_GAT_CACHE_ROOT:-<using default>}"
