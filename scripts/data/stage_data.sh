#!/usr/bin/env bash
# Stage data from permanent storage to scratch / $TMPDIR for fast job I/O.
#
# Data flow:  KD_GAT_DATA_ROOT (home/project NFS)
#         →   KD_GAT_SCRATCH/kd-gat-data/ (GPFS scratch, persistent across jobs)
#         →   $TMPDIR/kd-gat-data/ (local SSD, per-job, fastest)
#
# Smart caching: skips NFS→scratch copy if a marker file exists and the source
# hasn't changed (based on file count). Scratch has a 90-day purge policy, so
# the marker disappears after purge and triggers a fresh copy.
#
# Usage:
#   source scripts/data/stage_data.sh           # default: stage raw + cache
#   source scripts/data/stage_data.sh --cache   # cache only (for training jobs)
#   source scripts/data/stage_data.sh --raw     # raw only (for preprocessing jobs)
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

for arg in "$@"; do
    case "$arg" in
        --cache) STAGE_RAW=false ;;
        --raw)   STAGE_CACHE=false ;;
    esac
done

echo "=== Data staging ==="
echo "Source:  ${DATA_ROOT}"
echo "Scratch: ${SCRATCH_DATA}"
echo "TMPDIR:  ${TMPDIR:-<not set>}"

# --- Helper: check if scratch copy is fresh ---
# Uses a marker file that stores the source file count. If the marker exists
# and the count matches, skip the copy. Scratch purge (90 days) deletes the
# marker, triggering a fresh sync on next job.
_needs_sync() {
    local src_dir="$1"
    local dst_dir="$2"
    local marker="${dst_dir}/.staged_marker"

    # No destination or no marker → needs sync
    if [[ ! -d "$dst_dir" ]] || [[ ! -f "$marker" ]]; then
        return 0  # true, needs sync
    fi

    # Compare file count as a lightweight staleness check
    local src_count dst_count
    src_count=$(find "$src_dir" -type f 2>/dev/null | wc -l)
    dst_count=$(cat "$marker" 2>/dev/null || echo "0")

    if [[ "$src_count" != "$dst_count" ]]; then
        echo "  File count changed (source=$src_count, staged=$dst_count) — re-syncing"
        return 0  # true, needs sync
    fi

    return 1  # false, skip sync
}

_write_marker() {
    local src_dir="$1"
    local dst_dir="$2"
    find "$src_dir" -type f 2>/dev/null | wc -l > "${dst_dir}/.staged_marker"
}

# --- Step 1: NFS → Scratch (rsync, incremental, skip if fresh) ---
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
    if _needs_sync "${DATA_ROOT}/cache" "${SCRATCH_DATA}/cache"; then
        echo "Staging cache to scratch..."
        mkdir -p "${SCRATCH_DATA}/cache"
        rsync -a --info=progress2 "${DATA_ROOT}/cache/" "${SCRATCH_DATA}/cache/"
        _write_marker "${DATA_ROOT}/cache" "${SCRATCH_DATA}/cache"
    else
        echo "Scratch cache is fresh — skipping copy"
    fi
fi

# --- Step 2: Scratch → $TMPDIR (cp, per-job local SSD) ---
if [[ -n "${TMPDIR:-}" ]]; then
    TMPDIR_DATA="${TMPDIR}/kd-gat-data"
    mkdir -p "${TMPDIR_DATA}"

    if $STAGE_CACHE && [[ -d "${SCRATCH_DATA}/cache" ]]; then
        if [[ ! -d "${TMPDIR_DATA}/cache" ]]; then
            echo "Staging cache to TMPDIR..."
            cp -r "${SCRATCH_DATA}/cache" "${TMPDIR_DATA}/"
        else
            echo "TMPDIR cache already exists — skipping copy"
        fi
        export KD_GAT_CACHE_ROOT="${TMPDIR_DATA}/cache"
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
    # No TMPDIR (login node or non-SLURM) — use scratch as fastest tier
    if [[ -d "${SCRATCH_DATA}" ]]; then
        export KD_GAT_DATA_ROOT="${SCRATCH_DATA}"
        export KD_GAT_CACHE_ROOT="${SCRATCH_DATA}/cache"
    fi
fi

echo "=== Staging complete ==="
echo "KD_GAT_DATA_ROOT=${KD_GAT_DATA_ROOT}"
echo "KD_GAT_CACHE_ROOT=${KD_GAT_CACHE_ROOT:-<using default>}"
