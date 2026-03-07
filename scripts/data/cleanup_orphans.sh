#!/usr/bin/env bash
# Scan experimentruns/ for orphaned/failed run directories.
#
# An "orphan" is a run directory that has no best_model.pt or metrics.json,
# indicating the stage never completed successfully.
#
# Modes:
#   --dry-run  (default)  List orphaned dirs
#   --archive             Move orphaned dirs to archive/
#   --delete              Permanently remove orphaned dirs (destructive!)
#
# Usage:
#   bash scripts/data/cleanup_orphans.sh                # dry-run
#   bash scripts/data/cleanup_orphans.sh --archive      # archive orphans
#   bash scripts/data/cleanup_orphans.sh --delete       # delete orphans

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_ROOT="${PROJECT_ROOT}/experimentruns"
ARCHIVE_DIR="${EXPERIMENT_ROOT}/archive"

MODE="dry-run"
if [[ "${1:-}" == "--archive" ]]; then
    MODE="archive"
elif [[ "${1:-}" == "--delete" ]]; then
    MODE="delete"
fi

echo "=== Cleanup Orphaned Runs ==="
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Mode:            ${MODE}"
echo ""

# --- Find orphaned run directories (no best_model.pt or metrics.json) ---
orphans=()
if [[ -d "$EXPERIMENT_ROOT" ]]; then
    for dataset_dir in "$EXPERIMENT_ROOT"/*/; do
        [[ -d "$dataset_dir" ]] || continue
        dataset_name="$(basename "$dataset_dir")"
        [[ "$dataset_name" == "archive" || "$dataset_name" == "baselines" ]] && continue

        for run_dir in "$dataset_dir"*/; do
            [[ -d "$run_dir" ]] || continue
            [[ "$(basename "$run_dir")" == *.archive_* ]] && continue
            # A completed run has at least a checkpoint or metrics file
            if [[ ! -f "${run_dir}best_model.pt" && ! -f "${run_dir}metrics.json" ]]; then
                orphans+=("$run_dir")
            fi
        done
    done
fi

echo "Orphaned directories: ${#orphans[@]}"
for d in "${orphans[@]:-}"; do
    [[ -z "$d" ]] && continue
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  ${d} (${size})"
done
echo ""

# --- Take action ---
if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry-run complete. Use --archive or --delete to take action."
    exit 0
fi

if [[ ${#orphans[@]} -eq 0 ]]; then
    echo "No orphans to process."
    exit 0
fi

if [[ "$MODE" == "archive" ]]; then
    mkdir -p "$ARCHIVE_DIR"
    for d in "${orphans[@]}"; do
        dest="${ARCHIVE_DIR}/$(basename "$(dirname "$d")")_$(basename "$d")"
        echo "Archiving: $d → $dest"
        mv "$d" "$dest"
    done
    echo "Archived ${#orphans[@]} directories to ${ARCHIVE_DIR}"
elif [[ "$MODE" == "delete" ]]; then
    for d in "${orphans[@]}"; do
        echo "Deleting: $d"
        rm -r "$d"
    done
    echo "Deleted ${#orphans[@]} orphaned directories."
fi
