#!/usr/bin/env bash
# scripts/lake/migrate_to_ess.sh — Migrate data from home dir to ESS data lake.
#
# Syncs raw datasets, preprocessed cache, and experiment runs to ESS.
# Experiment runs are restructured to add seed_42/ subdirectories.
# Generates _manifest.json for migrated runs.
#
# Usage:
#   bash scripts/lake/migrate_to_ess.sh --dry-run    # Preview only
#   bash scripts/lake/migrate_to_ess.sh              # Execute migration
#   bash scripts/lake/migrate_to_ess.sh --raw        # Raw datasets only
#   bash scripts/lake/migrate_to_ess.sh --cache      # Cache only
#   bash scripts/lake/migrate_to_ess.sh --runs       # Experiment runs only

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
LAKE_ROOT="${KD_GAT_LAKE_ROOT:-/fs/ess/PAS1266/kd-gat}"
DRY_RUN=false
MIGRATE_RAW=true
MIGRATE_CACHE=true
MIGRATE_RUNS=true

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --raw)     MIGRATE_CACHE=false; MIGRATE_RUNS=false ;;
        --cache)   MIGRATE_RAW=false; MIGRATE_RUNS=false ;;
        --runs)    MIGRATE_RAW=false; MIGRATE_CACHE=false ;;
        --help|-h)
            echo "Usage: bash scripts/lake/migrate_to_ess.sh [--dry-run] [--raw|--cache|--runs]"
            exit 0
            ;;
    esac
done

RSYNC_OPTS="-av --info=progress2"
if $DRY_RUN; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run"
fi

echo "=== ESS Data Lake Migration ==="
echo "Source:  ${PROJECT_ROOT}"
echo "Lake:   ${LAKE_ROOT}"
echo "Dry run: ${DRY_RUN}"
echo ""

# --- Step 1: Raw datasets ---
if $MIGRATE_RAW; then
    echo "--- Migrating raw datasets ---"
    SRC="${PROJECT_ROOT}/data/automotive"
    DST="${LAKE_ROOT}/raw"

    if [[ -d "$SRC" ]]; then
        for dataset_dir in "$SRC"/*/; do
            dataset_name=$(basename "$dataset_dir")
            echo "  ${dataset_name}..."
            rsync $RSYNC_OPTS "$dataset_dir" "${DST}/${dataset_name}/"
        done
    else
        echo "  Source not found: ${SRC} — skipping"
    fi
    echo ""
fi

# --- Step 2: Preprocessed cache ---
if $MIGRATE_CACHE; then
    echo "--- Migrating preprocessed cache ---"
    SRC="${PROJECT_ROOT}/data/cache"
    # Read current preprocessing version from Python
    PREP_VERSION=$(python -c "from graphids.config.constants import PREPROCESSING_VERSION; print(PREPROCESSING_VERSION)" 2>/dev/null || echo "3.0.0")
    DST="${LAKE_ROOT}/cache/v${PREP_VERSION}"

    if [[ -d "$SRC" ]]; then
        if ! $DRY_RUN; then
            mkdir -p "$DST"
        fi
        for dataset_dir in "$SRC"/*/; do
            if [[ ! -d "$dataset_dir" ]]; then continue; fi
            dataset_name=$(basename "$dataset_dir")
            echo "  ${dataset_name} → v${PREP_VERSION}/${dataset_name}..."
            if ! $DRY_RUN; then
                mkdir -p "${DST}/${dataset_name}"
            fi
            rsync $RSYNC_OPTS "$dataset_dir" "${DST}/${dataset_name}/"
        done
    else
        echo "  Source not found: ${SRC} — skipping"
    fi
    echo ""
fi

# --- Step 3: Experiment runs (restructure with seed subdirectory) ---
if $MIGRATE_RUNS; then
    echo "--- Migrating experiment runs ---"
    SRC="${PROJECT_ROOT}/experimentruns"
    DST="${LAKE_ROOT}/production"

    if [[ -d "$SRC" ]]; then
        # Find all run directories (contain config.json)
        find "$SRC" -name "config.json" -maxdepth 3 | while read -r config_file; do
            run_dir=$(dirname "$config_file")
            # Skip if already has seed_ subdirectory structure
            if [[ "$(basename "$run_dir")" == seed_* ]]; then
                continue
            fi

            # Derive relative path: dataset/model_scale_stage[_aux]
            rel_path=$(realpath --relative-to="$SRC" "$run_dir")

            # Read seed from config.json (default 42)
            seed=$(python -c "import json; print(json.load(open('${config_file}')).get('seed', 42))" 2>/dev/null || echo "42")

            dst_dir="${DST}/${rel_path}/seed_${seed}"
            echo "  ${rel_path} → ${rel_path}/seed_${seed}"

            if ! $DRY_RUN; then
                mkdir -p "$dst_dir"
                rsync -a "$run_dir/" "$dst_dir/"
            fi
        done

        # Generate manifests for migrated runs
        if ! $DRY_RUN; then
            echo ""
            echo "--- Generating manifests for migrated runs ---"
            cd "$PROJECT_ROOT"
            python -c "
from pathlib import Path
from graphids.storage.manifest import write_manifest, read_manifest
import json

production = Path('${DST}')
for manifest_candidate in production.rglob('config.json'):
    run_dir = manifest_candidate.parent
    # Only process seed_ directories
    if not run_dir.name.startswith('seed_'):
        continue
    # Skip if manifest already exists
    if (run_dir / '_manifest.json').exists():
        continue
    try:
        cfg = json.loads(manifest_candidate.read_text())
        seed = int(run_dir.name.split('_')[1])
        # Derive identity from path
        parts = run_dir.parent.name.split('_')
        dataset = run_dir.parent.parent.name
        stage_parts = run_dir.parent.name
        model_type = cfg.get('model_type', parts[0])
        scale = cfg.get('scale', parts[1] if len(parts) > 1 else 'large')
        # Extract stage (third part after model_scale_)
        remaining = '_'.join(parts[2:])
        # Check for auxiliary suffix
        aux = 'none'
        for known_aux in ['kd_standard', 'kd']:
            if remaining.endswith('_' + known_aux):
                aux = known_aux
                remaining = remaining[:-(len(known_aux) + 1)]
                break
        stage = remaining

        write_manifest(run_dir, dataset, model_type, scale, stage, aux, seed)
    except Exception as e:
        print(f'  Warning: {run_dir}: {e}')
"
        fi
    else
        echo "  Source not found: ${SRC} — skipping"
    fi
    echo ""
fi

echo "=== Migration complete ==="

# --- Verification ---
if ! $DRY_RUN; then
    echo ""
    echo "--- Verification ---"
    if $MIGRATE_RAW; then
        src_count=$(find "${PROJECT_ROOT}/data/automotive" -name "*.csv" 2>/dev/null | wc -l)
        dst_count=$(find "${LAKE_ROOT}/raw" -name "*.csv" 2>/dev/null | wc -l)
        echo "Raw CSVs: source=${src_count}, lake=${dst_count}"
    fi
    if $MIGRATE_RUNS; then
        src_count=$(find "${PROJECT_ROOT}/experimentruns" -name "config.json" -maxdepth 3 2>/dev/null | wc -l)
        dst_count=$(find "${LAKE_ROOT}/production" -name "_manifest.json" 2>/dev/null | wc -l)
        echo "Runs: source=${src_count} configs, lake=${dst_count} manifests"
    fi
fi
