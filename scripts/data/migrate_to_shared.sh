#!/usr/bin/env bash
# Migrate KD-GAT data to shared PAS1266 project storage.
#
# Submit: sbatch --account=PAS1266 --partition=serial scripts/data/migrate_to_shared.sh
# Or run interactively on a compute node (salloc).
#
#SBATCH --job-name=kd-gat-migrate
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/migrate_%j.out
#SBATCH --error=slurm_logs/migrate_%j.err

set -euo pipefail

SRC="$HOME/KD-GAT"
SHARED="${KD_GAT_SHARED_ROOT:?Set KD_GAT_SHARED_ROOT in .env before running}"

echo "=== KD-GAT Data Migration ==="
echo "Source: $SRC"
echo "Target: $SHARED"
echo "Started: $(date)"
echo ""

# Ensure target dirs exist (should already from setup, but be safe)
mkdir -p "$SHARED/data/raw" "$SHARED/data/cache" \
         "$SHARED/data/datalake/artifacts" \
         "$SHARED/data/datalake/training_curves" \
         "$SHARED/data/datalake/loss_landscapes" \
         "$SHARED/experimentruns"

# 1. Raw CAN data (automotive/ → raw/ to match paths.py convention)
echo "--- Syncing raw data (automotive → raw) ---"
rsync -av --progress "$SRC/data/automotive/" "$SHARED/data/raw/"

# 2. Graph cache (preprocessed .pt files)
echo "--- Syncing graph cache ---"
rsync -av --progress "$SRC/data/cache/" "$SHARED/data/cache/"

# 3. Datalake (Parquet — small but critical for undergrad queries)
echo "--- Syncing datalake ---"
rsync -av --progress "$SRC/data/datalake/" "$SHARED/data/datalake/"

# 4. Experiment runs (checkpoints + configs + metrics)
echo "--- Syncing experiment runs ---"
rsync -av --progress "$SRC/experimentruns/" "$SHARED/experimentruns/"

# 5. Fix group ownership and permissions (setgid propagates to new files)
echo "--- Fixing permissions ---"
chgrp -R PAS1266 "$SHARED" 2>/dev/null || true
find "$SHARED" -type d -exec chmod 2770 {} + 2>/dev/null || true
find "$SHARED" -type f -exec chmod g+r {} + 2>/dev/null || true

# 6. Summary
echo ""
echo "=== Migration Complete ==="
echo "Finished: $(date)"
echo ""
echo "File counts:"
echo "  Raw CSVs:     $(find "$SHARED/data/raw" -name '*.csv' 2>/dev/null | wc -l)"
echo "  Cache files:  $(find "$SHARED/data/cache" -name '*.pt' -o -name '*.pkl' 2>/dev/null | wc -l)"
echo "  Parquet:      $(find "$SHARED/data/datalake" -name '*.parquet' 2>/dev/null | wc -l)"
echo "  Checkpoints:  $(find "$SHARED/experimentruns" -name 'best_model.pt' 2>/dev/null | wc -l)"
echo ""
echo "Disk usage:"
du -sh "$SHARED/data/raw" "$SHARED/data/cache" "$SHARED/data/datalake" "$SHARED/experimentruns" "$SHARED"
echo ""
echo "Next steps:"
echo "  1. source .env  (picks up KD_GAT_SHARED_ROOT)"
echo "  2. python -m graphids.pipeline.build_analytics  (rebuild DuckDB)"
echo "  3. Verify: python -c \"import duckdb; print(duckdb.connect('$SHARED/data/datalake/analytics.duckdb').execute('SELECT COUNT(*) FROM runs').fetchone())\""
