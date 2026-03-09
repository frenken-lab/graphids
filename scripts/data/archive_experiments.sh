#!/usr/bin/env bash
# scripts/data/archive_experiments.sh — One-time archive of pre-March 2026 experiment data.
#
# Archives metadata (configs, metrics, MLflow snapshot) to scratch space.
# Does NOT archive model weights (architecture changed, weights are useless).
#
# Usage:
#   bash scripts/data/archive_experiments.sh          # archive only
#   bash scripts/data/archive_experiments.sh --clean   # archive + delete old data
#
# After verifying the first successful new pipeline run, re-run with --clean.

set -euo pipefail

PROJECT_ROOT="/users/PAS2022/rf15/KD-GAT"
ARCHIVE_ROOT="/fs/scratch/PAS1266/kd-gat-archive/pre_march_2026"
CLEAN=false

if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
fi

cd "$PROJECT_ROOT"

echo "=== KD-GAT Experiment Archive ==="
echo "Archive target: $ARCHIVE_ROOT"
echo "Clean mode: $CLEAN"
echo ""

# --- Create archive directory ---
mkdir -p "$ARCHIVE_ROOT"

# --- 1. Metadata snapshot ---
echo "Writing metadata.json..."
cat > "$ARCHIVE_ROOT/metadata.json" <<EOF
{
    "archive_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "reason": "Pre-sweep infrastructure cleanup. Code has changed too much to compare with these runs.",
    "contents": [
        "configs_and_metrics.tar.gz — config.json + metrics.json from every run",
        "mlflow_export.parquet — mlflow.search_runs() snapshot",
        "sweep_state.tar.gz — data/sweep_results/ + data/sweep_state/",
        "slurm_logs.tar.gz — compressed SLURM logs"
    ]
}
EOF

# --- 2. Archive configs and metrics (tiny) ---
if [[ -d experimentruns ]]; then
    echo "Archiving configs + metrics from experimentruns/..."
    find experimentruns -name "config.json" -o -name "metrics.json" \
        | tar czf "$ARCHIVE_ROOT/configs_and_metrics.tar.gz" -T -
    echo "  → configs_and_metrics.tar.gz ($(du -sh "$ARCHIVE_ROOT/configs_and_metrics.tar.gz" | cut -f1))"
else
    echo "  experimentruns/ not found — skipping"
fi

# --- 3. MLflow export ---
if [[ -f data/mlflow/mlflow.db ]]; then
    echo "Exporting MLflow runs to parquet..."
    python3 -c "
import mlflow, os
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///$PROJECT_ROOT/data/mlflow/mlflow.db'))
runs = mlflow.search_runs(search_all_experiments=True)
if not runs.empty:
    runs.to_parquet('$ARCHIVE_ROOT/mlflow_export.parquet', index=False)
    print(f'  → mlflow_export.parquet ({len(runs)} runs)')
else:
    print('  No MLflow runs found')
" 2>&1 || echo "  (MLflow export failed — non-fatal)"
else
    echo "  data/mlflow/mlflow.db not found — skipping"
fi

# --- 4. Sweep state ---
if [[ -d data/sweep_results ]] || [[ -d data/sweep_state ]]; then
    echo "Archiving sweep state..."
    tar czf "$ARCHIVE_ROOT/sweep_state.tar.gz" \
        data/sweep_results/ data/sweep_state/ 2>/dev/null || true
    echo "  → sweep_state.tar.gz"
else
    echo "  No sweep state found — skipping"
fi

# --- 5. SLURM logs ---
if [[ -d slurm_logs ]] && ls slurm_logs/*.{out,err} &>/dev/null; then
    echo "Archiving SLURM logs..."
    tar czf "$ARCHIVE_ROOT/slurm_logs.tar.gz" slurm_logs/ 2>/dev/null || true
    echo "  → slurm_logs.tar.gz"
else
    echo "  No SLURM logs found — skipping"
fi

echo ""
echo "Archive complete: $ARCHIVE_ROOT"
du -sh "$ARCHIVE_ROOT"

# --- 6. Clean old data (only with --clean) ---
if [[ "$CLEAN" == "true" ]]; then
    echo ""
    echo "=== Cleaning old data ==="

    if [[ -f data/mlflow/mlflow.db ]]; then
        echo "Removing MLflow DB (will be recreated on next run)..."
        rm -f data/mlflow/mlflow.db
    fi

    for dir in experimentruns ray_results data/sweep_results data/sweep_state data/datalake; do
        if [[ -d "$dir" ]]; then
            echo "Removing $dir/..."
            rm -rf "$dir"
        fi
    done

    if [[ -d slurm_logs ]]; then
        echo "Removing old SLURM logs..."
        find slurm_logs -name "*.out" -o -name "*.err" -delete 2>/dev/null || true
    fi

    echo "Clean complete."
fi

echo ""
echo "=== Done ==="
