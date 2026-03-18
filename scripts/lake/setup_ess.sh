#!/usr/bin/env bash
# scripts/lake/setup_ess.sh — Create the ESS data lake directory tree.
#
# Prerequisites:
#   - User must be in group PAS1266
#   - /fs/ess/PAS1266/ must be writable by user (group::rwx ACL)
#
# Usage:
#   bash scripts/lake/setup_ess.sh              # Create tree
#   bash scripts/lake/setup_ess.sh --dry-run    # Preview only

set -euo pipefail

LAKE_ROOT="/fs/ess/PAS1266/kd-gat"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: bash scripts/lake/setup_ess.sh [--dry-run]"
            echo "Creates the ESS data lake directory tree at ${LAKE_ROOT}"
            exit 0
            ;;
    esac
done

_mkdir() {
    if $DRY_RUN; then
        echo "[DRY RUN] mkdir -p $1"
    else
        mkdir -p "$1"
        echo "Created: $1"
    fi
}

echo "=== ESS Data Lake Setup ==="
echo "Lake root: ${LAKE_ROOT}"
echo ""

# --- Top-level directories ---
_mkdir "${LAKE_ROOT}"
_mkdir "${LAKE_ROOT}/raw"
_mkdir "${LAKE_ROOT}/cache"
_mkdir "${LAKE_ROOT}/production"
_mkdir "${LAKE_ROOT}/dev"
_mkdir "${LAKE_ROOT}/exports"
_mkdir "${LAKE_ROOT}/exports/paper"
_mkdir "${LAKE_ROOT}/exports/paper/metadata"
_mkdir "${LAKE_ROOT}/exports/paper/csv"
_mkdir "${LAKE_ROOT}/exports/paper/figures"
_mkdir "${LAKE_ROOT}/sweeps"
_mkdir "${LAKE_ROOT}/catalog"
_mkdir "${LAKE_ROOT}/mlflow"

# --- Dev sandbox for current user ---
_mkdir "${LAKE_ROOT}/dev/${USER}"

# --- Write layout.json ---
if ! $DRY_RUN; then
    cat > "${LAKE_ROOT}/layout.json" << 'LAYOUT_JSON'
{
    "path_version": 1,
    "created": "2026-03-16",
    "owner": "rf15",
    "group": "PAS1266",
    "description": "KD-GAT shared data lake on ESS (GPFS)",
    "path_convention": "{dataset}/{model}_{scale}_{stage}[_{aux}]/seed_{N}/",
    "directories": {
        "raw": "Immutable raw CAN bus datasets",
        "cache": "Preprocessed graph cache, versioned by PREPROCESSING_VERSION",
        "production": "Production experiment runs (Dagster orchestration)",
        "dev": "Per-user development sandboxes (same structure as production)",
        "exports": "Derived datasets for dashboards (parquet files)",
        "sweeps": "HPO sweep best-config YAMLs and searcher state",
        "catalog": "DuckDB index (disposable, rebuilt from files)",
        "mlflow": "MLflow tracking backend (SQLite on GPFS)"
    }
}
LAYOUT_JSON
    echo "Wrote: ${LAKE_ROOT}/layout.json"

    # --- Write LAYOUT.md ---
    cat > "${LAKE_ROOT}/LAYOUT.md" << 'LAYOUT_MD'
# KD-GAT Shared Data Lake

## Quick Start (read-only, no repo needed)

```python
import duckdb
db = duckdb.connect("/fs/ess/PAS1266/kd-gat/catalog/kd_gat.duckdb", read_only=True)
db.sql("SELECT dataset, model_type, scale, f1_macro FROM experiments ORDER BY f1_macro DESC")
```

## Directory Structure

```
/fs/ess/PAS1266/kd-gat/
  layout.json           # Machine-readable layout metadata
  LAYOUT.md             # This file
  raw/                  # Immutable raw datasets (hcrl_sa, hcrl_ch, set_01..04)
  cache/                # Preprocessed graphs, versioned (v3.0.0/)
    v3.0.0/{dataset}/   #   processed_graphs.pt, id_mapping.pkl, cache_metadata.json
  production/           # Production runs (Dagster-managed)
    {dataset}/{model}_{scale}_{stage}[_{aux}]/seed_{N}/
      config.json       #   Frozen PipelineConfig
      metrics.json      #   Evaluation metrics
      best_model.pt     #   Model checkpoint
      _manifest.json    #   Artifact inventory + checksums
  dev/{username}/       # Development sandboxes (same structure)
  sweeps/               # HPO sweep results
    {dataset}/          #   Per-dataset sweep outputs
      {stage}_{scale}_best.yaml    # Best config from Ray Tune
      {stage}_{scale}_searcher.pkl # Optuna searcher state (warm-start)
  exports/              # Dashboard datasets
    experiments.parquet  #   Flattened experiment results
    sweeps.parquet       #   Sweep results
    paper/               #   Paper-ready exports (CSVs + figure JSONs)
      _manifest.json     #     Artifact inventory + SHA-256 checksums
      metadata/          #     Dataset metadata (attack_type mappings)
      csv/               #     Result table CSVs (6 files)
      figures/           #     Interactive figure data.json files (5 files)
  catalog/              # DuckDB index (rebuilt from files)
    kd_gat.duckdb
  mlflow/               # MLflow SQLite backend
    mlflow.db
```

## Path Convention

Path encodes identity only: `{dataset}/{model}_{scale}_{stage}[_{aux}]/seed_{N}/`

New parameters, metrics, or config fields go in JSON files — paths don't change.

## Access

- **Read**: All PAS1266 members via default ACL
- **Write**: Currently Robert (rf15) only. Contact advisor for group write access.
- **Query**: `import duckdb` from any OSC Jupyter notebook — no repo clone needed.

## Rebuilding the Catalog

```bash
cd ~/KD-GAT
python -m graphids.lake rebuild-catalog
```
LAYOUT_MD
    echo "Wrote: ${LAKE_ROOT}/LAYOUT.md"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set KD_GAT_LAKE_ROOT in .env:"
echo "     export KD_GAT_LAKE_ROOT=\"${LAKE_ROOT}\""
echo "  2. Run migration:"
echo "     bash scripts/lake/migrate_to_ess.sh --dry-run"
