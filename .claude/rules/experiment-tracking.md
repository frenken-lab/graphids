---
paths:
  - "graphids/pipeline/lakehouse.py"
  - "graphids/pipeline/export.py"
  - "graphids/pipeline/cli.py"
  - "data/datalake/**"
---

# KD-GAT Experiment Tracking

## Architecture: Two-Tier (Metadata Catalog + Artifact Store)

Design decision: custom Parquet datalake, not MLflow. See `~/plans/experiment-tracking-design-decision.md`. W&B integration removed (was write-only dead weight).

**Metadata catalog** (`data/datalake/`): All structured, queryable data lives here as Parquet files.
**Artifact store** (`experimentruns/`): Binary blobs (checkpoints, embeddings, attention weights) live here, referenced by URI in the catalog.

## Datalake (Primary)

Parquet-based structured storage in `data/datalake/`:

| File | Contents |
|------|----------|
| `runs.parquet` | Run metadata (dataset, model, scale, stage, KD, success, timestamps, lineage) |
| `metrics.parquet` | Per-run per-model core metrics (F1, accuracy, AUC, etc.) |
| `configs.parquet` | Key hyperparameters + full frozen config JSON |
| `datasets.parquet` | Dataset catalog with cache stats |
| `artifacts.parquet` | Artifact catalog: run_id, type, URI, size, content_hash, producer, timestamp |
| `sweeps.parquet` | Ray Tune HPO trial results |
| `training_curves/{run_id}.parquet` | Per-epoch metrics from Lightning CSV logs |
| `queries/leaderboard.sql` | Best metrics per config (ad-hoc: `duckdb < queries/leaderboard.sql`) |
| `queries/kd_impact.sql` | KD transfer analysis |

**Write path**: `graphids/pipeline/lakehouse.py` appends to Parquet on run completion. `register_artifacts()` scans run directories and catalogs files with content hashes.

**Read path**: `graphids/pipeline/export.py` reads datalake Parquet for report generation. DuckDB CLI for ad-hoc queries: `duckdb -c "SELECT * FROM 'data/datalake/runs.parquet'"`.

**Lineage**: `runs.parquet.input_checkpoint_uri` tracks which teacher checkpoint a KD run consumed. `artifacts.parquet.produced_by_run_id` tracks which run created each artifact.

## Artifacts

Binary artifacts stored in `experimentruns/{dataset}/{run}/`:
- `best_model.pt` — model checkpoint (loaded for inference, evaluation, next-stage training)
- `embeddings.npz` — VGAE z-mean + GAT hidden layers
- `attention_weights.npz` — GAT attention head weights
- `explanations.npz` — GNNExplainer feature importance (when `run_explainer=True`)
- `dqn_policy.json` — DQN alpha values by class
- `cka_matrix.json` — CKA similarity (teacher vs student)

All artifacts are indexed in `artifacts.parquet` with content hashes for deduplication tracking.

## Report Export

`python -m graphids.pipeline.export` reads datalake Parquet → static JSON/Parquet in `reports/data/` (leaderboard, runs, metrics, training curves, datasets, KD transfer, model sizes). ~2s, login node safe. Quarto site auto-deploys via GitHub Actions on push to main.

## Sweep Tracking

Ray Tune results → `sweep_export.ingest_and_push()` → `data/datalake/sweeps.parquet` → HF Dataset (`buckeyeguy/kd-gat-sweeps`) → public Streamlit dashboard.
