# MLflow Migration: New Dataflow & Pipeline

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         TRAINING (SLURM GPU)         │
                    │                                      │
                    │  cli.py                              │
                    │    with mlflow.start_run():          │
                    │      log_params(dataset, model, ...) │
                    │      log_artifact(config.json)       │
                    │                                      │
                    │  trainer_factory.py                  │
                    │    mlflow.pytorch.autolog()          │
                    │      → per-epoch metrics (val_loss)  │
                    │      → checkpoint logging            │
                    │                                      │
                    │  cli.py (post-training)              │
                    │    log_metrics(duration, peak_gpu)   │
                    │    log_artifact(best_model.pt, ...)  │
                    │    set_tag(status=success/failed)    │
                    └──────────────┬────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────────┐
                    │  MLflow SQLite (data/mlflow/mlflow.db)│
                    │                                      │
                    │  Single store:                       │
                    │    • runs (params, tags, status)     │
                    │    • metrics (per-epoch values)      │
                    │    • artifacts (checkpoints, etc.)   │
                    └──────────────┬────────────────────────┘
                                  │
               ┌──────────────────┼──────────────────────┐
               ▼                  ▼                      ▼
    ┌──────────────────┐ ┌───────────────┐  ┌────────────────────┐
    │ SLURM Epilog     │ │ Ad-hoc query  │  │ Export pipeline     │
    │ _epilog.sh       │ │               │  │ export.py           │
    │   ↓              │ │ mlflow        │  │   reads filesystem  │
    │ push_experiments │ │   .search_    │  │   → reports/data/   │
    │ _to_hf.py       │ │   runs()      │  │   → Quarto paper    │
    │   ↓              │ │               │  │                     │
    │ HF Dataset       │ │ DuckDB       │  │ dev-server.html     │
    │ (experiments     │ │   ATTACH      │  │   YAML spec dev     │
    │  .parquet)       │ │   sqlite DB   │  │   sub-second reload │
    └────────┬─────────┘ └───────────────┘  └────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Streamlit Dashboard (HF Spaces)     │
    │  ~/kd-gat-dashboard/                 │
    │                                      │
    │  Experiments page:                   │
    │    reads buckeyeguy/kd-gat-experiments│
    │    Leaderboard, KD Transfer,         │
    │    Model Comparison, Raw Data        │
    │                                      │
    │  Sweeps page:                        │
    │    reads buckeyeguy/kd-gat-sweeps    │
    │    Overview, Parallel Coords,        │
    │    HP Sensitivity, Raw Trials        │
    └──────────────────────────────────────┘
```

## What Happens in a Single Training Run

1. `cli.py` resolves config, archives previous run if exists
2. `mlflow.start_run()` opens a tracked run with tags:
   - dataset, model_type, scale, stage
   - slurm_job_id, gpu_name, run_type, config_hash
   - teacher_run_id (for KD runs), sweep_id (for sweep trials)
3. `mlflow.log_params()` records key hyperparameters (dataset, model, scale, stage, has_kd, batch_size, max_epochs, lr)
4. `mlflow.log_artifact(config.json)` saves frozen config
5. `make_trainer()` calls `mlflow.pytorch.autolog()` — Lightning logs `val_loss`, `train_loss`, learning rate, etc. per epoch automatically
6. Training runs via `STAGE_FNS[stage](cfg)`
7. Post-training: `mlflow.log_metrics()` for duration_seconds, peak_gpu_mb, plus any numeric values from the result dict
8. `mlflow.log_artifact()` for each output file: `best_model.pt`, `embeddings.npz`, `attention_weights.npz`, `dqn_policy.json`, `metrics.json`, `explanations.npz`
9. `mlflow.set_tag("status", "success")` or `"failed"` with failure_reason
10. The `with` block ends, MLflow run closes automatically
11. SLURM epilog runs `push_experiments_to_hf.py` → `mlflow.search_runs()` → Parquet → HF Dataset

## Sweep-Specific Flow

```
tune_config.py → run_tune()
  → Ray Tune trials
    → subprocess mode: each trial runs cli.py → gets its own MLflow run
    → inprocess mode: each trial uses make_trainer() → mlflow.pytorch.autolog()
  → After tuner.fit():
    1. export_best_config() → YAML to data/sweep_results/
    2. MLflow sweep summary run (experiment: kd-gat-sweep-{stage})
       - logs best_val_loss, num_trials, num_errors
       - logs best config as params
       - logs best config YAML as artifact
    3. sweep_export.ingest_and_push()
       → parse trial dirs → data/datalake/sweeps.parquet → HF Dataset
    4. Dashboard Sweeps tab reads from HF Dataset
```

## How to Query Data

### Python (MLflow API)
```python
import mlflow
from graphids.config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# All runs
runs = mlflow.search_runs(search_all_experiments=True)

# Filter by experiment
runs = mlflow.search_runs(experiment_names=["kd-gat-autoencoder"])

# Filter by tags
runs = mlflow.search_runs(
    filter_string="tags.dataset = 'hcrl_sa' AND tags.stage = 'autoencoder'"
)

# Get a specific run's artifacts
run = mlflow.get_run("run_id_here")
mlflow.artifacts.download_artifacts(run_id="...", artifact_path="best_model.pt")
```

### DuckDB (SQL on SQLite)
```sql
-- DuckDB can read MLflow's SQLite DB directly
INSTALL sqlite;
LOAD sqlite;
ATTACH 'data/mlflow/mlflow.db' AS m (TYPE sqlite);

-- List all runs
SELECT * FROM m.runs LIMIT 10;

-- Metrics
SELECT * FROM m.metrics WHERE key = 'val_loss' ORDER BY timestamp;
```

### MLflow UI (OSC OnDemand)
```bash
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
# Opens web UI on port 5000
```

### Push to Dashboard
```bash
# Manual push (also auto-runs in SLURM epilog)
python scripts/data/push_experiments_to_hf.py
```

## Consolidated Dashboard

The Streamlit dashboard at `~/kd-gat-dashboard/` replaces both:
- The old Quarto `dashboard.qmd` (445 lines of Mosaic/OJS)
- The old standalone `~/kd-gat-sweep-dashboard/`

### Structure
- **Experiments page**: Leaderboard, KD Transfer comparison, Model Comparison (scatter: performance vs time, box: GPU memory), Raw Data with CSV export
- **Sweeps page**: Trial outcomes pie chart, best per sweep table, duration histogram, parallel coordinates, HP sensitivity scatter plots, raw trials with CSV export

### Data Sources
| HF Dataset | Producer | Contents |
|-----------|----------|----------|
| `buckeyeguy/kd-gat-experiments` | `push_experiments_to_hf.py` (SLURM epilog) | MLflow runs as Parquet |
| `buckeyeguy/kd-gat-sweeps` | `sweep_export.py` (after Ray Tune) | Trial-level sweep results |

### Deployment
```bash
# Upload to HF Space
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id='buckeyeguy/kd-gat-dashboard',
    folder_path='$HOME/kd-gat-dashboard',
    repo_type='space',
)
"
```

## Paper Figure Workflow

Figures develop independently from the paper prose:

```
reports/figures/*.yaml (Mosaic specs)
    │
    ├── dev-server.html + python3 -m http.server
    │     Edit YAML → refresh browser → sub-second feedback
    │     No Quarto rebuild needed
    │
    ├── Paper: quarto preview (hot-reload for prose changes)
    │     Specs referenced via {{< include >}} or OJS cells
    │
    └── Beyond PDF: export_tmlr.py extracts specs → standalone iframes
```

### Using the Dev Server
```bash
cd ~/KD-GAT/reports
python3 -m http.server 8765
# Open http://localhost:8765/dev-server.html
# Select a YAML spec from the dropdown, click Reload
# Edit the YAML, click Reload again — instant preview
```

## Key Differences from Old System

| Aspect | Before | After |
|--------|--------|-------|
| **Metrics store** | Custom Parquet files (`data/datalake/runs.parquet`) via `lakehouse.py` | MLflow SQLite (`data/mlflow/mlflow.db`) via `mlflow.start_run()` |
| **Per-epoch logging** | CSVLogger → flat CSV files in run dirs | `mlflow.pytorch.autolog()` → structured DB with queryable API |
| **Artifact catalog** | `artifacts.parquet` with manual SHA-256 hashing | MLflow artifact store (automatic, per-run) |
| **Sync code** | `_sync_lakehouse()` — 50-line fire-and-forget function with 20+ params | Built into MLflow context manager (`with mlflow.start_run()`) |
| **Dashboard data** | `export.py` copies Parquet → `reports/data/` → Quarto OJS/Mosaic | `push_experiments_to_hf.py` → HF Dataset → Streamlit |
| **Dashboard tech** | Quarto dashboard.qmd (445 lines, Mosaic vgplot, DuckDB-WASM, fragile) | Streamlit + Plotly on HF Spaces (Python, proven pattern) |
| **Sweep tracking** | `sweep_export.py` → Parquet → HF → separate Streamlit app | Same data flow, plus MLflow sweep summary run, consolidated into one app |
| **Experiment UI** | None (ad-hoc DuckDB queries or Quarto dashboard) | MLflow UI (local) + Streamlit dashboard (public) |
| **Dead weight removed** | — | W&B (42MB dir, 757 tracked files), `lakehouse.py` (302 lines), CSVLogger, `_sync_lakehouse()` (50 lines), `register_artifacts()` (57 lines) |
| **Config management** | Wandb config + lakehouse params + CSV logs (3 places) | MLflow params + tags + artifacts (1 place) |
| **Failure handling** | `_sync_lakehouse(..., success=False, failure_reason=...)` | `mlflow.set_tag("status", "failed")` + `mlflow.set_tag("failure_reason", ...)` |
| **CI** | `pip install wandb` | `pip install mlflow` |

## Migration Strategy

- **New runs** go to MLflow from day one
- **Existing runs** stay in `data/datalake/` Parquet and `experimentruns/` (not migrated — ~72 runs, not worth the effort)
- `export.py` continues reading from filesystem during transition
- Once enough new data accumulates in MLflow, the datalake Parquet files become read-only historical archives
- The `data/datalake/sweeps.parquet` path remains active (sweep_export.py still writes there for HF push)

## Files Changed

### Modified
| File | Change |
|------|--------|
| `pyproject.toml` | `wandb>=0.18` → `mlflow>=2.18` |
| `.env` | Added `MLFLOW_TRACKING_URI` |
| `graphids/config/paths.py` | Added `MLFLOW_TRACKING_URI` constant |
| `graphids/config/__init__.py` | Re-exported `MLFLOW_TRACKING_URI` |
| `graphids/pipeline/cli.py` | Added `_setup_mlflow()`, `mlflow.start_run()` context, artifact logging; deleted `_sync_lakehouse()` |
| `graphids/pipeline/stages/trainer_factory.py` | Replaced `CSVLogger` with `_setup_mlflow_autolog()` via `mlflow.pytorch.autolog()`; removed `logger=` from Trainer |
| `graphids/pipeline/orchestration/tune_config.py` | Added MLflow sweep summary run after `tuner.fit()` |
| `scripts/slurm/_preamble.sh` | Added `MLFLOW_TRACKING_URI` export |
| `scripts/slurm/_epilog.sh` | Added `push_experiments_to_hf.py` call |
| `scripts/profiling/run_pygod_baselines.py` | `--wandb` → `--mlflow` |
| `scripts/data/cleanup_orphans.sh` | Removed `.done` sentinel checks; uses `best_model.pt`/`metrics.json` |
| `.github/workflows/ci.yml` | `wandb` → `mlflow` in pip install |
| `.gitignore` | Added `wandb/`, `data/mlflow/`; removed `**/.done` |
| `CLAUDE.md` | Updated skill descriptions, export commands |
| `.claude/rules/experiment-tracking.md` | Full rewrite for MLflow |
| `.claude/rules/project-structure.md` | Updated tree (removed lakehouse.py, added mlflow/) |
| `.claude/rules/architecture.md` | Updated tracking references |
| `.claude/skills/check-status/SKILL.md` | W&B → MLflow |
| `.claude/skills/sync-state/SKILL.md` | W&B → MLflow |

### Deleted
| File | Lines | Reason |
|------|-------|--------|
| `graphids/pipeline/lakehouse.py` | 302 | Replaced by MLflow |
| `wandb/` directory | 757 git-tracked files, 42MB | Dead W&B data |

### Created
| File | Purpose |
|------|---------|
| `data/mlflow/` | MLflow SQLite backend directory |
| `scripts/data/push_experiments_to_hf.py` | MLflow → Parquet → HF Dataset push |
| `reports/dev-server.html` | Standalone YAML spec development shell |
| `~/kd-gat-dashboard/app.py` | Consolidated Streamlit dashboard |
| `~/kd-gat-dashboard/data_loader.py` | HF Dataset loaders with caching |
| `~/kd-gat-dashboard/Dockerfile` | Docker config for HF Spaces |
| `~/kd-gat-dashboard/requirements.txt` | Python dependencies |
| `~/kd-gat-dashboard/README.md` | HF Space metadata |
| `docs/mlflow-migration.md` | This document |

## Concurrency Notes

SQLite write locks are brief (milliseconds per INSERT). Ray Tune trials finish asynchronously over hours, not simultaneously. Risk is low at <500 runs.

**If SQLite locks become an issue:**
1. **First fallback:** `MLFLOW_TRACKING_URI=file:///path/to/mlruns` (flat file backend, same API, zero code changes)
2. **Second fallback:** Run `mlflow server` as sidecar in SLURM job (serializes writes, 3 lines in script)
