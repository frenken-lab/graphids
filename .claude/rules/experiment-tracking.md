---
paths:
  - "graphids/pipeline/cli.py"
  - "graphids/pipeline/stages/trainer_factory.py"
  - "data/mlflow/**"
---

# KD-GAT Experiment Tracking

## Architecture: MLflow SQLite Backend

Single store for metrics, params, tags, AND artifacts — all under one run ID.

```
Training (Lightning)
    → mlflow.pytorch.autolog() for per-epoch metrics
    → mlflow.log_artifact() for checkpoints, embeddings, configs
    → MLflow SQLite DB (data/mlflow/mlflow.db)

Consumers:
    ├── Streamlit dashboard → mlflow.search_runs() → DataFrame → Plotly
    ├── DuckDB CLI         → ATTACH 'mlflow.db' AS mlflow (TYPE sqlite)
    └── OSC OnDemand       → MLflow UI (mlflow ui --backend-store-uri ...)
```

## MLflow Integration Points

| Component | How it logs |
|-----------|-----------|
| `cli.py` | `mlflow.start_run()` context wraps dispatch; logs params, tags, post-training metrics, artifacts |
| `trainer_factory.py` | `mlflow.pytorch.autolog()` + DeviceStatsMonitor for per-epoch metrics + GPU stats |
| `tune_config.py` | Sweep summary run with best config, val_loss, trial counts |
| `run_pygod_baselines.py` | Optional `--mlflow` flag for baseline metrics |

## Key Environment

- `MLFLOW_TRACKING_URI` — set in `.env` and `_preamble.sh` (default: `sqlite:///data/mlflow/mlflow.db`)
- `data/mlflow/` — gitignored, contains SQLite DB and artifact store

## Artifacts

Binary artifacts stored in `experimentruns/{dataset}/{run}/` AND logged to MLflow:
- `best_model.pt` — model checkpoint
- `embeddings.npz` — VGAE z-mean + GAT hidden layers
- `attention_weights.npz` — GAT attention head weights
- `dqn_policy.json` — DQN alpha values by class
- `config.json` — frozen PipelineConfig
- `metrics.json` — training metrics summary

## HF Dataset Push

`scripts/data/push_experiments_to_hf.py` reads MLflow → Parquet → HF Dataset (`buckeyeguy/kd-gat-experiments`). Auto-triggered by `_epilog.sh` after SLURM jobs.
