# KD-GAT Experiment Datalake

Shared experiment results for the KD-GAT CAN bus intrusion detection project.
All PAS1266 members have read access.

## Quick Start (DuckDB CLI)

```bash
# Use the bundled DuckDB CLI (no Python env needed)
SHARED="/fs/scratch/PAS1266/kd-gat-shared"
$SHARED/tools/duckdb < $SHARED/data/datalake/queries/leaderboard.sql
```

## Example Queries

```sql
-- Leaderboard: best F1 per model x dataset
SELECT * FROM v_leaderboard ORDER BY best_f1 DESC;

-- KD impact: does knowledge distillation help?
SELECT * FROM v_kd_impact ORDER BY f1_delta DESC;

-- All runs with status
SELECT dataset, model_type, scale, has_kd, stage, success
FROM runs ORDER BY started_at DESC;

-- Training curves for a specific run
SELECT * FROM read_parquet('training_curves/hcrl_sa_gat_large_curriculum.parquet');

-- Count runs per dataset
SELECT dataset, COUNT(*) as n_runs, SUM(CASE WHEN success THEN 1 ELSE 0 END) as n_success
FROM runs GROUP BY dataset ORDER BY dataset;
```

## Directory Layout

```
data/
  raw/           # Raw CAN bus CSVs (hcrl_ch, hcrl_sa, set_01-04)
  cache/         # Preprocessed graph tensors (.pt)
  datalake/      # Parquet structured storage (query with DuckDB)
    runs.parquet, metrics.parquet, configs.parquet, datasets.parquet, artifacts.parquet
    queries/           # SQL query files (leaderboard, kd_impact)
    training_curves/   # Per-run training loss/metric curves
    loss_landscapes/   # 2D loss surface visualizations
    artifacts/         # Registered model artifacts
experimentruns/  # Model checkpoints, configs, evaluation metrics
tools/           # Bundled CLI tools (DuckDB)
```

## Notes

- This data lives on scratch (`/fs/scratch/PAS1266/`) which has a 90-day purge policy.
  A weekly cron job touches all files to prevent purging.
- Ad-hoc queries: `duckdb < data/datalake/queries/leaderboard.sql`
