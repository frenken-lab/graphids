# Observability & Logging

> Status: **current**

GraphIDS has three observability surfaces:

- Per-run saved configuration, authoritative manifest and lifecycle events.
- MLflow SQLite for post-run params, tags, metrics, and artifacts.
- SLURM stdout/stderr logs for process text.

## Run Directory

```text
{run_root}/{dataset}/{experiment_name}/{run_name}/
  resolved-run.yml
  .graphids/
    manifest.json
    events.jsonl
  checkpoints/
  artifacts/
```

`gx exp status <run_dir>` reads `.graphids/manifest.json` and
`.graphids/events.jsonl`.

## MLflow

By default:

```text
GRAPHIDS_MLFLOW_MODE=offline
```

Training and evaluation jobs do not write to the shared MLflow SQLite database
while running. Schema-2 manifests hold final status, metrics and failures; a
separate serialized step derives an MLflow export from that manifest and
`resolved-run.yml`. Historical `mlflow_ingest.json` files remain supported:

```bash
gx exp ingest <run_dir>
gx exp ingest-root <root>
```

For controlled debugging, `GRAPHIDS_MLFLOW_MODE=online` restores the live
Lightning `MLFlowLogger`. In either path, the default MLflow tracking URI is:

```text
sqlite:///$GRAPHIDS_LAKE_ROOT/mlflow.db
```

`GRAPHIDS_MLFLOW_MODE=disabled` (also `off` or `none`) keeps results local.
The saved manifest records this choice; batch ingestion skips those runs.

Experiment names are created as:

```text
graphids/{dataset}/{stage}
```

Run names are `RunConfig.name`, normally the experiment name.

### Tags

`graphids.exp.tracking.mlflow_tags(run)` writes:

```text
graphids.stage
graphids.run_dir
graphids.git_sha
graphids.dataset
graphids.seed
graphids.plan_id
graphids.representation
```

### Params

`graphids.exp.tracking.mlflow_hparams(run)` writes:

```text
graphids.stage
graphids.dataset
graphids.seed
graphids.plan_id
graphids.git_sha
graphids.run_dir
graphids.representation
graphids.representation_cfg/*
graphids.payload/*
graphids.resource/*
```

Nested payload fields are flattened by `gx exp ingest`.

### Metrics

The event model logs separately normalized classification and normal-only payload
losses, plus exact frame metrics such as `val/classification_frame_ap`.
`graphids.exp.run` captures scalar `trainer.callback_metrics` after
`fit` and `test` into the authoritative manifest; vector metrics are omitted.

`graphids._mlflow.MLflowSystemMetricsCallback` records phase throughput under
`.graphids/phase_metrics.json`. When a live MLflow run is available it also starts
MLflow's native system metrics monitor.
Phase bookkeeping accumulates detached integer event counts and float64 time
sums on each batch device. It transfers totals once at phase end before recording
elapsed wall time. The scored mask determines event counts independently of
label availability; exceptions discard unfinished counters.


Offline diagnostic runs can include
`graphids._mlflow.MLflowSystemMetricsSummaryCallback`. It maps native MLflow
histories to `resource/*` CPU, system-memory, and GPU summaries, which are copied
from final callback metrics into the manifest for subsequent MLflow export.
`resource/rss_gib_*` is a retained metric name for system memory usage, not
process RSS. Missing optional GPU histories leave GPU summary keys absent.
Summary metric families share history collection and reductions. GPU utilization
means pool samples across devices, memory summaries retain their existing units,
and `resource/samples` is the largest individual history length. Disabling GPU
summaries avoids GPU-history lookups; unavailable histories remain optional.


## SLURM Logs

Current jobs write under:

```text
/fs/ess/PAS1266/graphids/slurm_logs/
  scripts/{experiment_name}.sbatch
  {experiment_name}_{job_id}.out
  {experiment_name}_{job_id}.err
```

Examples:

```bash
tail -n 80 /fs/ess/PAS1266/graphids/slurm_logs/graph_mamba_events_source_seed42_47912353.err
sacct -j 47912353 --format=JobID,JobName%36,State,ExitCode,Elapsed -P
squeue -j 47912353 -o '%i %j %T %M %R'
```

## Querying MLflow

```python
from graphids._mlflow import configure_tracking_uri
from mlflow.tracking import MlflowClient

configure_tracking_uri()
client = MlflowClient()
experiments = client.search_experiments(filter_string="name LIKE 'graphids/%'")
exp_ids = [e.experiment_id for e in experiments]

runs = client.search_runs(
    exp_ids,
    filter_string="tags.graphids.plan_id = 'graph_mamba_events_source_seed42'",
)

for run in runs:
    print(run.info.run_id, run.data.metrics)
```

## MLflow UI

```bash
source .env
mlflow ui --backend-store-uri "sqlite:///$GRAPHIDS_LAKE_ROOT/mlflow.db" --port 5000
```
