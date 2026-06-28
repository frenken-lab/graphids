# Observability & Logging

> Status: **current**

GraphIDS has three observability surfaces:

- Per-run `.graphids/` journals and offline MLflow ingest payloads.
- MLflow SQLite for post-run params, tags, metrics, and artifacts.
- SLURM stdout/stderr logs for process text.

## Run Directory

```text
~/ray_results/{dataset}/{experiment_name}/{run_name}/
  .graphids/
    manifest.json
    events.jsonl
    mlflow_ingest.json
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
while running. They write `.graphids/mlflow_ingest.json`; a separate serialized
step replays that payload into MLflow:

```bash
gx exp ingest <run_dir>
gx exp ingest-root <root>
```

For controlled debugging, `GRAPHIDS_MLFLOW_MODE=online` restores the live
Lightning `MLFlowLogger`. In either path, the default MLflow tracking URI is:

```text
sqlite:///$GRAPHIDS_LAKE_ROOT/mlflow.db
```

Experiment names are created as:

```text
graphids/{dataset}/{stage}
```

Run names are `RunConfig.name`, normally the experiment name.

### Tags

`RunConfig.mlflow_tags()` writes:

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

`RunConfig.mlflow_hparams()` writes:

```text
graphids.stage
graphids.dataset
graphids.seed
graphids.plan_id
graphids.git_sha
graphids.run_dir
graphids.backend
graphids.representation
graphids.representation_cfg/*
graphids.payload/*
graphids.resource/*
```

Nested payload fields are flattened by `gx exp ingest`.

### Metrics

Lightning modules log keys such as:

```text
train_loss
train_acc
val_loss
val_acc
val_auroc
```

`graphids.exp.ray_backend` captures final `trainer.callback_metrics` after `fit`
and `test` into the offline ingest payload, so short runs still leave final
metric values.

System metrics are sampled through `MLflowSystemMetricsCallback` only in
`GRAPHIDS_MLFLOW_MODE=online`.

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
tail -n 80 /fs/ess/PAS1266/graphids/slurm_logs/gat_snapshot_sequence_real_47912353.err
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
    filter_string="tags.graphids.plan_id = 'gat_snapshot_sequence_real'",
)

for run in runs:
    print(run.info.run_id, run.data.metrics)
```

## MLflow UI

```bash
source .env
mlflow ui --backend-store-uri "sqlite:///$GRAPHIDS_LAKE_ROOT/mlflow.db" --port 5000
```
