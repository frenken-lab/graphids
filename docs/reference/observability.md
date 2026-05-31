# Observability & Logging

> Status: **current**

GraphIDS has three live observability surfaces:

- MLflow SQLite for params, tags, metrics, and system metrics.
- `.graphids/` journals in each run directory for launch status.
- SLURM stdout/stderr logs for process text.

## Run Directory

```text
~/ray_results/{dataset}/{experiment_name}/{run_name}/
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

Nested payload fields are flattened by Lightning's `MLFlowLogger`.

### Metrics

Lightning modules log keys such as:

```text
train_loss
train_acc
val_loss
val_acc
val_auroc
```

`graphids.exp.runtime` explicitly logs final `trainer.callback_metrics` after
`fit` and `test`, so short runs still leave final metric values.

System metrics are sampled through `MLflowSystemMetricsCallback` when a
Lightning run starts.

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
