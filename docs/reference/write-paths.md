# GraphIDS Write Path Inventory

> Status: **current**

## Rule

- Source code lives in the repo and should be read-only at runtime.
- Shared persistent data lives under `$GRAPHIDS_LAKE_ROOT`.
- Per-run manifests, events, checkpoints, and artifacts live under the
  resolved run directory.
- SLURM text logs live under the configured SLURM log directory.

## Roots

| Root | Default/current behavior | Owner |
|---|---|---|
| Repo | `/users/PAS2022/rf15/graphids` | source code |
| Lake | `$GRAPHIDS_LAKE_ROOT`, typically `/fs/ess/PAS1266/graphids` | raw data, caches, MLflow DB, MLflow artifacts |
| Runs | `Path.home() / "ray_results"` outside Ray | run journals, checkpoints, non-MLflow artifacts |
| SLURM logs | `GRAPHIDS_SLURM_LOG_DIR`, `.env`, or `{lake_root}/slurm`; current jobs use `/fs/ess/PAS1266/graphids/slurm_logs` | sbatch scripts, stdout, stderr |

Relevant code:

- `graphids/paths.py`: lake, cache, run, and catalog helpers.
- `graphids/exp/config.py`: `OutputConfig` and run-directory resolution.
- `graphids/exp/slurm.py`: SLURM script/log path resolution.
- `graphids/_mlflow.py`: MLflow tracking URI.

## Filesystem Layout

```text
$GRAPHIDS_LAKE_ROOT/
  raw/{dataset}/
  cache/v{PREPROCESSING_VERSION}/{dataset}/
  mlflow.db
  mlartifacts/{dataset}/{stage}/
  slurm_logs/
    scripts/{experiment_name}.sbatch
    {experiment_name}_{job_id}.out
    {experiment_name}_{job_id}.err

~/ray_results/{dataset}/{experiment_name}/{run_name}/
  .graphids/
    manifest.json
    events.jsonl
  checkpoints/
    best_model.ckpt
    best_model.ckpt.sha256
    last.ckpt
    last.ckpt.sha256
  artifacts/
```

Checkpoint files exist only when the experiment config enables checkpointing
and includes a checkpoint callback.

## Cache Paths

Graph caches are versioned by `graphids.paths.PREPROCESSING_VERSION`.

Snapshot and snapshot-sequence graph caches currently live under:

```text
{lake_root}/cache/v{PREPROCESSING_VERSION}/{dataset}/{representation_kind}_{representation_digest}_voc_{scope}/
  processed/
    data_train.pt
    data_test_<split>.pt
    .complete
  cache_metadata.json
```

The representation digest is part of the cache path and cache key, so changing
representation settings creates a distinct cache.

## Run Journals

`graphids.exp.runtime.launch_run` writes:

| File | Purpose |
|---|---|
| `.graphids/manifest.json` | resolved run identity, config, outputs, status, failure |
| `.graphids/events.jsonl` | launch, finish, and failure events |

`gx exp status <run_dir>` reads these files.

## MLflow

`graphids._mlflow.configure_tracking_uri()` defaults MLflow to:

```text
sqlite:///{lake_root}/mlflow.db
```

`graphids.exp.runtime.launch_run` creates an MLflow logger, logs run
hyperparameters/tags, and marks the run `FINISHED` or `FAILED`. For Lightning
`fit` and `test`, final callback metrics are explicitly logged after the
trainer returns.

MLflow artifact roots are explicit lake paths:

```text
{lake_root}/mlartifacts/{dataset}/{stage}/
```

MLflow system metrics are sampled by `MLflowSystemMetricsCallback` for
Lightning-created runs.

## SLURM

`gx exp submit <yaml>` writes one script per experiment name:

```text
{slurm_log_dir}/scripts/{experiment_name}.sbatch
```

The script runs:

```bash
cd /users/PAS2022/rf15/graphids
source scripts/slurm/_preamble.sh
python -m graphids exp launch /abs/path/to/config.yml
source scripts/slurm/_epilog.sh
```

Stdout/stderr go to:

```text
{slurm_log_dir}/{experiment_name}_%j.out
{slurm_log_dir}/{experiment_name}_%j.err
```

## Execution Order

```text
gx exp submit <yaml>
  -> ExperimentConfig.from_yaml
  -> build RunConfig for validation
  -> write sbatch script
  -> sbatch

compute node:
  -> scripts/slurm/_preamble.sh
  -> python -m graphids exp launch <yaml>
  -> ExperimentConfig.from_yaml
  -> launch_run
  -> run_stage
  -> manifest/events + MLflow
  -> scripts/slurm/_epilog.sh
```
