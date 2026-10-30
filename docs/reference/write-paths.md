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
| Repo | `/users/PAS2022/rf15/projects/lab/graphids` | source code |
| Lake | `$GRAPHIDS_LAKE_ROOT`, typically `/fs/ess/PAS1266/graphids` | raw data, caches, MLflow DB, MLflow artifacts |
| Runs | `GRAPHIDS_RUN_ROOT`, otherwise `{lake_root}/runs`, otherwise `{repo}/runs` | run journals, checkpoints, non-MLflow artifacts |
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
    scripts/ray-{experiment_name}.sbatch
    scripts/{experiment_name}-{config_hash}.yml
    ray-{experiment_name}_{job_id}.out
    ray-{experiment_name}_{job_id}.err

$GRAPHIDS_RUN_ROOT/{dataset}/{experiment_name}/{run_name}/
  resolved-run.yml
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

Event validation writes exactly two persistent files per completed epoch under
`artifacts/val-events-epochNNNN/`: `scores.bin` and `index.json`. The packed file
contains all six frame-score columns as little-endian FP64 rows. The schema-2
index stores shape/dtype, per-chunk file offsets, recording/source identity,
coverage, aggregate metrics and per-recording metrics. Input chunk size does not
change the persistent file count. Metric scratch mappings use local `TMPDIR`.

Read either legacy NPZ shards or packed artifacts with
`graphids.evaluation.iter_score_chunks`. To consolidate a complete legacy
epoch without changing its scores or metrics:

```bash
python scripts/pack_event_scores.py /path/to/val-events-epoch0000 --remove-shards
```

Migration verifies the packed data's SHA256 and recording/row coverage before
publishing the new index and removing the old shards. Incomplete epochs without
an index are not migrated. Score artifact schema changes do not alter model
checkpoint identity or the raw-message input contract.

October 23 cleanup archived obsolete run directories and logs outside ESS after
content comparisons, while retaining current inputs/caches, baseline results and
checkpoints. Inventories, archive paths, checksums and the final file-count audit
are under `/users/PAS2022/rf15/graphids-cleanup-audits/20261023T182040Z/`.

## Cache Paths

Lossless event caches live under the configured `config.data.cache_root`:

```text
{cache_root}/event-v1-{recording_digest}/
  metadata.json
  times.npy
  ids.npy
  payload.npy
  lengths.npy
  labels.npy
```

`recording_digest` is the SHA256 of the reviewed recording dataclass serialized
with sorted JSON keys. Cache construction verifies source bytes, builds in a
staging directory, and publishes the complete directory under a file lock.
The removed generic representation digest was not used by this active cache.

## Run Journals

Run manifests use `graphids._fs.atomic_write_json`:
indented, sorted UTF-8 JSON with a final newline, written through a unique
same-directory temporary file and atomic replacement. Failed serialization or
replacement preserves the last published file and removes the temporary file.

Ray Train workers run through `graphids.exp.run`, which writes:

| File | Purpose |
|---|---|
| `resolved-run.yml` | expanded `RunConfig` including payload, resources, and output paths |
| `.graphids/manifest.json` | authoritative status, final metrics, failure and result metadata |
| `.graphids/events.jsonl` | startup, finish and failure history |

`gx exp status <run_dir>` reads these files.

## MLflow

Training runs default to offline MLflow mode. They do not write to the shared
MLflow SQLite database during `fit`/`test`; instead they save
status and metrics in the schema-2 manifest. A separate single-writer step derives
an MLflow export from that manifest and `resolved-run.yml`. Historical separate
`mlflow_ingest.json` payloads remain readable:

```bash
gx exp ingest <run_dir>
gx exp ingest-root <root>
```

`GRAPHIDS_MLFLOW_MODE=online` restores the live MLflow logger for controlled
debug runs. In online mode, `graphids._mlflow.configure_tracking_uri()` defaults
MLflow to:

```text
sqlite:///{lake_root}/mlflow.db
```

`gx exp ingest` creates the MLflow run, logs tags/params/final metrics, copies
run artifacts, and marks the run `FINISHED` or `FAILED`. For Lightning `fit`
and `test`, final callback metrics are captured after the trainer returns and
stored in the manifest.

MLflow artifact roots are explicit lake paths:

```text
{lake_root}/mlartifacts/{dataset}/{stage}/
```

MLflow system metrics are sampled by `MLflowSystemMetricsCallback` only when
`GRAPHIDS_MLFLOW_MODE=online`. Offline diagnostic profiles can still record
aggregate `resource/*` CPU/RSS/GPU metrics through `graphids._mlflow.MLflowSystemMetricsSummaryCallback`
because those values are copied from final `trainer.callback_metrics` into
the manifest.

## SLURM

`gx exp submit <yaml>` writes one Ray allocation script per experiment name:

```text
{slurm_log_dir}/scripts/ray-{experiment_name}.sbatch
```

The script starts one Ray head inside the single-node Slurm allocation, then runs:

```bash
cd /users/PAS2022/rf15/graphids
source scripts/slurm/_preamble.sh
python -m graphids exp launch /abs/path/to/config.yml --address "${RAY_ADDRESS}"
source scripts/slurm/_epilog.sh
```

Stdout/stderr go to:

```text
{slurm_log_dir}/ray-{experiment_name}_%j.out
{slurm_log_dir}/ray-{experiment_name}_%j.err
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
  -> start one Ray head in the allocation
  -> python -m graphids exp launch <yaml> --address ${RAY_ADDRESS}
  -> ExperimentConfig.from_yaml
  -> run.launch_run
  -> run.execute_run
  -> fit | test
  -> authoritative manifest + lifecycle events
  -> scripts/slurm/_epilog.sh
```
