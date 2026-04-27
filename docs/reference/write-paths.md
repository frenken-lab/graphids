# GraphIDS Write Path Inventory

> Audited 2026-04-09.

## The Rule

- **Code** lives in `~/graphids/` (NFS). Read-only at runtime.
- **ALL runtime writes** go to `/fs/ess/PAS1266/graphids/` (ESS), routed via `GRAPHIDS_LAKE_ROOT`.
- **Scratch** (`/fs/scratch/PAS1266/`) is for transient data: staged data copies.
- **Nothing** should write to the repo directory. Ever.

## Single Source of Truth

`graphids/config/constants.py` declares write path constants. `graphids/config/settings.py` owns all `GRAPHIDS_*` env vars.

Constants: `CKPT_SUBPATH`, `LAST_CKPT_SUBPATH`, `PHASE_MARKERS` (all in `graphids/config/constants.py`). MLflow backend path (`mlflow.db`) and artifact subpath (`mlartifacts`) live in `graphids/_mlflow.py`.

## Filesystem Layout

```
/fs/ess/PAS1266/graphids/                        <-- $GRAPHIDS_LAKE_ROOT (persistent, shared)
+-- dev/{user}/{dataset}/
|   +-- ablations/{group}/{variant}/
|       +-- seed_{N}/                             <-- trainer.default_root_dir
|           +-- checkpoints/
|           |   +-- best_model.ckpt               <-- ModelCheckpoint (dirpath pinned by instantiate.py)
|           |   +-- last.ckpt                     <-- ModelCheckpoint (save_last: true)
|           +-- traces.jsonl                      <-- OTel spans (wire_file_exporters, SimpleSpanProcessor)
|           +-- artifacts/                        <-- analysis outputs (written by `graphids analyze`, not the pipeline driver)
|           +-- .train_complete                   <-- phase marker (fit done; diagnostic only)
|           +-- .test_complete                    <-- phase marker (test done; diagnostic only)
|           +-- resolved.json                     <-- Pydantic-validated rendered config (cli/training._prepare)
|           +-- overrides.json                    <-- TLA dict + --set payload (cli/training._prepare)
+-- mlflow.db                                    <-- MLflow SQLite backend (runs + params + metrics + tags)
+-- mlartifacts/{exp_id}/{run_id}/               <-- MLflow artifact store (per-experiment per-run)
+-- raw/{dataset}/                               <-- source CSV data
+-- cache/v{ver}/{dataset}/                      <-- preprocessed graph .pt files
+-- slurm/                                       <-- SLURM stdout/stderr (default)

/fs/scratch/PAS1266/                             <-- transient (90-day purge)
+-- graphids-data/                               <-- staged data (scratch -> TMPDIR)

$TMPDIR/graphids-data/                           <-- per-job local SSD (ephemeral)
```

Checkpoint dirpath is owned by `core.callbacks.ModelCheckpoint._resolve_dirpath`: `{default_root_dir}/checkpoints` unless an explicit `dirpath` is configured. The `/checkpoints` subdir convention lives on the callback, not the instantiator.

## Write Path Detail

### 1. Trainer (custom, post-Lightning)

All training writes land under `trainer.default_root_dir` from the rendered jsonnet config.

| What | Relative path | Who writes |
|------|---------------|-----------|
| Best checkpoint | `checkpoints/best_model.ckpt` | `core.callbacks.ModelCheckpoint` (self-describing — `class_path` + `state_dict` + `hyper_parameters`) |
| Resume checkpoint | `checkpoints/last.ckpt` | `ModelCheckpoint` (`save_last: true`) |
| Train/val predictions | `predictions/{train,val}.pt` | `orchestrate.stage.train` after `trainer.fit` |
| Per-test-set predictions | `predictions/test/{set_name}.pt` | `orchestrate.stage.evaluate` |
| OTel spans + log events | `traces.jsonl` | `wire_file_exporters` -> `BatchSpanProcessor` -> `ConsoleSpanExporter` |

### 2. Training-time tracking

`graphids/_mlflow.py::start_training_run` opens the MLflow run in `stage.train` before `trainer.fit`, logs params + tags + cache digest, and enables the MLflow system-metrics sampler (psutil + nvidia-ml-py, 5s interval).

`graphids/core/mlflow_callback.py::MLflowTrainingCallback` forwards every key in `trainer.callback_metrics` (whatever the model logged via `self.log(...)`) plus `lr` + `early_stop.{wait,best_score}` to MLflow at `step=epoch` via `on_train_epoch_end`, stamps `peak_vram_mb` + `epochs_run` + ckpt SHA256 tag at `on_fit_end`, and closes the run (FINISHED / FAILED via `on_exception`).

`graphids/_otel.py::wire_file_exporters` wires the `traces.jsonl` span exporter (Phase B). Structured-log events emitted via `log.info("event_name", ...)` land here alongside the single `training.fit` span. Wandb Weave OTLP export is optional when `WANDB_API_KEY` is set.

### 3. Phase Markers

Diagnostic only — resume is authoritative on `checkpoints/best_model.ckpt`
existence, not these markers.

| What | Path | Who writes | Code |
|------|------|-----------|------|
| Train phase marker | `{run_dir}/.train_complete` | `stage.py::train` after `trainer.fit` | `PHASE_MARKERS["train"]` |
| Test phase marker | `{run_dir}/.test_complete` | `stage.py::evaluate` after `trainer.test` | `PHASE_MARKERS["test"]` |
| Analysis artifacts | `{run_dir}/artifacts/` | `core/analysis/analyzer.py` via `python -m graphids analyze` (not pipeline) | `analyze.py` |

### 4. SLURM

| What | Path | Who writes |
|------|------|-----------|
| Job stdout/stderr | `{slurm_log_dir}/{name}_%j.{out,err}` | sbatch/OS |

`slurm_log_dir` defaults to `{lake_root}/slurm` (`settings.py:46`); override via `GRAPHIDS_SLURM_LOG_DIR`.

### 5. Data / Preprocessing

| What | Path | Who writes |
|------|------|-----------|
| Graph cache .pt files | `{lake_root}/cache/v{ver}/{dataset}/processed/data_*.pt` | `core/data/io.py::atomic_save` |
| Cache metadata (v2) | `{lake_root}/cache/v{ver}/{dataset}/cache_metadata.json` | `core/data/metadata.py::merge_split_into_metadata` |
| NFS advisory lock | `{cache_dir}/processed/.lock` | `core/data/io.py::nfs_lock` |
| Metadata merge lock | `{cache_dir}/.metadata_lock` | `core/data/metadata.py` (fcntl.flock) |

### 6. MLflow Store

`{lake_root}/mlflow.db` — MLflow SQLite backend (runs, params, metrics, tags). Written by `graphids/_mlflow.py::log_run` at the end of `orchestrate/stage.py::evaluate`. Each run is an MLflow row keyed by `run_name = {group}_{variant}_{dataset}_seed{N}[_{cluster}]`. Artifacts (if any) land under `{lake_root}/mlartifacts/{exp_id}/{run_id}/`. Browse via the OSC OnDemand MLflow app pointed at the SQLite URI.

## Execution Order

```
SLURM JOB (compute node)
------------------------
_preamble.sh (env, venv)
python -m graphids fit --config <preset.jsonnet> --tla dataset=... --tla seed=...
+- ensure_spawn()
+- render(config, tla)
+- apply_overrides(rendered, --set ...)
+- ResolvedConfig.from_rendered(rendered)
+- build(resolved)
|   +- gc + torch.cuda reset
|   +- build_run(rendered) -> trainer/model/datamodule
+- wire_file_exporters(run_dir)
+- train(artifacts, resolved, resume_from=...)
|   +- trainer.fit() -> touch .train_complete
# (separate invocation for eval)
python -m graphids test --config <preset.jsonnet> --tla dataset=... --tla seed=...
+- ... evaluate(...) -> trainer.test() -> touch .test_complete

# Analysis is decoupled:
python -m graphids analyze --ckpt-path <run_dir>/checkpoints/best_model.ckpt \
    --dataset <dataset>
```

## Env Var -> Path Mapping

| Env var | Default | Set in | Controls |
|---------|---------|--------|----------|
| `GRAPHIDS_LAKE_ROOT` | `"experimentruns"` (relative) | `.env` -> `/fs/ess/PAS1266/graphids` | All experiment IO |
| `GRAPHIDS_SLURM_LOG_DIR` | `{lake_root}/slurm` (derived) | `.env` | SLURM stdout/stderr |
| `GRAPHIDS_LAKE_WRITE` | `false` | `.env` (set to `1` in SLURM jobs) | Guards lake writes |
| `MLFLOW_TRACKING_URI` | `sqlite:///{lake_root}/mlflow.db` | derived by `_mlflow.ensure_tracking_uri` | MLflow backend |
| `WANDB_API_KEY` | (none) | `.env` | Enables Wandb Weave OTLP export |
| `TMPDIR` | (SLURM sets) | OS | Per-job local SSD |
