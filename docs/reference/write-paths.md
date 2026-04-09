# GraphIDS Write Path Inventory

> Audited 2026-04-09.

## The Rule

- **Code** lives in `~/KD-GAT/` (NFS). Read-only at runtime.
- **ALL runtime writes** go to `/fs/ess/PAS1266/kd-gat/` (ESS), routed via `KD_GAT_LAKE_ROOT`.
- **Scratch** (`/fs/scratch/PAS1266/`) is for transient data: staged data copies.
- **Nothing** should write to the repo directory. Ever.

## Single Source of Truth

`graphids/config/constants.py` declares write path constants. `graphids/config/settings.py` owns all `KD_GAT_*` env vars.

Constants: `CKPT_SUBPATH`, `LAST_CKPT_SUBPATH`, `COMPLETE_MARKER`, `PHASE_MARKERS`, `CATALOG_SUBPATH`

## Filesystem Layout

```
/fs/ess/PAS1266/kd-gat/                          <-- $KD_GAT_LAKE_ROOT (persistent, shared)
+-- dev/{user}/{dataset}/
|   +-- {model}_{scale}_{stage}{identity}{kd}/
|       +-- seed_{N}/                             <-- trainer.default_root_dir
|           +-- checkpoints/
|           |   +-- best_model.ckpt               <-- ModelCheckpoint (dirpath pinned by instantiate.py)
|           |   +-- last.ckpt                     <-- ModelCheckpoint (save_last: true)
|           +-- traces.jsonl                      <-- OTel spans (wire_file_exporters, SimpleSpanProcessor)
|           +-- metrics.jsonl                     <-- OTel metrics (PeriodicExportingMetricReader, 10s)
|           +-- artifacts/                        <-- analysis outputs (embeddings, CKA, landscape)
|           +-- .complete                         <-- Monarch marker (eval_stage done)
|           +-- .train_complete                   <-- Monarch marker (fit done)
|           +-- .test_complete                    <-- Monarch marker (test done)
|           +-- .analyze_complete                 <-- Monarch marker (analyze done)
+-- catalog/kd_gat.duckdb                        <-- rebuilt from traces.jsonl (rebuild-catalog)
+-- raw/{dataset}/                               <-- source CSV data
+-- cache/v{ver}/{dataset}/                      <-- preprocessed graph .pt files
+-- slurm/                                       <-- SLURM stdout/stderr (default)

/fs/scratch/PAS1266/                             <-- transient (90-day purge)
+-- kd-gat-data/                                 <-- staged data (scratch -> TMPDIR)

$TMPDIR/kd-gat-data/                             <-- per-job local SSD (ephemeral)
```

Checkpoint dirpath is pinned at runtime: `instantiate.py` sets `ModelCheckpoint.dirpath` to `{default_root_dir}/checkpoints` derived from `CKPT_SUBPATH` (`constants.py:89`).

## Write Path Detail

### 1. Lightning (via Trainer)

All Lightning writes land under `trainer.default_root_dir` from the rendered jsonnet config.

| What | Relative path | Who writes |
|------|---------------|-----------|
| Best checkpoint | `checkpoints/best_model.ckpt` | ModelCheckpoint |
| Resume checkpoint | `checkpoints/last.ckpt` | ModelCheckpoint (`save_last: true`) |
| OTel spans | `traces.jsonl` | `wire_file_exporters` -> `SimpleSpanProcessor` -> `ConsoleSpanExporter` |
| OTel metrics | `metrics.jsonl` | `wire_file_exporters` -> `PeriodicExportingMetricReader` (10s) |

### 2. OTel Instrumentation

`graphids/core/monitoring.py` — `OTelTrainingCallback` creates a `training.fit` span on fit start; records per-batch VRAM gauges, per-epoch events (LR, early stopping), final metrics, and best checkpoint path as span attributes. Discovers upstream stage `traces.jsonl` files and records span links for KD lineage.

`graphids/core/otel.py` — `wire_file_exporters(run_dir)` wires the file exporters (Phase B). Called from `cli/_training.py:37` and `orchestrate/actors.py:130`. Wandb Weave OTLP export is optional when `WANDB_API_KEY` is set.

`OTelTrainingLogger` captures Lightning `self.log()` calls as OTel histograms.

### 3. Monarch Orchestration

| What | Path | Who writes | Code |
|------|------|-----------|------|
| Train complete marker | `{run_dir}/.train_complete` | `actors.py` after `trainer.fit` | `PHASE_MARKERS["train"]` |
| Test complete marker | `{run_dir}/.test_complete` | `actors.py` after `trainer.test` | `PHASE_MARKERS["test"]` |
| Analyze complete marker | `{run_dir}/.analyze_complete` | `actors.py` after `run_analysis` | `PHASE_MARKERS["analyze"]` |
| Run complete marker | `{run_dir}/.complete` | `actors.py` at `eval_stage` end | `COMPLETE_MARKER` |
| Analysis artifacts | `{run_dir}/artifacts/` | `run_analysis` via `AnalysisSpec` | `actors.py:182` |

### 4. SLURM

| What | Path | Who writes |
|------|------|-----------|
| Job stdout/stderr | `{slurm_log_dir}/{name}_%j.{out,err}` | sbatch/OS |

`slurm_log_dir` defaults to `{lake_root}/slurm` (`settings.py:46`); override via `KD_GAT_SLURM_LOG_DIR`.

### 5. Data / Preprocessing

| What | Path | Who writes |
|------|------|-----------|
| Graph cache .pt files | `{lake_root}/cache/v{ver}/{dataset}/` | `preprocessing/utils.py` (atomic_save) |
| NFS advisory lock | `{cache_dir}/.lock` | preprocessing/utils.py |
| Staging marker | `{scratch}/kd-gat-data/.staged_marker` | stage_data.sh |
| Node-local staged data | `$TMPDIR/kd-gat-data/` | stage_data.sh |

### 6. DuckDB Catalog

`{lake_root}/catalog/kd_gat.duckdb` — `runs` table rebuilt by `orchestrate/ops/catalog.py` from `traces.jsonl` files. Scans `{lake_root}/dev/**/traces.jsonl`, filters `training.fit` spans, extracts identity/status/metrics as columns. Disposable — rebuildable via `python -m graphids rebuild-catalog`. Requires `KD_GAT_LAKE_WRITE=1`.

## Execution Order (Monarch path)

```
LOGIN NODE (Monarch)                    SLURM JOB (compute node)
--------------------                    ------------------------
monarch.run_pipeline()
+- check .complete marker (skip?)
+- submit SLURM job ----------------->  sbatch allocates node
|                                       +- _preamble.sh (env, venv, stage data)
|                                       +- actors.py::train_stage()
|                                       |   +- wire_file_exporters(run_dir)
|                                       |   |   +- opens traces.jsonl, metrics.jsonl
|                                       |   +- instantiate() -> trainer/model/datamodule
|                                       |   |   +- ModelCheckpoint.dirpath pinned
|                                       |   +- trainer.fit()
|                                       |   |   +- OTelTrainingCallback spans + gauges
|                                       |   |   +- OTelTrainingLogger -> metrics.jsonl
|                                       |   |   +- ModelCheckpoint -> best_model.ckpt
|                                       |   +- touch .train_complete
|                                       +- actors.py::eval_stage()
|                                       |   +- trainer.test() -> touch .test_complete
|                                       |   +- run_analysis() -> artifacts/ -> touch .analyze_complete
|                                       |   +- touch .complete
|                                       +- _epilog.sh (GPU utilization report)
+- poll -> COMPLETED
```

## Env Var -> Path Mapping

| Env var | Default | Set in | Controls |
|---------|---------|--------|----------|
| `KD_GAT_LAKE_ROOT` | `"experimentruns"` (relative) | `.env` -> `/fs/ess/PAS1266/kd-gat` | All experiment IO |
| `KD_GAT_SLURM_LOG_DIR` | `{lake_root}/slurm` (derived) | `.env` | SLURM stdout/stderr |
| `KD_GAT_LAKE_WRITE` | `false` | `.env` (set to `1` in SLURM jobs) | Guards catalog/lake writes |
| `WANDB_API_KEY` | (none) | `.env` | Enables Wandb Weave OTLP export |
| `TMPDIR` | (SLURM sets) | OS | Per-job local SSD |
