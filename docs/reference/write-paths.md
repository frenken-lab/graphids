# GraphIDS Write Path Inventory

> Audited 2026-04-09.

## The Rule

- **Code** lives in `~/graphids/` (NFS). Read-only at runtime.
- **ALL runtime writes** go to `/fs/ess/PAS1266/graphids/` (ESS), routed via `GRAPHIDS_LAKE_ROOT`.
- **Scratch** (`/fs/scratch/PAS1266/`) is for transient data: staged data copies.
- **Nothing** should write to the repo directory. Ever.

## Single Source of Truth

`graphids/config/constants.py` declares write path constants. `graphids/config/settings.py` owns all `GRAPHIDS_*` env vars.

Constants: `CKPT_SUBPATH`, `LAST_CKPT_SUBPATH`, `PHASE_MARKERS`, `CATALOG_SUBPATH`

## Filesystem Layout

```
/fs/ess/PAS1266/graphids/                        <-- $GRAPHIDS_LAKE_ROOT (persistent, shared)
+-- dev/{user}/{dataset}/
|   +-- {model}_{scale}_{stage}{identity}{kd}/
|       +-- seed_{N}/                             <-- trainer.default_root_dir
|           +-- checkpoints/
|           |   +-- best_model.ckpt               <-- ModelCheckpoint (dirpath pinned by instantiate.py)
|           |   +-- last.ckpt                     <-- ModelCheckpoint (save_last: true)
|           +-- traces.jsonl                      <-- OTel spans (wire_file_exporters, SimpleSpanProcessor)
|           +-- metrics.jsonl                     <-- OTel metrics (PeriodicExportingMetricReader, 10s)
|           +-- artifacts/                        <-- analysis outputs (written by `graphids analyze`, not the pipeline driver)
|           +-- .train_complete                   <-- phase marker (fit done; diagnostic only)
|           +-- .test_complete                    <-- phase marker (test done; diagnostic only)
+-- catalog/graphids.duckdb                      <-- (catalog builder removed 2026-04-10, pending redesign)
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
| OTel spans | `traces.jsonl` | `wire_file_exporters` -> `SimpleSpanProcessor` -> `ConsoleSpanExporter` |
| OTel metrics | `metrics.jsonl` | `wire_file_exporters` -> `PeriodicExportingMetricReader` (10s) |

### 2. OTel Instrumentation

`graphids/core/monitoring.py` — `OTelTrainingCallback` creates a `training.fit` span on fit start; records per-batch VRAM gauges, per-epoch events (LR, early stopping), final metrics, and best checkpoint path as span attributes. Tags the span with `campaign.manifest`/`campaign.cell_id` when `GRAPHIDS_CAMPAIGN_CELL` is set so `cell_statuses()` can derive cell state from `traces.jsonl`. Discovers upstream stage `traces.jsonl` files and records span links for KD lineage.

`graphids/_otel.py` — `wire_file_exporters(run_dir)` wires the file exporters (Phase B). Called from `cli/training.py::_prepare` and `orchestrate/stage.py::train`. Wandb Weave OTLP export is optional when `WANDB_API_KEY` is set.

`OTelTrainingLogger` captures `model.log()` calls as OTel histograms. Trainer wires it via `self.loggers` and calls `log_metrics` / `log_hyperparams` directly — no abstract base class, duck typing via attribute access.

### 3. Phase Markers

Diagnostic only — `run_pipeline`'s resume skip-check reads
`checkpoints/best_model.ckpt` directly, not these markers.

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

### 6. DuckDB Catalog

`{lake_root}/catalog/graphids.duckdb` — DuckDB catalog over `training.fit` OTel spans from `traces.jsonl`. The builder (`orchestrate/ops/catalog.py`) and `rebuild-catalog` CLI were removed 2026-04-10 pending redesign; no current way to populate. Disposable once rebuilt.

## Execution Order (pipeline path)

```
SLURM JOB (compute node)
------------------------
_preamble.sh (env, venv, stage data)
python -m graphids pipeline-run
+- run.run_pipeline(config)
|  +- ensure_spawn()
|  +- build_pipeline_stages(config)
|  +- for each StageConfig (with retry):
|     +- ResolvedConfig.resolve(...)
|     +- skip if checkpoints/best_model.ckpt exists     <-- checkpoint is authoritative
|     +- stage.build(resolved)
|     |   +- gc + torch.cuda reset
|     |   +- instantiate() -> trainer/model/datamodule
|     +- stage.train(artifacts, resolved)
|     |   +- wire_file_exporters(run_dir)
|     |   +- trainer.fit() -> touch .train_complete
|     +- stage.evaluate(artifacts, resolved)
|         +- trainer.test() -> touch .test_complete
|         +- save predictions/test/*.pt
+- _epilog.sh (GPU utilization report)

# Analysis is decoupled: run after the pipeline finishes.
python -m graphids analyze --ckpt-path <run_dir>/checkpoints/best_model.ckpt \
    --dataset <dataset>
```

## Env Var -> Path Mapping

| Env var | Default | Set in | Controls |
|---------|---------|--------|----------|
| `GRAPHIDS_LAKE_ROOT` | `"experimentruns"` (relative) | `.env` -> `/fs/ess/PAS1266/graphids` | All experiment IO |
| `GRAPHIDS_SLURM_LOG_DIR` | `{lake_root}/slurm` (derived) | `.env` | SLURM stdout/stderr |
| `GRAPHIDS_LAKE_WRITE` | `false` | `.env` (set to `1` in SLURM jobs) | Guards catalog/lake writes |
| `WANDB_API_KEY` | (none) | `.env` | Enables Wandb Weave OTLP export |
| `TMPDIR` | (SLURM sets) | OS | Per-job local SSD |
