# GraphIDS Write Path Inventory

> Audited 2026-04-09.

## The Rule

- **Code** lives in `~/graphids/` (NFS). Read-only at runtime.
- **ALL runtime writes** go to `/fs/ess/PAS1266/graphids/` (ESS), routed via `GRAPHIDS_LAKE_ROOT`.
- **Scratch** (`/fs/scratch/PAS1266/`) is for transient data: staged data copies.
- **Nothing** should write to the repo directory. Ever.

## Single Source of Truth

`graphids/config/constants.py` declares write path constants. `graphids/config/settings.py` owns all `GRAPHIDS_*` env vars.

Constants: `CKPT_SUBPATH`, `LAST_CKPT_SUBPATH`, `COMPLETE_MARKER`, `PHASE_MARKERS`, `CATALOG_SUBPATH`

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
|           +-- artifacts/                        <-- analysis outputs (embeddings, CKA, landscape)
|           +-- .complete                         <-- pipeline marker (evaluate done)
|           +-- .train_complete                   <-- pipeline marker (fit done)
|           +-- .test_complete                    <-- pipeline marker (test done)
|           +-- .analyze_complete                 <-- pipeline marker (analyze done)
+-- catalog/graphids.duckdb                      <-- (catalog builder removed 2026-04-10, pending redesign)
+-- raw/{dataset}/                               <-- source CSV data
+-- cache/v{ver}/{dataset}/                      <-- preprocessed graph .pt files
+-- slurm/                                       <-- SLURM stdout/stderr (default)

/fs/scratch/PAS1266/                             <-- transient (90-day purge)
+-- graphids-data/                               <-- staged data (scratch -> TMPDIR)

$TMPDIR/graphids-data/                           <-- per-job local SSD (ephemeral)
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

`graphids/_otel.py` — `wire_file_exporters(run_dir)` wires the file exporters (Phase B). Called from `cli/_training.py::_prepare` and `orchestrate/stage.py::train`. Wandb Weave OTLP export is optional when `WANDB_API_KEY` is set.

`OTelTrainingLogger` captures Lightning `self.log()` calls as OTel histograms.

### 3. Pipeline Markers

| What | Path | Who writes | Code |
|------|------|-----------|------|
| Train complete marker | `{run_dir}/.train_complete` | `stage.py::train` after `trainer.fit` | `PHASE_MARKERS["train"]` |
| Test complete marker | `{run_dir}/.test_complete` | `stage.py::evaluate` after `trainer.test` | `PHASE_MARKERS["test"]` |
| Analyze complete marker | `{run_dir}/.analyze_complete` | `run.py::_run_one_stage` after `run_single_analysis` | `PHASE_MARKERS["analyze"]` |
| Run complete marker | `{run_dir}/.complete` | `stage.py::evaluate` at end (unconditional) | `COMPLETE_MARKER` |
| Analysis artifacts | `{run_dir}/artifacts/` | `analyze.py::run_single_analysis` via `AnalysisSpec` | `analyze.py` |

### 4. SLURM

| What | Path | Who writes |
|------|------|-----------|
| Job stdout/stderr | `{slurm_log_dir}/{name}_%j.{out,err}` | sbatch/OS |

`slurm_log_dir` defaults to `{lake_root}/slurm` (`settings.py:46`); override via `GRAPHIDS_SLURM_LOG_DIR`.

### 5. Data / Preprocessing

| What | Path | Who writes |
|------|------|-----------|
| Graph cache .pt files | `{lake_root}/cache/v{ver}/{dataset}/` | `preprocessing/utils.py` (atomic_save) |
| NFS advisory lock | `{cache_dir}/.lock` | preprocessing/utils.py |
| Staging marker | `{scratch}/graphids-data/.staged_marker` | stage_data.sh |
| Node-local staged data | `$TMPDIR/graphids-data/` | stage_data.sh |

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
|     +- skip if .complete marker present
|     +- stage.build(rendered, validated)
|     |   +- gc + torch.cuda reset
|     |   +- instantiate() -> trainer/model/datamodule
|     +- stage.train(artifacts, ...)
|     |   +- wire_file_exporters(run_dir)
|     |   +- trainer.fit() -> touch .train_complete
|     +- stage.evaluate(artifacts, ...)
|     |   +- trainer.test() -> touch .test_complete
|     |   +- touch .complete (unconditional)
|     +- if analyzable (vgae/dgi/gat):
|        run_single_analysis(spec)
|        +- Analyzer(...).run()
|        +- write analysis_manifest.json
|        +- touch .analyze_complete
+- _epilog.sh (GPU utilization report)
```

## Env Var -> Path Mapping

| Env var | Default | Set in | Controls |
|---------|---------|--------|----------|
| `GRAPHIDS_LAKE_ROOT` | `"experimentruns"` (relative) | `.env` -> `/fs/ess/PAS1266/graphids` | All experiment IO |
| `GRAPHIDS_SLURM_LOG_DIR` | `{lake_root}/slurm` (derived) | `.env` | SLURM stdout/stderr |
| `GRAPHIDS_LAKE_WRITE` | `false` | `.env` (set to `1` in SLURM jobs) | Guards catalog/lake writes |
| `WANDB_API_KEY` | (none) | `.env` | Enables Wandb Weave OTLP export |
| `TMPDIR` | (SLURM sets) | OS | Per-job local SSD |
