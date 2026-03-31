# KD-GAT Write Path Inventory

> Audited 2026-03-30. Updated same day after consolidation.

## The Rule

- **Code** lives in `~/KD-GAT/` (NFS). Read-only at runtime.
- **ALL runtime writes** go to `/fs/ess/PAS1266/kd-gat/` (ESS), routed via `KD_GAT_LAKE_ROOT` env var.
- **Scratch** (`/fs/scratch/PAS1266/`) is for transient data: wandb run files, dagster state DB, staged data.
- **Nothing** should write to the repo directory. Ever.
- **All training goes through dagster.** No direct CLI runs to ESS. Dagster sets `default_root_dir`, `cli.py` pins `dirpath`.

## Single Source of Truth

`graphids/config/write_paths.yaml` declares every write path. `config/__init__.py` loads it and exports constants. Python code imports constants — no hardcoded path strings.

```
write_paths.yaml → config/__init__.py → cli.py, component.py, slurm.py
```

Constants exported: `CKPT_SUBPATH`, `LAST_CKPT_SUBPATH`, `COMPLETE_MARKER`, `DAGSTER_IO_DIR_TEMPLATE`, `DAGSTER_HOME_DEFAULT`

## Filesystem Layout

```
/fs/ess/PAS1266/kd-gat/                          ← $KD_GAT_LAKE_ROOT (persistent, shared)
├── dev/{user}/{dataset}/                         ← run_dir() base
│   └── {model}_{scale}_{stage}{identity}{kd}/
│       └── seed_{N}/                             ← trainer.default_root_dir
│           ├── checkpoints/
│           │   ├── best_model.ckpt               ← ModelCheckpoint (dirpath pinned by cli.py)
│           │   └── last.ckpt                     ← ModelCheckpoint (save_last: true)
│           ├── lightning_logs/version_*/
│           │   ├── metrics.csv                   ← CSVLogger (diagnostics only)
│           │   ├── hparams.yaml                  ← save_hyperparameters()
│           │   └── config.yaml                   ← SaveConfigCallback
│           └── .complete                         ← dagster marker
├── .dagster/io/{asset_key}/{partition}.json       ← IOManager sidecar
├── raw/{dataset}/                                ← source CSV data
├── cache/v{ver}/{dataset}/                       ← preprocessed graph .pt files
└── slurm_logs/                                   ← SLURM stdout/stderr

/fs/scratch/PAS1266/                              ← transient (90-day purge)
├── wandb/                                        ← $WANDB_DIR (currently dead)
├── dagster/                                      ← $DAGSTER_HOME (SQLite event log)
└── kd-gat-data/                                  ← staged data (scratch → TMPDIR)

$TMPDIR/kd-gat-data/                              ← per-job local SSD (ephemeral)
```

Key change from initial audit: checkpoints are now at `{run_dir}/checkpoints/`, **decoupled from CSVLogger versioning**. CSVLogger still writes to `lightning_logs/version_N/` but nothing depends on that path.

## Write Path Detail

### 1. Lightning (via Trainer)

All Lightning writes land under `trainer.default_root_dir`, set by dagster via `--trainer.default_root_dir={rd}`.

| What | Relative path | Who writes | Config |
|------|---------------|-----------|--------|
| Best checkpoint | `checkpoints/best_model.ckpt` | ModelCheckpoint | `trainer.yaml`, `cli.py` pins dirpath |
| Resume checkpoint | `checkpoints/last.ckpt` | ModelCheckpoint | `save_last: true` in trainer.yaml |
| Metrics CSV | `lightning_logs/version_*/metrics.csv` | CSVLogger | Lightning default logger |
| Hyperparameters | `lightning_logs/version_*/hparams.yaml` | CSVLogger | automatic |
| Resolved config | `lightning_logs/version_*/config.yaml` | SaveConfigCallback | `cli.py` (overwrite: True) |

**Checkpoint path is pinned**: `cli.py` `before_instantiate_classes` sets `ModelCheckpoint.dirpath` to `{default_root_dir}/checkpoints`. Derived from `CKPT_SUBPATH` (loaded from `write_paths.yaml`). No version directory in checkpoint path.

**CSVLogger versioning is irrelevant**: metrics/hparams/config go to `lightning_logs/version_N/` but nothing depends on the version number. Auto-increment on crash is harmless.

### 2. Dagster Orchestration

| What | Path | Who writes | Constant |
|------|------|-----------|----------|
| Checkpoint path sidecar | `{lake_root}/.dagster/io/{asset_key}/{partition}.json` | CheckpointPathIOManager | `DAGSTER_IO_DIR_TEMPLATE` |
| Run completion marker | `{run_dir}/.complete` | _make_asset after COMPLETED | `COMPLETE_MARKER` |
| Event log + run history | `$DAGSTER_HOME/storage/` (SQLite) | dagster internals | `DAGSTER_HOME` in `.env` |

### 3. SLURM

| What | Path | Who writes | Code |
|------|------|-----------|------|
| Job stdout/stderr | `{SLURM_LOG_DIR}/{name}_%j.{out,err}` | sbatch/OS | `slurm.py` |
| Log rotation (30d) | deletes from SLURM_LOG_DIR | _epilog.sh | cleanup |

### 4. Wandb

| What | Path | Who writes | Config |
|------|------|-----------|--------|
| Run data (metrics, system stats) | `$WANDB_DIR/{project}/{run_id}/` | WandbLogger | `trainer.yaml` logger list |
| Full jsonargparse config | wandb run config | WandbSaveConfigCallback | `cli.py` save_config_callback |

`_preamble.sh` sets `WANDB_DIR=/fs/scratch/PAS1266/wandb` (scratch, 90-day purge). Auth via `~/.netrc`.

### 5. Data / Preprocessing

| What | Path | Who writes | Code |
|------|------|-----------|------|
| Graph cache .pt files | `{lake_root}/cache/v{ver}/{dataset}/` | atomic_save | `preprocessing/utils.py` |
| NFS advisory lock | `{cache_dir}/.lock` | preprocessing/utils.py | flock |
| Staging marker | `{scratch}/kd-gat-data/.staged_marker` | stage_data.sh | rsync guard |
| Node-local staged data | `$TMPDIR/kd-gat-data/` | stage_data.sh | per-job copy |

### 6. Analysis Artifacts

| What | Path | Who writes |
|------|------|-----------|
| Embeddings, CKA, landscape | `{analyzer.output_dir}/` (required CLI param) | Analyzer.run() |

## Execution Order

```
DAGSTER (login node)                    SLURM JOB (compute node)
────────────────────                    ────────────────────────
_make_asset() called
├─ read: ckpt + .complete (skip?)
├─ slurm.submit() ──────────────────►  sbatch allocates node
│   ├─ mkdir SLURM_LOG_DIR              │
│   └─ sbatch --output/--error          │
│                                       ├─ _preamble.sh
│                                       │   ├─ source .env
│                                       │   ├─ stage_data.sh → .staged_marker
│                                       │   └─ mkdir KD_GAT_STAGE_DIR
│                                       │
│                                       ├─ python -m graphids fit
│                                       │   ├─ cli.py pins ModelCheckpoint.dirpath
│                                       │   ├─ SaveConfigCallback → config.yaml
│                                       │   ├─ DataModule.setup()
│                                       │   │   ├─ acquire .lock
│                                       │   │   ├─ torch.save → cache .pt
│                                       │   │   └─ release .lock
│                                       │   ├─ training loop
│                                       │   │   ├─ CSVLogger → metrics.csv
│                                       │   │   ├─ ModelCheckpoint → best_model.ckpt
│                                       │   │   └─ ModelCheckpoint → last.ckpt
│                                       │   └─ trainer.fit() returns
│                                       │
│   slurm.poll() ◄──────────────────   ├─ _epilog.sh (log cleanup)
│   state = COMPLETED                   └─ job exits
│
├─ .complete marker touch
├─ IOManager → sidecar .json
└─ return ckpt_path

dagster SQLite — writes throughout (login node)
analyzer — separate invocation, after all training
```

## Resolved Issues

### RESOLVED: version_0 crash reuse → decoupled

Checkpoints now write to `{run_dir}/checkpoints/` via explicit `dirpath` pin in `cli.py`. CSVLogger version auto-increment is irrelevant — nothing depends on `lightning_logs/version_N/`. No version number in any checkpoint path.

### RESOLVED: last.ckpt resume was dead code

`save_last: true` added to ModelCheckpoint in `trainer.yaml` and `fusion.yaml`. Resume path in `component.py` uses `LAST_CKPT_SUBPATH` constant from `write_paths.yaml`.

### RESOLVED: DAGSTER_HOME in two places

Consolidated to `.env`. `dagster-ui.sh` asserts `DAGSTER_HOME` is set. `__main__.py run` errors if unset. Pending: `dagster-ui.sh` needs `source .env`.

### RESOLVED: MLflow in-repo writes

`MLFLOW_TRACKING_URI` removed from `.env`. All MLflow references cleaned from `.gitignore`, `ci.yml`, `data_loader.py`, and 2 stale skills deleted.

### RESOLVED: test SLURM logs in-repo

`run_tests_slurm.sh` was already deleted.

### RESOLVED: smoke test duplication

`smoke_test()` deleted. `orchestrate/__main__.py` deleted. Single entry point at `graphids/__main__.py`. Training goes through dagster only.

## Open Issues

- `experimentruns` fallback: `LAKE_ROOT` defaults to relative in-repo path when `KD_GAT_LAKE_ROOT` unset. `.env.example` should include it.
- `production` path never used: `run_dir()` always emits `dev/{user}`. Remove claim from docs.
- `dagster-ui.sh` needs `source .env`: script asserts `DAGSTER_HOME` but doesn't source it.

## Env Var → Path Mapping

| Env var | Default | Set in | Controls |
|---------|---------|--------|----------|
| `KD_GAT_LAKE_ROOT` | `"experimentruns"` (relative) | `.env` → `/fs/ess/PAS1266/kd-gat` | All experiment IO |
| `KD_GAT_SLURM_LOG_DIR` | `constants.yaml` → ESS | `.env` → ESS | SLURM stdout/stderr |
| `WANDB_DIR` | (none) | `_preamble.sh` → scratch | wandb run data (dead) |
| `DAGSTER_HOME` | (none — errors if unset) | `.env` → scratch | dagster state |
| `TMPDIR` | (SLURM sets) | OS | per-job local SSD |
| `KD_GAT_STAGE_DIR` | (none) | `_preamble.sh` | staged data on local SSD |
