# Framework Consolidation: Hydra + Lightning (Research-First)

**Date:** 2026-03-20
**Status:** Brainstorm (research-first)
**Prior version:** `hydra-optuna-sweeper.freeform.md` (narrow scope, now superseded)
**Depends on:** `codebase-reduction.md`, `stage-executor-and-launcher.research.md`

---

## Problem (revised)

The original plan was narrow: replace `optuna_sweep.py` with hydra-optuna-sweeper (374 lines → 0). Research revealed a much bigger opportunity: the project already depends on Lightning and Hydra but underutilizes both. The result is ~2,755 lines of custom infrastructure reimplementing features these frameworks already provide.

**Root cause:** Custom layers were built across sessions without checking what the existing dependencies offered. The `graphids/storage/` layer (1,107 lines) duplicates Lightning's CSVLogger + ModelCheckpoint + callbacks. The config wrapper (179 lines) duplicates Hydra's `@hydra.main` entry point. The sweep code (374 lines) duplicates hydra-optuna-sweeper.

**Evidence of duplication:**
- `CSVLogger` is already wired at `trainer_factory.py:284` but a parallel manifest system writes the same metrics
- `ModelCheckpoint` is already wired at `trainer_factory.py:287` but `mapper.save_checkpoint()` duplicates it
- `save_hyperparameters()` is called in DQN modules but not GAT/VGAE — config saving is inconsistent
- `trainer.log_dir` is never used — custom `gateway.resolve()` does the same path resolution

## Goals

1. Use Lightning's full experiment management (CSVLogger, ModelCheckpoint, callbacks, `trainer.log_dir`)
2. Use Hydra-as-framework (`@hydra.main`) to unlock sweeper/launcher plugins + `instantiate()`
3. Eliminate the storage layer (`graphids/storage/`, 1,107 lines)
4. Eliminate the sweep code (`optuna_sweep.py` + `subprocess_utils.py`, 374 lines)
5. Thin remaining infrastructure via `hydra.utils.instantiate()`

## Non-goals

- Multi-node distributed training (Ray Train territory, overkill for single-GPU SLURM)
- MLflow or other external experiment trackers (tried twice, dropped twice — server overhead, heavy deps, NFS friction)
- Metaflow (artifact I/O requires Metaflow to own execution; @slurm is v0.0.4 beta)

---

## Framework inventory: what we HAVE vs what we USE

### Lightning (already a dependency)

| Feature | Used today? | Where | What it could replace |
|---|---|---|---|
| `Trainer` | **Yes** | `trainer_factory.py:316` | — (core training loop) |
| `ModelCheckpoint` | **Yes** | `trainer_factory.py:287` | `mapper.save_checkpoint()` / `mapper.load_checkpoint()` — **duplicate** |
| `EarlyStopping` | **Yes** | `trainer_factory.py:295` | — |
| `DeviceStatsMonitor` | **Yes** | `trainer_factory.py:301` | — |
| `SLURMEnvironment` | **Yes** | `trainer_factory.py:312` | — (auto-requeue on preemption) |
| `CSVLogger` | **Yes** | `trainer_factory.py:284` | `manifest.py` metrics recording — **duplicate** |
| `save_hyperparameters()` | **Partial** | `dqn.py:536,585` only | `mapper.save_config()` — should be on ALL modules |
| `self.log()` | **Partial** | In modules, but metrics also manually extracted | `mapper._extract_training_metrics()` — **duplicate** |
| `trainer.log_dir` | **No** | Never referenced | `gateway.resolve()` path resolution — **replaced** |
| `Callback.on_test_end` | **No** | Not implemented | `mapper.save_embeddings/attention/dqn_policy` — **replaced** |
| `BasePredictionWriter` | **No** | Not implemented | `eval_inference.py` output handling — potential |
| `Tuner.scale_batch_size` | **No** | Not implemented | `batch_sizing.py` (37 lines) — **replaced** |
| `Multiple loggers` | **No** | Single CSVLogger | Free TensorBoard alongside CSV |
| `LightningCLI` | **No** | Not implemented | Alternative to Typer (but conflicts with Hydra — not recommended) |
| `Fabric` | **No** | Not implemented | Lightweight option for non-training tasks |

**Source:** [Lightning CSVLogger docs](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html), [Lightning Callbacks docs](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html), [Lightning ModelCheckpoint docs](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html), [Lightning Trainer docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

### Hydra (already a dependency)

| Feature | Used today? | Where | What it could replace |
|---|---|---|---|
| Compose API | **Yes** | `_hydra_bridge.py:40-51` | — (config resolution core) |
| Config groups | **Yes** | `conf/model/`, `conf/dataset/`, `conf/auxiliary/` | — |
| `oc.env` resolver | **Yes** | `conf/config.yaml:11` | — |
| `@hydra.main` | **No** | — | `cli.py` Typer routing — enables all plugins below |
| `--multirun` | **No** (requires @hydra.main) | — | `optuna_sweep.py` (302 lines) — **replaced** |
| `hydra-optuna-sweeper` | **No** | — | `optuna_sweep.py` + `subprocess_utils.py` (374 lines) — **replaced** |
| `hydra-submitit-launcher` | **No** | — | `slurm.py` for sweep trial submission — **replaced** |
| `hydra.utils.instantiate()` | **No** | — | `trainer_factory.py` callback/optimizer assembly, `registry.py` model dispatch — **thinned** |
| `hydra.run.dir` template | **No** | — | `paths.py` + `gateway.resolve()` lake path layout — **replaced** |
| Structured configs | **No** | — | Could complement Pydantic (not a priority) |
| Custom resolvers | **No** | — | Some `constants.py` lookups — minor |

**Source:** [Hydra Compose API](https://hydra.cc/docs/advanced/compose_api), [Hydra @hydra.main](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/), [Hydra Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper), [Hydra submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher), [Hydra instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/)

### Ray (optional dependency — `ray[default]>=2.49, ray[tune]>=2.49`)

| Component | What it does | Used? | Relevant? |
|---|---|---|---|
| **Ray Core** | Distributed tasks/actors via `@ray.remote` | No | No — single-GPU SLURM jobs don't need distributed primitives |
| **Ray Tune** | HPO with schedulers (ASHA, PBT), search algorithms | Removed in Phase 2 | Replaced by Optuna. Tune has broader scheduler support but more complexity |
| **Ray Train** | Distributed training. Lightning integration via `RayTrainReportCallback`. Persistent storage, checkpoint management, fault tolerance | No | **Partially interesting:** managed output dirs + checkpoint persistence overlap with storage layer. But wants to own training loop via `TorchTrainer`, designed for multi-node |
| **Ray Data** | Distributed data processing (map, filter, batch) | No | No — dataset sizes don't warrant distributed processing |
| **Ray Serve** | Model serving | No | No — HPC, not serving |

**Ray Train's storage model** ([docs.ray.io/en/master/train/user-guides/persistent-storage](https://docs.ray.io/en/master/train/user-guides/persistent-storage)):
```
{storage_path}/{run_name}/
  ├── *_snapshot.json       ← run metadata
  ├── checkpoint_epoch=0/   ← checkpoints
  └── ...
```
Overlaps with storage layer but requires adopting `TorchTrainer` wrapper around Lightning's `Trainer`. Adds complexity for a problem Lightning already solves natively.

**Verdict:** Ray stays as optional dep for future multi-node work. Not part of this consolidation.

---

## Hydra-as-library vs Hydra-as-framework

The project currently uses Hydra as a **library** (Compose API). Switching to **framework** mode (`@hydra.main`) unlocks plugins but changes how the application works.

| | Library mode (current) | Framework mode (proposed) |
|---|---|---|
| **Who calls whom** | You call `compose()`, get DictConfig back | Hydra calls your task function with composed config |
| **Who owns the process** | Your code (Typer, structlog, `mp.set_start_method`) | Hydra (singleton lifecycle, output dirs, logging) |
| **CLI parsing** | Typer + manual override parsing | Hydra parses `sys.argv` directly |
| **Multirun / sweepers** | Not available | Built-in — `--multirun` enables sweeper plugins |
| **Launcher plugins** | Not available | Built-in — submitit, joblib, etc. |
| **Output directories** | Custom (`gateway.resolve()`) | Configurable: `hydra.run.dir` template |
| **Subcommands** | Typer (5 subcommands) | Not supported — separate entry points needed |

**Key evidence:** Hydra docs explicitly state ([hydra.cc/docs/advanced/compose_api](https://hydra.cc/docs/advanced/compose_api)):
> "Avoid using the Compose API in cases where @hydra.main() can be used, as doing so forfeits many of the benefits of Hydra such as Tab completion, **Multirun**, Working directory management, and Logging management."

**Compose API stays for programmatic callers:** `resolve()` is called from tests, notebooks, and `execute_stage()`. These still need Compose API. The change is that `@hydra.main` becomes the primary CLI entry point, Compose API becomes the programmatic/test path. Hydra docs explicitly support this: Compose API is "for notebooks and tests."

**Subcommand handling:** Current `cli.py` has 5 subcommands via Typer. `@hydra.main` doesn't support subcommands — it's one decorator, one task function. Solution: multiple entry points in `pyproject.toml`:
- `graphids-train` → `train.py` (@hydra.main)
- `graphids-sweep` → `sweep.py` (@hydra.main --multirun)
- `graphids-orchestrate` → `orchestrate.py` (thin script, uses Compose API)
- `graphids-lake` → `lake.py` (thin script)
- `graphids-preprocess` → `preprocess.py` (thin script)

Or keep a single `python -m graphids.cli` dispatcher that delegates to @hydra.main for training/sweep paths.

---

## File-by-file replacement plan

### DELETED: `storage/gateway.py` (232 lines)

**What it does:** NFS-safe transport layer. Path resolution (`resolve`, `exists`, `require`, `ensure_dir`), atomic writes (tmpfile + `os.fsync()` + `os.rename()`), advisory locking (`fcntl.flock`), JSON read/write.

**Replaced by:**
- **Path resolution** → `trainer.log_dir` (Lightning manages per-run output dirs) + `hydra.run.dir` template for Hydra-managed output structure
  - Evidence: `trainer.log_dir` is documented as "the directory where the current experiment's logs and artifacts are stored" ([Lightning Trainer docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html))
- **Atomic checkpoint writes** → `ModelCheckpoint` already handles atomic checkpoint saving (Lightning writes to tmp then renames internally)
- **Atomic eval artifact writes** → Not needed. Eval artifacts (embeddings, attention, policy) are write-once at test end. No concurrent writers, no NFS race conditions. Simple `np.savez_compressed()` + `json.dump()` suffices.
- **Advisory locking (`fcntl.flock`)** → Only used for preprocessing graph cache. Moves to `core/preprocessing/_cache.py` as ~20 lines. Not a storage layer concern.
- **JSON read/write** → stdlib `json.load()` / `json.dump()` (no wrapper needed)

**What stays:** ~20 lines of `fcntl` locking in preprocessing (not experiment storage).

### DELETED: `storage/mapper.py` (358 lines)

**What it does:** Domain-aware serialization. 8 artifact types: checkpoints, configs, training results, embeddings, attention weights, DQN policy, CKA matrices, collated cache, pickle, generic JSON/npz.

**Replaced by (method-by-method):**

| Method | Lines | Replacement | Evidence |
|---|---:|---|---|
| `save_checkpoint()` | 7 | **`ModelCheckpoint` callback** (already wired at `trainer_factory.py:287`). `checkpoint_callback.best_model_path` gives the path. Currently duplicated — mapper re-saves what ModelCheckpoint already saved. | [Lightning ModelCheckpoint docs](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html) |
| `load_checkpoint()` | 3 | `torch.load(checkpoint_callback.best_model_path)` — direct load, no wrapper needed | — |
| `save_dqn_checkpoint()` | 7 | Same as `save_checkpoint()` — DQN fusion stage can use ModelCheckpoint or a direct `torch.save()` (3 lines, no wrapper class) | — |
| `save_config()` | 7 | **`save_hyperparameters()`** on all LightningModules. Currently only on DQN (`dqn.py:536,585`). When called, Lightning auto-saves all init args to the checkpoint AND `hparams.yaml` in `trainer.log_dir`. | [Lightning save_hyperparameters docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) |
| `load_config()` | 6 | Read `hparams.yaml` from the run directory, or load from checkpoint dict (`checkpoint["hyper_parameters"]`) | — |
| `save_training_result()` | 16 | **Eliminated.** ModelCheckpoint saves the model. CSVLogger saves metrics. `save_hyperparameters()` saves config. The 3-in-1 wrapper becomes unnecessary. | — |
| `_extract_training_metrics()` | 19 | **`self.log()` in LightningModule** — metrics logged via `self.log("val_loss", loss)` are automatically recorded by CSVLogger. No post-hoc extraction needed. Currently partially used but duplicated by manual extraction. | [Lightning logging docs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) |
| `save_embeddings()` | 25 | **`EvalArtifactCallback.on_test_end()`** — ~10 lines. Cache embeddings during `test_step()` on the LightningModule, save to `trainer.log_dir` in `on_test_end`. | [Lightning Callback docs](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html): "`on_test_end` — called when the test epoch ends" |
| `save_attention()` | 20 | Same callback, ~8 lines. Attention weights cached during eval, written in `on_test_end`. | — |
| `save_dqn_policy()` | 12 | Same callback, ~5 lines. JSON dump of alphas/q-values. | — |
| `save_cka()` | 60 | **~55 lines move to `cka.py`, ~5 lines I/O in callback.** This method is 90% computation (load models, collect representations, compute CKA matrix) and 10% I/O. The computation is domain logic that belongs in a stage file, not the storage layer. | — |
| `save_collated()` / `load_collated()` | 22 | **Stays in `core/preprocessing/_cache.py`** — preprocessing graph cache is not experiment storage. Needs `fsync` + atomic rename for NFS safety with parallel workers. | — |
| `save_pickle()` / `load_pickle()` | 20 | **Stays in `core/preprocessing/`** — vocabulary pickle. Same NFS concern as collated cache. | — |
| `save_npz()` | 7 | Inlined: `np.savez_compressed(path, **data)` — no wrapper needed | — |
| `save_json()` | 4 | Inlined: `json.dump(data, open(path, "w"))` — no wrapper needed | — |
| `_atomic_torch_save()` | 14 | **Eliminated.** ModelCheckpoint handles atomicity for checkpoints. Other saves (eval artifacts) are write-once, no atomicity needed. | — |
| `open_gateway()` | 4 | **Eliminated.** No gateway or mapper to construct. | — |

**Net result:** ~30-line `EvalArtifactCallback` + ~40 lines moved to preprocessing cache. 358 → ~70 lines total, and those 70 aren't in a storage layer.

### DELETED: `storage/manifest.py` (213 lines)

**What it does:** Writes `_manifest.json` per completed run with: identity fields (dataset, model_type, scale, stage, seed), timestamps, git SHA, metrics dict, artifact inventory with SHA-256 checksums. Pydantic models for `ManifestEntry` and `Manifest`.

**Replaced by:**
- **Metrics** → CSVLogger's `metrics.csv` (already being written at `trainer_factory.py:284`). All `self.log()` calls in modules automatically appear here.
- **Config/identity fields** → `save_hyperparameters()` writes `hparams.yaml` in the same directory. Contains all config values including dataset, model_type, scale, seed.
- **Timestamps** → CSVLogger records step/epoch. Hydra's output dir template can include timestamp. Filesystem `mtime` on the metrics.csv.
- **Git SHA** → ~5-line callback: `subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()` written to a `run_info.json` in `on_fit_start`.
- **SHA-256 checksums** → ~15-line callback: compute checksums of checkpoint + artifact files in `on_test_end` or `on_fit_end`, write to `checksums.json`.

**Net result:** ~20-line `RunMetadataCallback` replaces 213-line manifest system.

### DELETED: `storage/catalog.py` (178 lines)

**What it does:** Scans `production/` and `dev/` for `_manifest.json` files, joins with `config.json`, builds a DuckDB database for cross-run queries. Powers the dashboard.

**Replaced by:**
- **Cross-run querying** → DuckDB can read CSVs directly:
  ```sql
  SELECT * FROM read_csv_auto('logs/**/metrics.csv')
  ```
  Or with pandas: `pd.concat([pd.read_csv(f) for f in Path("logs").rglob("metrics.csv")])` — 3 lines.
- **Dashboard data** → `scripts/data/push_experiments_to_hf.py` reads from whatever format the runs are in. Change source from `_manifest.json` glob to `metrics.csv` glob. Same destination (HF Dataset).

**Evidence:** DuckDB's `read_csv_auto` with glob patterns is documented and handles schema inference across multiple files.

**Net result:** 178 lines → 3 lines (inline query) or ~10-line utility function.

### DELETED: `storage/paths.py` (97 lines)

**What it does:** Lake path layout primitives: `lake_run_dir()`, `lake_cache_dir()`, `lake_raw_dir()`, etc. Computes paths like `{lake_root}/{dataset}/{model_type}_{scale}_{stage}/seed_{N}`.

**Replaced by:**
- **Hydra `run.dir` template:**
  ```yaml
  hydra:
    run:
      dir: ${lake_root}/${dataset}/${model}_${stage}/seed_${seed}
    sweep:
      dir: ${lake_root}/sweeps/${dataset}
      subdir: trial_${hydra.job.num}
  ```
  Hydra creates the directory and sets `trainer.log_dir` (via `default_root_dir`) to match.
- **`lake_cache_dir`** → stays as a simple function in `config/paths.py` (~5 lines). Preprocessing cache path isn't per-run; it's per-dataset.
- **`lake_raw_dir`** → same, ~5 lines in `config/paths.py`.

**Net result:** 97 lines → ~10 lines (cache/raw path helpers) + YAML config.

### DELETED: `storage/__init__.py` (29 lines)

Re-exports. Gone with the package.

### DELETED: `pipeline/orchestration/optuna_sweep.py` (302 lines)

**What it does:** Custom Optuna HPO. Three layers of translation: YAML → tuples → `trial.suggest_*()` → CLI command → subprocess → read manifest.

**Replaced by:** `hydra-optuna-sweeper` plugin.

Search spaces move from custom YAML (`config/search_spaces/*.yaml`) to Hydra config:

```yaml
# Current custom format (config/search_spaces/vgae.yaml)
training.lr:
  type: loguniform
  low: 1.0e-4
  high: 1.0e-2
vgae.latent_dim:
  type: choice
  values: [16, 32, 48, 64]

# Hydra sweeper format (conf/config.yaml or conf/sweep/vgae.yaml)
hydra:
  sweeper:
    params:
      training.lr: interval(1e-4, 1e-2)
      vgae.latent_dim: choice(16, 32, 48, 64)
      vgae.dropout: range(0.05, 0.4)
```

Translation is mechanical: `loguniform` → `interval(..., log=true)`, `uniform` → `range(...)`, `choice` → `choice(...)`.

**Evidence:** [hydra.cc/docs/plugins/optuna_sweeper](https://hydra.cc/docs/plugins/optuna_sweeper) — search spaces defined in YAML `hydra.sweeper.params`, sweeper handles `trial.suggest_*()` and trial dispatch.

**Pipeline sweep orchestration** (`run_sweep_pipeline`, `_run_multi_seed_final`, ~66 lines): This is DAG logic, not HPO. Moves to `orchestration/dag.py` or a sweep orchestration script.

**Warm-start** (`_enqueue_warm_start`, ~10 lines): Pre-populate the Optuna SQLite study before launching. ~10-line standalone function, not part of the sweep file.

**CUDA isolation:** The sweeper calls the task function in-process by default. For GPU isolation between trials, use `hydra-submitit-launcher` (local mode):
```yaml
defaults:
  - override hydra/launcher: submitit_local
```
Each trial runs as a separate subprocess via submitit's `LocalExecutor`. Source: [hydra.cc/docs/plugins/submitit_launcher](https://hydra.cc/docs/plugins/submitit_launcher).

**Key constraint:** The sweeper requires `@hydra.main` + `--multirun`. Cannot use with Compose API. This is the forcing function for Hydra-as-framework on the sweep entry point.

### DELETED: `pipeline/subprocess_utils.py` (72 lines)

**What it does:** `build_cli_cmd()` — constructs CLI command strings for subprocess dispatch. Only consumer is `optuna_sweep.py`.

**Replaced by:** Nothing. With the sweeper plugin, each trial IS a Hydra run — no CLI string building. Dies when `optuna_sweep.py` dies.

### DELETED: `config/search_spaces/*.yaml` (3 files, 57 lines)

**What they do:** Custom search space format for the custom sweep code.

**Replaced by:** Hydra sweeper params in `conf/config.yaml` or separate sweep config groups.

### RESTRUCTURED: `cli.py` (171 lines)

**What it does:** Typer CLI with 5 subcommands + Hydra override grammar for training.

**Replaced by:** Multiple `@hydra.main` entry points for training and sweep. Thin scripts for non-Hydra subcommands:

| Current | New | Notes |
|---|---|---|
| `python -m graphids.cli stage=autoencoder model=vgae_large` | `python -m graphids.train stage=autoencoder model=vgae_large` | `@hydra.main` entry point |
| `python -m graphids.cli sweep --stage autoencoder` | `python -m graphids.sweep --multirun stage=autoencoder` | `@hydra.main` + sweeper |
| `python -m graphids.cli orchestrate --dataset hcrl_sa` | `python -m graphids.orchestrate --dataset hcrl_sa` | Thin script (Typer or argparse) |
| `python -m graphids.cli lake --action status` | `python -m graphids.lake --action status` | Thin script |
| `python -m graphids.cli show-config model=vgae_large` | `python -m graphids.train --cfg job model=vgae_large` | Hydra's built-in `--cfg` flag |
| `python -m graphids.cli preprocess --dataset hcrl_sa` | `python -m graphids.preprocess --dataset hcrl_sa` | Thin script |

**Net result:** `cli.py` (171 lines) → `train.py` (~15 lines) + `sweep.py` (~15 lines) + `orchestrate.py` (~30 lines) + `lake.py` (~15 lines) + `preprocess.py` (~10 lines) = ~85 lines total. More files, but each is trivial and single-purpose.

### SIMPLIFIED: `logging.py` (77 lines)

**What it does:** structlog setup with JSON/console renderers, stdlib bridge for Lightning/Hydra logs.

**With Hydra-as-framework:**
- Hydra auto-configures stdlib logging, writes `{app_name}.log` in output dir
- Lightning logs route through stdlib (which Hydra configures)
- **Option A:** Drop structlog, use Hydra's logging. Saves 77 lines. Loses structured JSON events.
- **Option B:** Keep structlog, disable Hydra's logging (`hydra.job_logging=null`). Keeps structured events, ~30 lines simpler (no stdlib bridge needed since Hydra handles it).

**Recommendation:** Option B — keep structlog for structured events but lean on Hydra for stdlib routing. ~30 lines saved.

### THINNED: `pipeline/stages/trainer_factory.py` (330 lines)

**What it does:** Creates Lightning Trainer with callbacks, optimizer, scheduler. Includes `load_model()`, `run_id()`, custom callback assembly.

**Thinned by `hydra.utils.instantiate()`:**

```yaml
# conf/config.yaml — callbacks from YAML, not Python
callbacks:
  checkpoint:
    _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: ${training.monitor_metric}
    mode: ${training.monitor_mode}
    save_top_k: ${training.save_top_k}
    save_weights_only: true
  early_stopping:
    _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: ${training.monitor_metric}
    patience: ${training.patience}
    mode: ${training.monitor_mode}
  device_stats:
    _target_: pytorch_lightning.callbacks.DeviceStatsMonitor
    cpu_stats: false
```

```python
# trainer_factory.py — thin
callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
trainer = pl.Trainer(callbacks=callbacks, logger=csv_logger, ...)
```

**Evidence:** [hydra.cc/docs/advanced/instantiate_objects/overview](https://hydra.cc/docs/advanced/instantiate_objects/overview) — "`_target_` field specifies the Python class or callable to be instantiated." Supports `_recursive_=True` for nested objects, `_partial_=True` for deferred construction.

**Also applies to:**
- **`registry.py` (124 lines)** — model dispatch dict. Could use `_target_: graphids.core.models.gat.GATWithJK` in YAML instead of custom registry.
- **Scheduler dispatch** — currently manual `getattr(torch.optim.lr_scheduler, name)`. With instantiate: `_target_: torch.optim.lr_scheduler.CosineAnnealingLR`.

**Estimated savings:** ~80-100 lines from trainer_factory, ~50 lines from registry. Needs spike to confirm.

### THINNED: `pipeline/stages/batch_sizing.py` (37 lines)

**What it does:** `safety_factor × configured batch_size`.

**Replaced by:** Lightning's `Tuner.scale_batch_size(model, mode="power")` — auto-finds max batch size that fits in GPU memory.

**Evidence:** [Lightning Tuner docs](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) — "automatically tries to find the largest batch size that fits into memory."

**Net result:** 37 lines → 2 lines (`tuner = Tuner(trainer); tuner.scale_batch_size(model)`).

### THINNED: `config/_hydra_bridge.py` (179 lines)

**What it does:** Wraps Hydra Compose API behind `resolve()` and `compose_config()`.

**With @hydra.main:** `compose_config()` (used only by CLI) is eliminated — Hydra handles CLI config composition natively. `resolve()` stays for programmatic callers (tests, notebooks, `execute_stage()`) but simplifies — no longer needs to handle CLI override parsing.

**Estimated:** 179 → ~80 lines (resolve() only, simpler).

### UNCHANGED: Domain logic and core ML

| File | Lines | Why unchanged |
|---|---:|---|
| All `core/` (models, preprocessing) | 3,879 | Pure ML domain — framework choice doesn't affect this |
| `pipeline/stages/evaluation.py` | 373 | Eval orchestration — domain logic |
| `pipeline/stages/eval_inference.py` | 276 | Batched inference via `trainer.predict()` — already Lightning |
| `pipeline/stages/fusion.py` | 208 | DQN/MLP/WeightedAvg fusion — domain logic |
| `pipeline/stages/temporal.py` | 281 | Temporal graph classification — domain logic |
| `pipeline/stages/data_loading.py` | 220 | Dataset loading + caching — domain logic |
| `pipeline/stages/training.py` | 161 | Calls `trainer.fit()` — already Lightning |
| `pipeline/stages/modules.py` | 284 | LightningModules — already Lightning |
| `pipeline/stages/cka.py` | 60 | CKA computation — domain logic (absorbs mapper.save_cka computation) |
| `pipeline/orchestration/dag.py` | 177 | graphlib + submitit dependency chains — no framework does SLURM afterok |
| `pipeline/orchestration/job.py` | 63 | ResourceSpec profiles — domain-specific |
| `pipeline/orchestration/slurm.py` | 70 | submitit executor factory — stays for DAG orchestration |
| `pipeline/executor.py` | 94 | `execute_stage()` — thins (no manual manifest write) but stays |
| `pipeline/validate.py` | 99 | Environment validation — stays |
| `config/schema.py` | 269 | Pydantic validation — complements Hydra |
| `config/constants.py` | 80 | Pipeline topology — stays |
| `config/paths.py` | 184 | EnvironmentSettings + path helpers — partially thinned |

---

## New code required

| File | Lines | Purpose |
|---|---:|---|
| `train.py` | ~15 | `@hydra.main` entry point — calls `execute_stage()` |
| `sweep.py` | ~15 | `@hydra.main` + `--multirun` entry point |
| `orchestrate.py` | ~30 | DAG orchestration CLI (thin, uses Compose API) |
| `lake.py` | ~15 | Lake management CLI |
| `preprocess.py` | ~10 | Preprocessing CLI |
| `EvalArtifactCallback` | ~30 | Saves embeddings, attention, DQN policy in `on_test_end` |
| `RunMetadataCallback` | ~20 | Git SHA, checksums, run info |
| `conf/config.yaml` updates | ~30 | Sweeper config, callback `_target_` configs, `run.dir` template |
| `conf/sweep/*.yaml` | ~40 | Search spaces in Hydra format (3 files) |
| **Total new** | **~205** | |

---

## Ledger

| Action | File(s) | Lines removed | Lines added |
|---|---|---:|---:|
| Delete storage layer | `storage/` (6 files) | -1,107 | 0 |
| Delete sweep code | `optuna_sweep.py`, `subprocess_utils.py` | -374 | 0 |
| Delete search spaces | `config/search_spaces/*.yaml` | -57 | 0 |
| Delete CLI | `cli.py` | -171 | 0 |
| Thin hydra bridge | `_hydra_bridge.py` | -99 | 0 |
| Thin trainer_factory | `trainer_factory.py` (instantiate) | -100 | 0 |
| Thin batch_sizing | `batch_sizing.py` | -35 | 0 |
| Thin logging | `logging.py` | -47 | 0 |
| Thin registry | `registry.py` | -50 | 0 |
| Thin executor | `executor.py` | -30 | 0 |
| Add entry points | `train.py`, `sweep.py`, etc. | 0 | +85 |
| Add callbacks | EvalArtifact + RunMetadata | 0 | +50 |
| Add Hydra config | sweeper, callbacks, run.dir | 0 | +70 |
| Move cache I/O | preprocessing/_cache.py | 0 | +40 |
| **Total** | | **-2,070** | **+245** |
| **Net** | | | **-1,825** |

**Codebase: 9,708 → ~7,883 lines. 19% reduction.**

Most of the remaining code is actual ML (models 1,668 lines, preprocessing 2,211 lines, training stages 2,363 lines) and DAG orchestration (310 lines). The infrastructure-to-ML ratio shifts dramatically.

---

## New dependencies

| Package | Purpose | PyPI downloads/mo |
|---|---|---|
| `hydra-optuna-sweeper` | Optuna sweeper plugin | First-party Hydra plugin |
| `hydra-submitit-launcher` | submitit launcher for CUDA isolation | First-party Hydra plugin |

Both are maintained in the `facebookresearch/hydra` repo. No new external dependencies — Hydra + Optuna + submitit are already in the project.

## Dependencies removed

| Package | Why |
|---|---|
| `ray[default]`, `ray[tune]` (optional) | Leftover from Phase 2. Nothing uses it. |

---

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CSVLogger metrics.csv schema varies across runs | Low | Medium | `self.log()` calls are consistent across modules. DuckDB `read_csv_auto` handles schema inference. |
| Hydra output dir naming doesn't match current lake layout | Low | Medium | `hydra.run.dir` template is fully configurable. Match current convention. |
| Dashboard data source change | Certain | Low | `push_experiments_to_hf.py` changes from manifest glob to CSV glob. Same output format. |

---

## Spike questions — RESOLVED

### Q1: @hydra.main + Compose API coexistence — YES, clean coexistence

`@hydra.main` calls `GlobalHydra.instance().clear()` in its `finally` block (source: `hydra/_internal/utils.py`, end of `_run_hydra()`). After it finishes, the singleton is reset. `resolve()` via Compose API (which also clears before init) works without conflict.

For tests/notebooks, the context manager form is recommended:
```python
with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
    cfg = compose(config_name="config", overrides=["model=vgae_large"])
```

**Gotcha:** `HydraConfig.get()` is only available inside `@hydra.main` execution. Code that reads `HydraConfig.get().runtime.output_dir` will fail in Compose mode. Pass output dir explicitly to `execute_stage()` instead.

Source: [Hydra GlobalHydra source](https://github.com/facebookresearch/hydra/blob/main/hydra/core/global_hydra.py), [Hydra issue #440](https://github.com/facebookresearch/hydra/issues/440)

### Q2: hydra-submitit-launcher local mode — SUBPROCESS, full CUDA isolation

`submitit_local` creates a `submitit.LocalExecutor` which spawns tasks via `subprocess.Popen` — full process isolation, separate Python interpreter, separate CUDA context. It even sets `CUDA_VISIBLE_DEVICES` per task.

**Gotcha:** Don't confuse with `DebugExecutor` (`cluster="debug"`) which runs in-process with zero isolation.

Source: `submitit/local/local.py` — `process = subprocess.Popen(proc_cmd, shell=need_shell, env=env)`

### Q3: CSVLogger + Hydra output dirs — WORKS with explicit config

CSVLogger always creates `{save_dir}/{name}/version_{N}/` by default. To write directly into the Hydra output dir with no nesting:

```python
CSVLogger(save_dir=hydra_output_dir, name="", version="")
```

This produces `{hydra_output_dir}/metrics.csv`. `trainer.log_dir` points to `hydra_output_dir`.

**Gotcha:** `version=None` (default) creates `version_0/`. Must pass `version=""` (empty string, not None).

Source: Lightning CSVLogger source — `log_dir = os.path.join(save_dir, name, version_str)`

### Q4: `instantiate()` with Pydantic — NOT WORTH IT for PipelineConfig

`instantiate()` can call a Pydantic `__init__` with `_convert_="all"`, but nested Pydantic sub-models (VGAEConfig, GATConfig, TrainingConfig) each need `_target_` entries, `_recursive_=True` fights Pydantic's validation model, and frozen models reject post-construction mutation. The current `OmegaConf.to_object()` → `model_validate()` is cleaner.

**Verdict:** Keep `model_validate()` for PipelineConfig. Use `instantiate()` for simple objects only (callbacks, optimizers, schedulers — no nested Pydantic models).

Source: `hydra/_internal/instantiate/_instantiate2.py`, [Hydra issue #1184](https://github.com/facebookresearch/hydra/issues/1184)

### Q5: save_hyperparameters() backward compat — SAFE for all KD-GAT load paths

| Load method | Old checkpoint (no hparams) | Result |
|---|---|---|
| `trainer.fit(ckpt_path=...)` | Hparams not read during resume — only `load_state_dict` called | **Safe** |
| `model.load_state_dict(...)` | Only weights, hparams irrelevant | **Safe** |
| `Model.load_from_checkpoint()` with kwargs | kwargs used as init args | **Safe** |
| `Model.load_from_checkpoint()` no kwargs | Fails if required init params have no defaults | **Not used in KD-GAT** |

KD-GAT uses `trainer.fit(ckpt_path=...)` for resume (confirmed: `training.py`) and `model.load_state_dict()` for eval (confirmed: `trainer_factory.py`). Both safe.

Source: Lightning `checkpoint_connector.py` — `restore_model()` only calls `load_state_dict`, never reads `hyper_parameters` key.

---

## Implementation discipline

The core risk is not technical — it's behavioral. Past sessions have: written custom code instead of using framework features, restored deleted code when tests failed instead of fixing the test, and left stale imports/shims "for safety." This section defines the rules to prevent that.

### Rule 1: Delete-first, add-second

Each phase starts by deleting the old code and its tests, THEN adds the replacement. Never run old and new in parallel "to be safe" — that's how duplicate systems survive.

**Procedure per phase:**
1. `git checkout -b phase-X-description`
2. Delete the target files completely (`git rm`)
3. Run `python -c "import graphids"` — collect every `ImportError` and `AttributeError`
4. Fix each error by wiring the new code (not by restoring old code)
5. Run tests — failures are expected; fix them by updating tests to use new interfaces
6. Commit when green

**If a test fails because it imports deleted code:** The test is testing the old infrastructure, not domain behavior. Delete or rewrite the test. Do NOT restore the deleted code to make the test pass.

### Rule 2: No shims, no re-exports, no backward compat

When a module is deleted, all imports of it must be updated or removed. Never add:
- `from graphids.storage import StorageGateway  # backward compat`
- `gateway = None  # removed, see framework-consolidation`
- `# TODO: remove after migration`

If something imports the deleted module, that something must change. Follow the `ImportError` chain to every caller.

### Rule 3: Tests validate behavior, not infrastructure

Current tests that should be DELETED (they test the infrastructure being removed):
- Any test that imports from `graphids.storage` (gateway, mapper, manifest, catalog)
- Any test that calls `build_cli_cmd()`
- Any test of `optuna_sweep.py` internals

Current tests that should be UPDATED (they test domain behavior via infrastructure):
- Tests that call `resolve()` — still valid, `resolve()` stays
- Tests that check training outputs exist — update to check `trainer.log_dir` paths
- Integration tests that run `execute_stage()` — update expected output locations

New tests to ADD:
- `test_csvlogger_writes_metrics()` — verify `metrics.csv` appears in hydra output dir
- `test_eval_callback_saves_artifacts()` — verify embeddings.npz, attention_weights.npz in `trainer.log_dir`
- `test_hydra_main_composes_config()` — verify `@hydra.main` entry point resolves config correctly
- `test_sweeper_config_valid()` — verify search space YAML parses without error

**Test count will decrease.** That's correct — fewer infrastructure layers = fewer infrastructure tests. Domain tests (model dims, training convergence, eval metrics) are unchanged.

### Rule 4: Each phase has a commit gate

Before merging each phase branch:
1. `python -c "import graphids"` succeeds
2. `python -m pytest tests/ -x` passes (submit to SLURM if needed)
3. `git diff --stat main` shows net negative lines (more deleted than added)
4. No file in `graphids/storage/` exists after Phase C
5. No import of deleted modules exists anywhere in the codebase: `grep -r "from graphids.storage" graphids/` returns nothing

### Rule 5: The plan is the source of truth, not the conversation

Each implementation session must:
1. Read this plan before starting
2. Identify which phase to work on
3. Follow the delete-first procedure
4. Update this plan's status after completing a phase
5. NOT deviate from the plan without updating the plan first

If a session discovers that the plan is wrong (e.g., a framework feature doesn't work as documented), the session must update the plan and stop — not improvise a custom solution.

---

## Implementation order

### Phase A: Lightning experiment management (non-breaking)

**Goal:** Add Lightning's experiment tracking alongside existing storage layer. Verify outputs match.

**Add:**
- `save_hyperparameters()` to VGAEModule, GATModule, CurriculumDataModule (DQN already has it)
- `EvalArtifactCallback` (~30 lines) — saves embeddings, attention, DQN policy in `on_test_end`
- `RunMetadataCallback` (~20 lines) — git SHA, checksums in `on_fit_end`
- Wire CSVLogger with `name=""`, `version=""` to write directly to stage output dir

**Verify:** After a training + eval run, the CSVLogger output (`metrics.csv`, `hparams.yaml`) contains the same data as `_manifest.json`. Eval artifacts appear in `trainer.log_dir`.

**No deletions yet.** This phase proves the new path works before cutting the old one.

### Phase B: Hydra-as-framework + sweep (breaking) — DONE

**What was built:**

- Deleted: `cli.py`, `optuna_sweep.py`, `subprocess_utils.py`, `config/search_spaces/*.yaml`, `compose_config()`, `_CONFIG_GROUPS`
- Deleted: 4 stale SLURM scripts (tune_sweep, sweep_pipeline, smoke_test, benchmark_orchestration)
- Added: `__main__.py` (155 lines) — single dispatcher. `@hydra.main` for training/sweep (same function, `--multirun` flag differentiates). Argparse for orchestrate/lake/preprocess.
- Updated: pyproject.toml (`graphids = "graphids.__main__:main"`), 3 SLURM scripts migrated to Hydra override grammar
- Added: `conf/config.yaml` — `checkpoints` dict with OmegaConf interpolation, `_tier`/`_output_base` private vars, `hydra.run.dir` template

**CLI surface:**
```
python -m graphids stage=autoencoder model=vgae_large dataset=hcrl_sa
python -m graphids --multirun stage=autoencoder model=vgae_large
python -m graphids orchestrate --dataset hcrl_sa
python -m graphids lake --action status
python -m graphids preprocess --dataset hcrl_sa
```

### Phase C: Delete storage layer (breaking)

**Goal:** Remove `graphids/storage/` entirely. All I/O through Lightning + stdlib.

**Delete:**
- `storage/gateway.py` (232 lines)
- `storage/mapper.py` (358 lines)
- `storage/manifest.py` (213 lines)
- `storage/catalog.py` (178 lines)
- `storage/paths.py` (97 lines)
- `storage/__init__.py` (29 lines)

**Move:**
- `fcntl` locking + atomic cache writes → `core/preprocessing/_cache.py` (~40 lines)
- `save_cka` computation → `pipeline/stages/cka.py` (already mostly there)

**Update:**
- `executor.py` — remove manifest write, use CSVLogger outputs
- `trainer_factory.py` — remove `StorageGateway` import, use `trainer.log_dir`
- All `from graphids.storage import ...` → fix or delete
- `pipeline/stages/evaluation.py` — use `EvalArtifactCallback` instead of `mapper.save_*`
- `pipeline/stages/training.py` — remove `mapper.save_training_result()`, ModelCheckpoint handles it

**Delete tests:** Any test importing from `graphids.storage`.

### Phase D: instantiate() + Tuner (thinning)

**Goal:** Use Hydra `instantiate()` for callbacks/schedulers. Use Lightning `Tuner` for batch sizing.

**Thin:**
- `trainer_factory.py` — callback assembly via `instantiate()` (~80-100 lines saved)
- `registry.py` — model dispatch via `_target_` in YAML (~50 lines saved)
- `batch_sizing.py` — replace with `Tuner.scale_batch_size()` (~35 lines saved)

**Add:**
- `_target_` entries in `conf/config.yaml` for callbacks
- Model `_target_` entries in `conf/model/*.yaml`

### Phase E: Dashboard + scripts migration

**Goal:** Update downstream consumers of experiment data.

**Update:**
- `scripts/data/push_experiments_to_hf.py` — read from `metrics.csv` glob instead of `_manifest.json`
- SLURM scripts — reference new entry points
- `PLAN.md`, `.claude/rules/` files — update architecture docs
- Remove `ray[default]`, `ray[tune]` optional deps from pyproject.toml

---

## Phase status

| Phase | Description | Status |
|---|---|---|
| Spike | Q1-Q5 research | **Done** (all resolved, no blockers) |
| A | Lightning experiment management | **Done** — save_hyperparameters on VGAE/GAT, CSVLogger name=""/version="", RunMetadataCallback, EvalArtifactCallback |
| B | Hydra-as-framework + sweep | **Done** — deleted cli.py/optuna_sweep/subprocess_utils/search_spaces/stale scripts. Added `__main__.py` (single dispatcher: @hydra.main for train/sweep, argparse for rest). Deleted `compose_config()`. -651 net lines. |
| C | Delete storage layer | **In progress** — storage/ deleted, partial fixes attempted but deviated from plan. Needs revert + redo per `plans/phase-c-implementation.md`. |
| D | instantiate() + Tuner | Pending (depends on B) |
| E | Dashboard + scripts | Pending (depends on C) |
