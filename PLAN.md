# GraphIDS Session Plan

> Last updated: 2026-04-10 (session 40 — Drop PyTorch Lightning Phase 1-5)

## What this session did (2026-04-10, session 40)

Removed PyTorch Lightning dependency from all source code. Replaced with a
custom ~200-line Trainer, callback protocol, ModelCheckpoint, and EarlyStopping.
All 9 model classes now inherit from `nn.Module` instead of `pl.LightningModule`.
Both datamodules are plain classes. OTel callback/logger use the new protocol.
Instantiator builds the custom Trainer. Zero `pytorch_lightning` imports remain
in `graphids/`.

### Changes (Phases 1-4 of `~/plans/drop-pytorch-lightning.md`)

- **New** `graphids/core/callbacks.py` — `CallbackBase`, `ModelCheckpoint`,
  `EarlyStopping`, `TrainingCallback`/`TrainingLogger` protocols (~170 lines)
- **New** `graphids/core/trainer.py` — `Trainer` with `fit/test/validate/predict`,
  AMP, gradient clipping, metric accumulation, `seed_everything` (~310 lines)
- **Modified** `graphids/core/models/base.py` — `GraphModuleBase(nn.Module)`,
  `_capture_hparams` helper, `log()`/`log_dict()` methods, `build_optimizers()`,
  `device` property. `safe_load_checkpoint` uses raw `state_dict` reconstruction.
- **Modified** `graphids/core/models/fusion/base.py` — `FusionModuleBase(nn.Module)`,
  same pattern. RL models keep `automatic_optimization=False`.
- **Modified** 7 model subclasses — `save_hyperparameters()` → `_capture_hparams()`,
  `configure_optimizers()` → `build_optimizers(max_epochs)`,
  `self.manual_backward(loss)` → `loss.backward()` (DQN).
  **Fix:** MLP + WeightedAvg now set `automatic_optimization=True` (were broken
  before — inherited `False` from FusionModuleBase, loss never backpropagated).
- **Modified** `graphids/core/data/datamodule/{graph,fusion}.py` — plain classes,
  `save_hyperparameters()` → `self.hparams = dict(...)`, `self.trainer.*` coupling
  removed. Device for PrefetchLoader set via `_set_device()`.
- **Modified** `graphids/core/monitoring.py` — `OTelTrainingCallback(CallbackBase)`,
  `OTelTrainingLogger` (plain class, no `pl.loggers.Logger` base).
- **Modified** `graphids/core/data/curriculum.py` — `CurriculumEpochCallback(CallbackBase)`.
- **Modified** `graphids/instantiate.py` — builds custom `Trainer` from `TrainerConfig`,
  `seed_everything` from `graphids.core.trainer`.
- **Modified** `configs/_lib/defaults.libsonnet` — `pytorch_lightning.callbacks.*` →
  `graphids.core.callbacks.*`. Fixed CurriculumEpochCallback path
  (`graphids.core.data.sampler` → `graphids.core.data.curriculum`).
- **Modified** `graphids/config/schemas.py` — `_ALLOWED_CLASS_PATH_ROOTS` now
  `("graphids.",)` only.
- **Modified** `~/.claude/hooks/kdgat-convention-check.sh` — removed Lightning
  enforcement blocks.

### Verified on login node

- All 17 modified source files pass `ruff check`
- All module imports succeed with zero `pytorch_lightning` in `sys.modules`
- CLI (`python -m graphids --help`) works
- Jsonnet renders with `graphids.core.callbacks.*` class_paths
- `pytest --collect-only` collects 118 tests (6 errors are pre-existing `_jsonnet` missing)

### Not verified on compute node

- Actual training (`fit` on gpudebug)
- Checkpoint save/load roundtrip
- AMP + gradient clipping behavior
- `rebuild-catalog` + `pipeline-status` against new checkpoint format
- Backward compat with existing Lightning checkpoints in `experimentruns/`

### Tests migrated (Phase 5, same session)

5 test files updated: `test_instantiate.py`, `test_fusion.py`,
`test_vgae.py`, `test_gat.py`, `test_validated_config.py`. Lightning
imports removed. `TestForcedCallbacks` now asserts the new callback
set (ModelCheckpoint, EarlyStopping, OTelTrainingCallback,
CurriculumEpochCallback). `fast_dev_run` tests rewritten to exercise
`training_step` + `backward` directly.

### PyTorch API audit (same session)

Replaced hand-rolled code with PyTorch built-ins:
- `next(self.parameters()).device` → `register_buffer("_device_tracker", torch.empty(0), persistent=False)`
- `cudnn.deterministic`/`benchmark` → `torch.use_deterministic_algorithms(True, warn_only=True)`
- Redundant `torch.cuda.manual_seed_all` removed (`torch.manual_seed` already covers it)
- `GradScaler(enabled=use_amp)` instead of if/else branch (no-op passthrough when disabled)

### `pytorch-lightning` removed from dependencies

`pyproject.toml` — `pytorch-lightning>=2.6.0` replaced with
`torchmetrics>=1.8.0` (needed standalone). Mypy overrides and
`lightning_fabric` filterwarning removed. `uv.lock` regenerated.

## Next session

### Track 1: Compute node validation (Phase 6)

1-epoch smoke test on gpudebug for all 3 stages. Verify checkpoint
save/load. Verify `rebuild-catalog` + `pipeline-status`.

### Track 2: Checkpoint backward compat (deferred)

Existing production checkpoints have Lightning's `pytorch-lightning_version`
key and wrapped state dict format. `safe_load_checkpoint` currently only
reads the new raw format. Needs dual-read path or a one-time migration
script before loading any pre-session-40 checkpoints.

### Known deferred items (from session 39)

- `analyze` command interface: `--tla 'ckpt_path="..."'` (jsonnet TLA)
- Fusion stage absorbs `auxiliaries=[]` and `vgae_ckpt_path=null` as
  ignored TLAs
- Cross-stage trace propagation (pipeline-level spans linking stages)
- DGI model fix (`GraphInfomaxModel` undefined)
- Dagster deletion (`rm -rf orchestrate/dagster/`)

## What session 39 did (2026-04-08)

### Changes

- **Added** `opentelemetry-api`, `opentelemetry-sdk`,
  `opentelemetry-exporter-otlp-proto-http` deps. **Removed** `wandb`.
- **New** `graphids/core/monitoring.py` — `OTelTrainingCallback` (span
  lifecycle, VRAM gauges, batch timing) + `OTelTrainingLogger` (Lightning
  Logger → OTel histograms). ~160 lines.
- **`__main__.py`** — Phase A: TracerProvider, MeterProvider, LoggerProvider,
  optional Wandb Weave OTLP exporter, stdlib logging bridge. Replaces
  `configure_logging()`.
- **`train_entrypoint.py`** — Phase B: `SimpleSpanProcessor` →
  `traces.jsonl`, `PeriodicExportingMetricReader` → `metrics.jsonl` once
  `run_dir` is known.
- **`defaults.libsonnet`** — removed `device_stats`, `resource_profile`,
  `run_record` callbacks. Added `otel` callback + `OTelTrainingLogger`.
- **`log.py`** — stripped to adapter-only (~28 lines). Deleted
  `_JSONFormatter`, `_SlurmFilter`, `configure_logging`.
- **Deleted** `run_record.py`, `finalize.py`, sidecar I/O from `io.py`,
  `_finalize-record` CLI command, wandb patching from `instantiate.py`,
  `RUN_RECORD_FILENAME`, `WANDB_DIR` from `_preamble.sh`.
- **Rewrote** `catalog.py` to read `traces.jsonl` (OTel span schema).
  `status.py` maps `OK/ERROR/UNSET` instead of `completed/failed/started`.
- **Monarch actor** — Phase A in `__init__`, Phase B per-stage via
  `_wire_file_exporters()`.
- **Updated** `docs/reference/observability.md`.

### Not verified on compute node

- `traces.jsonl` / `metrics.jsonl` output from actual training
- Wandb Weave OTLP connectivity from compute node
- `rebuild-catalog` + `pipeline-status` against real trace data

## Next session

### Track 1: SLURM validation of OTel integration

`fast_dev_run` on gpudebug → verify `traces.jsonl` + `metrics.jsonl` exist
and contain expected spans/metrics. Then `rebuild-catalog` + `pipeline-status`.

### Track 2: Production run

Run full training on `hcrl_sa` with production epochs. All 3 stages.

### Track 3: DGI model fix

`GraphInfomaxModel` undefined — probe and training both broken for DGI.
Fix or remove DGI from `VALID_MODEL_TYPES`.

### Track 4: Dagster deletion

`rm -rf orchestrate/dagster/` + remove `[tool.dg]` from pyproject.toml.
Gated on successful Monarch sweep validation.

### Known deferred items

- `analyze` command interface: `--tla 'ckpt_path="..."'` (jsonnet TLA)
- Fusion stage absorbs `auxiliaries=[]` and `vgae_ckpt_path=null` as
  ignored TLAs
- Cross-stage trace propagation (pipeline-level spans linking stages)

## What session 38 did (2026-04-08)

3-stage Monarch pipeline validated end-to-end. Autoencoder resource profile
right-sized. probe-budget fixed and re-validated with training-realistic
VRAM measurement.

### Track 1: 3-stage pipeline — PASSED

Job 46510583 on gpudebug (hcrl_sa, seed 54, 3 epochs). All 3 stages
completed in ~4.5 min: autoencoder (2m37s), supervised (47s), fusion (43s).
Checkpoints at `dev/rf15/hcrl_sa/*/seed_54/checkpoints/best_model.ckpt`.

**Fixes during validation:**

- **`safe_load_checkpoint` loss_fn reconstruction** (`core/models/base.py`):
  VGAEModule/GATModule exclude `loss_fn` from `save_hyperparameters`
  (it's an nn.Module). `load_from_checkpoint` failed because `loss_fn`
  is a required kwarg with no default. Fix: rebuild `loss_fn` from saved
  hparams via `build_loss()` and pass as extra kwarg.

### Track 2: Autoencoder resource profile right-sized

- `job_profiles.json` autoencoder: 20 CPUs/18 workers → 4 CPUs/2 workers.
  Memory auto-derives from `mem_per_cpu` (181G → 36G). Pre-batching
  eliminated the collation bottleneck that required 18 workers.

### Track 2b: probe-budget fixed

Three bugs fixed in `budget_probe.py`:

1. **Missing `family` TLA** — `_expand.jsonnet` needs `family` to select
   the libsonnet. Now looked up via `FAMILY_FOR_MODEL_TYPE`.
2. **Computed import in `_expand.jsonnet`** — jsonnet doesn't allow
   `import (family + '.libsonnet')`. Replaced with static dispatch via
   `libs` object keyed by family name.
3. **Missing `loss_fn`** — `_instantiate_model` now builds via
   `build_loss()`, matching `safe_load_checkpoint`.

**Probe now replicates training VRAM footprint:** `_warmup_training_state`
creates Adam optimizer and runs one fwd+bwd+step before measuring, so
`torch.cuda.mem_get_info()` reflects optimizer state and compile caches.
Impact: <0.3% budget change (GNN optimizer state is tiny vs 16GB VRAM).

### Probe results (job 46511451, V100 16GB, hcrl_sa/hcrl_ch/set_01)

48 data points (4 fractions × 4 models × 3 datasets). DGI failed
(pre-existing `GraphInfomaxModel` undefined). CSV written to
`/fs/ess/PAS1266/kd-gat/reference/budget_calibration.csv`.

### Pre-batch timing analysis

`docs/reference/prebatch-timing.md` — documents the CPU-GPU pipeline with
real numbers. Pre-batching moves collation from per-step (386ms) to one-time
setup. Per-step CPU cost drops to pin_memory (6ms) + H2D queue (10ms),
fully hidden by GPU step time (155ms+). Workers=0 is correct for
pre-batched path; PrefetchLoader overlaps H2D via CUDA streams.

## Key references

Work items live in GitHub issues now, not `docs/backlog/` (deleted
wholesale). Use `gh issue list` or the `/gh` skill.
