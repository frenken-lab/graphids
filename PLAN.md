# GraphIDS Session Plan

> Last updated: 2026-04-10 (session 42 — orchestrate refactor: data layer + decomposition)

## What this session did (2026-04-10, session 42 — orchestrate refactor: data layer + decomposition)

Began `plans/kd-gat-orchestrate-refactor.md` implementation. Step 1
(fix stale docs) was done in a prior session; this session landed
steps 2–5 of the refactor plan. Step 6 (delete actors.py + shrink
monarch.py) is partially done — monarch.py is already slimmed
(242 → ~110 lines), actors.py now a thin endpoint wrapper around
`stage.py` primitives. Full deletion deferred until gpudebug verifies
the chain still runs.

### Changes

- **`graphids/core/data/cache.py`** — now holds `get_or_build`,
  `clear_cache`, and `DatasetState`. The module-level `_REGISTRY`
  dict memoizes `dataset.build()` keyed by `dataset.cache_key` so
  subsequent stages in the same Python process hit the in-memory
  splits instead of remmapping torch tensors. Duck-types the
  Dataset protocol; no PyG/torch imports at module scope.
- **`graphids/core/data/rebuild.py`** — renamed from the old
  `cache.py` (which held `rebuild_caches`). Only caller
  `cli/_data.py` updated.
- **`CANBusSource`** — new frozen dataclass in
  `core/data/datasets/can_bus.py`. Exposes `cache_key: str`,
  `build() -> DatasetState`, and `resolved_lake_root()` helper.
  The body of `build()` is the `_load_datasets` logic pulled out
  of `GraphDataModule` — catalog lookup + train/val + test_subdirs.
- **`GraphDataModule`** — `__init__` now takes a `dataset` instance
  (any object with `cache_key` + `build()`). Dropped
  `dataset: str`, `dataset_cls: str`, `lake_root`, `window_size`,
  `stride`, `val_fraction`, `seed` from the signature — these
  moved into the Dataset instance. `setup()` calls
  `get_or_build(self.dataset)` and assigns splits from the
  returned `DatasetState`. `_load_datasets` + the `importlib`
  plumbing are deleted.
- **`Instantiator.build_block`** — now recurses into nested
  `{class_path, init_args}` dicts inside `init_args`. Lets jsonnet
  compose a `dataset: { class_path: "...CANBusSource", init_args }`
  block inside the datamodule init_args. KD auxiliaries are still
  popped before model instantiation via `inject_loss_fn`, so no
  interference.
- **`configs/stages/autoencoder.jsonnet`** and
  **`supervised.jsonnet`** — `data.init_args.dataset` is now a
  nested `CANBusSource` class_path block. Dropped the
  `window_size`/`stride`/`val_fraction`/`seed`/`dataset` scalars
  from the datamodule init_args — they live on the source now.
- **`PipelineActor`** — removed `_cached_datasets` /
  `_cache_datasets_from` / `_clone_to_cpu` and the direct
  mutation of `run.datamodule._train_ds`. Dataset reuse is now
  transparent via the process-level cache in the datamodule's
  `setup()`. Train-stage exception handler no longer has
  anything to reset.
- **`rebuild.py`** + **`fusion_states.py`** — updated to
  construct a `CANBusSource` and pass it to `GraphDataModule`.

### Verified on login node (non-torch-executing paths)

- `graphids.core.data.cache` + `CANBusSource` import cleanly,
  `cache_key` renders stable strings.
- Both `configs/stages/autoencoder.jsonnet` and `supervised.jsonnet`
  render via `~/.local/bin/jsonnet`.
- `validate_config(rendered)` passes on the new shape
  (pydantic's `ClassPathBlock.init_args: dict[str, Any]` is
  permissive for nested blocks).
- `Instantiator.build_datamodule(rendered)` returns a
  `GraphDataModule` whose `.dataset` is a live `CANBusSource`
  with the expected `cache_key`.
- `tests/core/data/` + `tests/core/preprocessing/test_datasets.py`
  + `tests/core/preprocessing/test_features.py` still collect
  cleanly (21 tests).

### Step 4 — single-stage primitives

- **`graphids/orchestrate/stage.py`** — new module with the atomic
  primitives per the plan: `build(resolved) → InstantiatedRun`,
  `train(artifacts, resolved) → ckpt_path`, `evaluate(artifacts,
  resolved, ckpt) → metrics`, and a `run_stage(resolved,
  force_retrain=False) → StageResult` driver. `build` owns GPU
  state reset; `train` wires file exporters and touches the
  train phase marker; `evaluate` handles test-only (analyze is
  pipeline-level now). Each primitive owns one verb at one level.
- **`PipelineActor`** — rewritten as a thin endpoint wrapper over
  `stage.build / train / evaluate`. Lost all the inline
  `gc.collect()` / `torch.cuda.empty_cache()` / OTel wiring calls
  — they live in `stage.py`. Added a dedicated `analyze_stage`
  endpoint for the pipeline-level analyze driver to dispatch to.
  `eval_stage` no longer touches the analyzer.

### Step 5 — allocation / chain / analyze / run_pipeline split

- **`graphids/orchestrate/allocate.py`** — new. Holds `JobSpec`
  (moved from monarch.py), `build_slurm_job(spec) → SlurmJob`,
  `spawn_actor(job, gpus_per_node, lake_root) → PipelineActor`,
  and a `configure_monarch()` helper. Zero pipeline knowledge.
- **`graphids/orchestrate/chain.py`** — new. `run_chain(actor,
  stages, dataset, seed, max_retries) → ChainResult` is a pure
  loop over `train_stage` then `eval_stage`, decoupled from the
  SlurmJob lifecycle. `ChainResult` carries both asset→ckpt and
  stage→asset maps.
- **`graphids/orchestrate/analyze.py`** — new pipeline-level
  driver. `analyze(actor, stages, chain, dataset, seed) →
  list[str]` iterates over analyzable stages (vgae/dgi/gat) and
  dispatches to the actor's `analyze_stage` endpoint. Lenient on
  failure. Per the plan's design decision #2, this runs *once*
  after `run_chain` returns.
- **`graphids/orchestrate/run.py`** — new top-level driver.
  `run_pipeline(config, job_spec) → PipelineResult` is the only
  module that sees every layer: plan → allocate → spawn → chain
  → analyze → teardown. Composes only its Layer N+1 peers.
- **`graphids/orchestrate/monarch.py`** — slimmed from 242 to
  ~110 lines. Keeps `PipelineConfig` schema + `build_pipeline_stages`
  only; `JobSpec` and `run_chain` are deleted (moved to
  allocate.py / chain.py).
- **`graphids/cli/_monarch.py`** — rewired to call `run_pipeline`.
  Dry-run path now uses `build_pipeline_stages` directly for the
  preview output.
- **`graphids/config/jsonnet.py`** — deferred the `import _jsonnet`
  call into `render_config` so the rest of the orchestrate package
  is importable on login nodes without the C binding installed.
  This unblocked 68 previously-skipped tests at collection time
  (110 → 178 collected).

### Verified on login node

- All of `orchestrate/{allocate,chain,analyze,run,stage,actors,monarch}.py`
  import cleanly with no `_jsonnet` dependency.
- `JobSpec()` constructs with OSC defaults (`account='PAS1266'`).
- `PipelineConfig()` uses the planner defaults (stages =
  autoencoder/supervised/fusion).
- CLI `__main__` + `cli._monarch` import cleanly.
- `pytest --collect-only` — 178 tests collected (was 110 at
  session start; jump is from deferring `_jsonnet`).

### Not verified / next steps

- **Single-stage fit on gpudebug** — still pending (task #6).
  Need to submit `scripts/slurm/submit.sh` autoencoder smoke on
  `hcrl_sa` and confirm preprocessing + mmap + the new
  `CANBusSource`/`get_or_build` path + the stage primitive
  extraction all work end-to-end on GPU.
- **Full chain verification on gpudebug** — after single-stage
  smoke passes, run the 3-stage chain via `monarch-run` to
  confirm `run_pipeline` + `run_chain` + `analyze` compose
  correctly through the Monarch endpoint boundary.
- **Task #11 — finish deletion pass** — `actors.py` can
  potentially disappear if Monarch's `call_one` can dispatch to
  free functions; otherwise leave it as the thin endpoint
  wrapper it already is. `monarch.py`'s remaining ~110 lines
  could fold into `run.py` or `cli/_monarch.py` if desired.
- **Pre-existing stale test imports** — `test_cli_routing_smoke.py`
  imports the deleted `graphids.cli._orchestrate`, and
  `test_vram_budget.py` imports a removed `_FALLBACK_BYTES_PER_NODE`
  symbol. Neither is caused by this session's changes; both should
  be fixed or deleted in a cleanup pass. All other previously-failing
  `_jsonnet` collection errors are resolved by the deferred import.

## What this session did (2026-04-10, session 41)

### core/ audit

- **Replaced `pynvml` direct usage with `torch.cuda` NVML wrappers** in
  `graphids/core/monitoring.py`. Dropped ~16 lines, eliminated an
  undeclared transitive dep, removed the `_nvml_handle` attribute +
  `_shutdown_nvml` lifecycle helper. `torch.cuda.{utilization,
temperature, power_draw, memory_allocated, memory_reserved}` are all
  present in torch 2.8.0+cu128 (verified on login node).
  **TODO on compute node**: verify `torch.cuda.power_draw()` unit — docs
  say W, historical NVML passthrough is mW. Kept `/1000.0` divisor to
  match pre-swap behavior.
- **Deleted `_trainer` write-only dead state** — Trainer assigned
  `model._trainer = self` at 4 sites, `base.py` + `fusion/base.py`
  initialised it to `None`, nothing ever read it. Leftover from the
  Lightning migration (where `self.trainer` was a public API). Removed
  all 6 lines plus the now-unused `TYPE_CHECKING` import of `Trainer`.
- **Fixed `fusion_states.py:57`** — added defensive `.clone()` before
  `.to(device)` per `.claude/rules/critical-constraints.md` (PyG
  `Data.to()` is in-place).

### DGI parity with VGAE/GAT

DGI was trainable (`GraphInfomaxModel` and `DGIModule` existed) but
multiple downstream features silently skipped or crashed on DGI
checkpoints. Restored full parity.

## What this session did (2026-04-10, session 40 — Drop PyTorch Lightning Phase 1-5)

## What this session did (2026-04-10, session 40)

Removed PyTorch Lightning dependency from all source code. Replaced with a
custom ~200-line Trainer, callback protocol, ModelCheckpoint, and EarlyStopping.
All 9 model classes now inherit from `nn.Module` instead of `pl.LightningModule`.
Both datamodules are plain classes. OTel callback/logger use the new protocol.
Instantiator builds the custom Trainer. Zero `pytorch_lightning` imports remain
in `graphids/`.

### Probe results (job 46511451, V100 16GB, hcrl_sa/hcrl_ch/set_01)

48 data points (4 fractions × 4 models × 3 datasets). DGI previously
failed (`GraphInfomaxModel` undefined — now fixed in session 41). CSV
written to `/fs/ess/PAS1266/kd-gat/reference/budget_calibration.csv`.

### Pre-batch timing analysis

`docs/reference/prebatch-timing.md` — documents the CPU-GPU pipeline with
real numbers. Pre-batching moves collation from per-step (386ms) to one-time
setup. Per-step CPU cost drops to pin_memory (6ms) + H2D queue (10ms),
fully hidden by GPU step time (155ms+). Workers=0 is correct for
pre-batched path; PrefetchLoader overlaps H2D via CUDA streams.

## Key references

Work items live in GitHub issues now, not `docs/backlog/` (deleted
wholesale). Use `gh issue list` or the `/gh` skill.
