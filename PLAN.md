# GraphIDS Session Plan

> Last updated: 2026-04-10 (session 44 — orchestrate refactor + consolidation pass)

## Session 44 follow-up — orchestrate consolidation pass

Critical audit of the just-landed refactor identified 20 hand-rolling,
decomposition, and bloat issues. Worked through them in priority order.

### What changed

- **New**: `graphids/_reflect.py` — single source for `import_class` +
  `filter_kwargs` (lru_cached). Dedupes identical copies that had lived
  in `orchestrate/instantiate.py` and `core/models/factory.py`.
- **`orchestrate/instantiate.py`** — now imports from `_reflect`.
  `_TRAINER_CONFIG_KEYS` is now derived from
  `dataclasses.fields(TrainerConfig)` at module load (no more
  hand-maintained set). `build_trainer` is 4 lines instead of 10
  (dict comprehension + constructor instead of mutating loop).
  `build_callbacks` no longer deepcopies + string-matches
  ModelCheckpoint class_path to inject dirpath. Hoisted all deferred
  imports (`inject_loss_fn`, `validate_config`) to the top.
- **`core/callbacks.py::ModelCheckpoint`** — owns the
  `{default_root_dir}/checkpoints` convention via `_resolve_dirpath`.
  Runtime-patching of `dirpath` from the instantiator is gone;
  callbacks test rewritten to exercise the new contract directly.
- **`orchestrate/config.py`** —
  - Dropped `_coerce_stages` and `_coerce_auxiliaries` field
    validators; Pydantic v2 coerces `list→tuple` and validates tuple
    item types (including nested `KDEntry` from dicts) natively.
  - `ResolvedConfig` collapsed: dropped `paths: PathContext`
    field. Now 5 fields — `rendered`, `validated`, `stage_name`,
    `run_dir`, `ckpt_file`. `paths` was a debugging convenience that
    duplicated state with the explicit run_dir/ckpt_file fields.
  - Added `ResolvedConfig.from_rendered` classmethod (absorbs the old
    `resolve_from_rendered` free function into the type that owns it).
  - `StageConfig.from_recipe` → `StageConfig.for_stage` with explicit
    kwargs (`trainer_overrides`, `stage_overrides`,
    `resource_overrides`) instead of a recipe dict. No more
    `recipe.get("trainer_overrides", {})` dict probing in the constructor.
- **`orchestrate/planning.py`** —
  - Extracted `_resolve_upstream(stage, stage_to_asset)` helper,
    shared between `build_pipeline_stages` and `enumerate_assets` (was
    duplicated inline in both).
  - `build_pipeline_stages` is now a direct path: no more
    TrainingRunConfig round-trip (`to_training_run().model_dump() →
    TrainingRunConfig(**defaults) → .merge({}) → ...`), no more
    synthetic recipe dict wrapper, no more `enumerate_assets` call for
    a single-config pipeline-run.
- **`orchestrate/resolve.py`** — shrunk to one function
  (`resolve_config`). `resolve_from_rendered` moved to
  `ResolvedConfig.from_rendered`.
- **`orchestrate/run.py`** — extracted `_retry(fn, stage, max_retries)`
  helper and moved `_run_one_stage` into a closure inside
  `run_pipeline` that captures dataset/seed/lake_root/user/config (was
  7-arg positional threading). Also uses `analysis_spec_for(ckpt_file,
  ...)` from `core/analysis/runner.py` instead of reconstructing
  `ckpt_file.resolve().parent.parent / 'artifacts'` path arithmetic inline.
- **`core/analysis/runner.py`** — added `analysis_spec_for` helper
  owning the `{run_dir}/artifacts` layout convention.
- **`graphids/_spawn.py`** — dropped `importlib.import_module` in
  favor of plain `import multiprocessing; import torch.multiprocessing`
  at the top. The indirection was a stale workaround.
- **`graphids/cli/_training.py`** — hoisted `dataclasses.replace`
  import to module level. `_prepare` now uses
  `ResolvedConfig.from_rendered` directly (no `resolve.py` detour).
- **`tests/test_instantiate.py`** — `TestCheckpointDirpathPatched` →
  `TestCheckpointDirpathConvention` — tests the
  `ModelCheckpoint._resolve_dirpath` contract directly with a mocked
  trainer instead of peeking at an instantiator-mutated `.dirpath`
  attribute that no longer exists.
- **`orchestrate/__init__.py`** — dropped `resolve_from_rendered`
  re-export; updated layer docstring for the simpler resolve.py.

### Consolidation scorecard

| Issue | Resolution |
|-------|-----------|
| P5 Duplicate `filter_kwargs` in instantiate + factory | → `_reflect.py` single source |
| P6 Duplicate `import_class` in instantiate + factory | → `_reflect.py` single source |
| P7 Runtime dirpath patching on ModelCheckpoint | → callback owns its convention |
| P8 `build_trainer` manual-loop pop on trainer_dict | → dict comprehension |
| P10 `resolve_from_rendered` free function | → `ResolvedConfig.from_rendered` |
| P11 ResolvedConfig.paths duplicates run_dir/ckpt_file | → dropped paths field |
| P12 Inline path arithmetic for analyze output_dir | → `analysis_spec_for` helper |
| P13 Lazy `dataclasses.replace` import in CLI test cmd | → hoisted to module level |
| P14 `importlib.import_module` in `_spawn.py` | → normal imports |
| P17 `build_pipeline_stages` round-trips via fake recipe | → direct path |
| P18 Deferred `inject_loss_fn` / `validate_config` imports | → hoisted |
| P19 Retry loop scope (turned out to be fine, refactored for clarity) | → extracted `_retry` helper |
| P20 Hand-maintained `_TRAINER_CONFIG_KEYS` | → `dataclasses.fields(TrainerConfig)` |
| Dead: `_coerce_stages` / `_coerce_auxiliaries` | Pydantic v2 handles natively |
| Dead: `validated = validate_config(...)` binding in build_run | unused, dropped |

### Verified

- All `graphids.orchestrate.*` + `graphids.cli.*` + `graphids.core.analysis.runner` imports clean.
- `python -m graphids pipeline-run --dry-run --dataset hcrl_sa -O trainer.max_epochs=3` resolves all 3 stages with the override threaded through.
- `pytest --collect-only` — 183 tests collected (up from 182
  pre-consolidation; +1 from splitting the new `TestCheckpointDirpathConvention` into two cases).
- `tests/test_cli_routing_smoke.py` — 5/5 pass.
- Subset of non-jsonnet tests: 23 failed / 15 passed — **identical to
  the post-refactor baseline**, zero new regressions from the consolidation.
- `_reflect.filter_kwargs` verified end-to-end: drops `dataset` kwarg
  for `BanditFusionModule` (which doesn't accept it), keeps accepted
  fields. `lru_cache` warms on first call.

### Post-consolidation layout (line counts)

```
  47  graphids/orchestrate/__init__.py
 387  graphids/orchestrate/config.py       (pure data)
 170  graphids/orchestrate/planning.py     (+ _resolve_upstream helper)
  60  graphids/orchestrate/resolve.py      (one function)
 133  graphids/orchestrate/instantiate.py  (lean, no deepcopy, dataclass fields source)
 108  graphids/orchestrate/stage.py
 128  graphids/orchestrate/run.py          (closure + _retry helper)
  25  graphids/_fs.py
  31  graphids/_spawn.py                   (no importlib workaround)
  53  graphids/_reflect.py                 (new: import_class + filter_kwargs)
  66  graphids/core/analysis/runner.py     (+ analysis_spec_for helper)
  45  graphids/core/models/factory.py      (now uses _reflect)
----
1253 TOTAL
```

## What this session did (2026-04-10, session 44 — orchestrate 6-layer refactor)

Implemented the full refactor described in
`docs/reference/orchestrate-architecture.md`. Every "open question" was
decided per the doc's recommendation: Q1 = move build_model_from_spec
to core/models/factory.py; Q2 = drop `InstantiatedRun.merged`; Q3 = CLI
fit/test go through `resolve_from_rendered`; Q4 = `_spawn.py` as its own
file; Q5 = `core/analysis/runner.py` name; Q6 = `PipelineResult` in
`config.py`.

### Architecture (6 strict layers)

```
Layer 0  orchestrate/config.py      — PipelineConfig, StageConfig (+to_tla_dict),
                                       TrainingRunConfig, KDEntry, ResolvedConfig,
                                       InstantiatedRun, PipelineResult
Layer 1  orchestrate/planning.py    — enumerate_assets, build_pipeline_stages,
                                       resolve_jsonnet_path, expand_recipe_configs
Layer 2  orchestrate/resolve.py     — resolve_config, resolve_from_rendered
Layer 3  orchestrate/instantiate.py — build_run + build_model/datamodule/trainer/…
                                       (flat module, no Instantiator class)
Layer 4  orchestrate/stage.py       — build(resolved), train(artifacts, resolved),
                                       evaluate(artifacts, resolved)
Layer 5  orchestrate/run.py         — run_pipeline, _run_one_stage
Layer 6  cli/{_training,_pipeline}.py
```

### Changes

- **New**: `orchestrate/config.py` (375 lines) — every frozen data type.
  `StageConfig.to_tla_dict` is the single place field names map to
  jsonnet TLA keys.
- **New**: `orchestrate/planning.py` (136 lines) — flattens the old
  `planning/planner.py` + `planning/recipes.py` into one file. Keeps
  `expand_recipe_configs` since tests consume it.
- **Rewrote**: `orchestrate/resolve.py` (89 lines, was 134). `resolve_config`
  is a free function; the private `_build_tla_dict` packer is gone
  (logic lives on `StageConfig.to_tla_dict`). Added
  `resolve_from_rendered(rendered, stage_name)` for the CLI path.
- **Rewrote**: `orchestrate/instantiate.py` (161 lines) — moved from
  top-level `graphids/instantiate.py`. Flat module, no `Instantiator`
  class wrapper. Dropped dead `_init_kwargs` helper. `InstantiatedRun`
  no longer carries `merged` (debugging leftover, Q2).
- **Rewrote**: `orchestrate/stage.py` (108 lines, was 128). `build` /
  `train` / `evaluate` each take `ResolvedConfig` directly — no more
  unpacking `(rendered, validated, run_dir, ckpt_file, stage)` at every
  call site. `wire_file_exporters` moved to the caller so it fires
  once per stage, not twice (train + evaluate). Handles None run_dir
  for CLI smoke invocations.
- **Rewrote**: `orchestrate/run.py` (142 lines, was 262). Only
  `run_pipeline` + `_run_one_stage` live here now. `PipelineConfig` /
  `build_pipeline_stages` / `PipelineResult` moved to Layer 0/1.
  `_ANALYZABLE_MODEL_TYPES` constant moved to `core/analysis/runner.py`.
- **Rewrote**: `orchestrate/__init__.py` — clean re-exports of the
  public API (PipelineConfig, PipelineResult, StageConfig, run_pipeline,
  resolve_config, build_run, etc).
- **New**: `core/analysis/runner.py` — moved from `orchestrate/analyze.py`.
  Exports `ANALYZABLE_MODEL_TYPES` alongside `run_single_analysis` so
  orchestration callers don't maintain a parallel drift-prone constant.
- **New**: `core/models/factory.py` — moved `build_model_from_spec` from
  the old Instantiator class (per Q1: it's a model concern, not an
  orchestration concern). Updated `cli/_slurm.py::probe_budget` import.
- **New**: `graphids/_fs.py` — `touch_marker` only.
- **New**: `graphids/_spawn.py` — `ensure_spawn` only.
- **Deleted**: `graphids/instantiate.py`, `orchestrate/_setup.py`,
  `orchestrate/analyze.py`, `orchestrate/planning/` (subdir).
- **`config/schemas.py`** — moved the stage-family / monitor-mode
  cross-field check from `resolve.py` inline warning into
  `validate_config` as a hard-failing model validator. Derives family
  from `model.class_path` (looking for `.models.fusion`), so it fires
  at render→validate time instead of waiting for resolve. Verified
  against all three stage jsonnets via the `jsonnet` binary.
- **`cli/_training.py`** — rewrote `_prepare` to render →
  `resolve_from_rendered` → `build(resolved)` → `wire_file_exporters`
  once. `fit` / `test` / `validate` / `predict` all go through the same
  resolved-config path as the pipeline driver (Q3).
- **`cli/_pipeline.py`** — updated import path for
  `build_pipeline_stages` (now in `orchestrate.planning`).
- **`tests/test_instantiate.py`** — rewrote to use the new API
  (`build_run` + `filter_kwargs` instead of `Instantiator` +
  `_init_kwargs`). Dropped all `run.merged["..."]` inspection;
  tests now read the rendered dict directly where needed. 13/13 tests
  collect cleanly.
- **`tests/config/test_config.py`** — fixed stale
  `from graphids.orchestrate.planning import KDEntry, TrainingRunConfig`
  import (those types now live in `orchestrate.config`).
- Stale docstring refs to `graphids.instantiate._build_loss` in
  `core/losses/__init__.py`, `core/losses/distillation.py`, and the
  gat/vgae modules → point to `core.losses.build.build_loss` instead.

### Verified on login node

- `python -m graphids --help` — all commands present, no import churn.
- `python -m graphids pipeline-run --dry-run --dataset hcrl_sa`
  resolves all 3 stages through the refactored planner:
  `['autoencoder_ff9f9014', 'supervised_48a8e0b3', 'fusion_40579b22']`.
- `-O trainer.max_epochs=3 -O trainer.fast_dev_run=true` override
  flow works end-to-end.
- `pytest --collect-only -q` — 182 tests collect (was 178 pre-refactor;
  +4 from `tests/test_instantiate.py` restructuring). Only collection
  error is the pre-existing `test_vram_budget.py` stale import
  documented in session 43.
- `tests/test_cli_routing_smoke.py` — 5/5 pass.
- Subset run on `tests/test_cli_routing_smoke.py + tests/test_submit_sh.py +
  tests/orchestrate/ + tests/config/test_config.py`: 23 failed / 15
  passed vs baseline 25 failed / 14 passed (net **+1 pass,
  −2 failures** — refactor introduces zero regressions). All remaining
  failures are pre-existing: `_jsonnet` binding missing on login node,
  `STAGES`/`STAGE_DEPENDENCIES` NameErrors in
  `tests/config/test_config.py::test_no_cycles` (pre-existing from an
  earlier refactor), and `test_kd_teachers.py` (pre-existing).
- `jsonnet configs/stages/{autoencoder,supervised,fusion}.jsonnet |
  validate_config` all pass, including the new hard-failing
  monitor/mode convention validator.
- End-to-end `StageConfig.to_tla_dict` projection verified: pipeline
  path packs all expected TLAs including `vgae_ckpt_path` threading
  for supervised and `fusion_method` for fusion.

### Invariants established

1. **One TLA invariant, one place**: `StageConfig.to_tla_dict` is the
   single site where field names map to jsonnet TLA keys. Adding a new
   TLA is a two-edit change (method + jsonnet signature), not three.
2. **Layered imports**: config.py → planning.py → resolve.py →
   instantiate.py → stage.py → run.py → cli/. Each layer only imports
   from layers below it.
3. **Login-node safety**: `orchestrate/config.py` uses `TYPE_CHECKING`
   for torch imports so planning + CLI imports don't pull torch.
4. **No namespace-spelled-as-class**: `Instantiator.build_run` →
   `build_run`.
5. **Cross-field validation lives at validate_config time**, not at
   resolve time. Bad monitor/mode pairs fail loudly before any torch
   import.

### Still outstanding (carried forward)

- **Single-stage fit on gpudebug** — verify preprocessing + mmap +
  `CANBusSource`/`get_or_build` + new stage primitives + new
  `resolve_from_rendered` CLI path all work end-to-end on GPU.
- **Full pipeline verification on gpudebug** — run
  `scripts/slurm/submit.sh pipeline-run --dataset hcrl_sa` to confirm
  the 3-stage chain composes correctly through the refactored
  `_run_one_stage`.
- **Pre-existing stale test imports** —
  `tests/core/preprocessing/test_vram_budget.py` imports a removed
  `_FALLBACK_BYTES_PER_NODE` symbol. `tests/config/test_config.py`
  uses deleted `STAGES` / `STAGE_DEPENDENCIES` constants.
  `tests/orchestrate/test_kd_teachers.py` uses an old 2-arg
  `enumerate_assets` signature. All pre-existing; should be fixed or
  deleted in a cleanup pass.
- **Layer 2 workflow DB** (from session 43) — build per
  `docs/reference/observability-data-layers.md`.
- **Layer 3 catalog rebuild** (from session 43) — low priority until
  ≥20 completed runs accumulate.

## What this session did (2026-04-10, session 43 — delete Monarch scaffold)

Investigated the two open items from session 42 (`stage.run_stage`
dead-code, `actors.py` disappearance question) and found that the
premises were wrong:

1. **`stage.run_stage` does not exist.** `cli/_training.py` already
   routes `fit`/`test` through `stage.build/train/evaluate`; the
   primitive-sharing story from session 42's Step 4 is already shipped.
   Nothing to rewire.
2. **The entire Monarch path was dead at runtime.** `torchmonarch`
   was never added to `pyproject.toml` and is not installed in the
   venv. `allocate.py` did `from monarch.job import SlurmJob` and
   `from monarch.config import configure` unguarded, so any
   `monarch-run` invocation would ImportError on the first call.
   `actors.py` had a stub fallback for `monarch.actor`, but
   `chain.run_chain` called `actor.X.call_one(...).get()` — plain
   methods don't have `.call_one`, so the fallback path would
   AttributeError anyway. ~600 lines of unreachable plumbing.

Chose option B (delete everything, replace with a plain in-process
loop) instead of installing torchmonarch.

### Changes

- **Deleted**: `graphids/orchestrate/actors.py` (167), `allocate.py` (96),
  `chain.py` (82), `cli/_monarch.py` (113), `scripts/slurm/monarch_python.sh`,
  `docs/reference/3-chain.md` (67).
- **Deleted from `graphids/_slurm.py`**: `patch_clusterscope_for_osc`
  (clusterscope workaround for `SlurmJob`) and `slurm_log_dir`
  (only caller was `allocate.py`). `__all__` updated.
- **Rewrote `graphids/orchestrate/run.py`** — `run_pipeline(config)`
  now loops `ResolvedConfig.resolve → stage.build → stage.train →
  stage.evaluate → run_single_analysis` directly, with per-stage
  retries. No `JobSpec`, no `run_chain`, no actor plumbing.
  `PipelineResult` carries `checkpoints` + `analyzed_assets` +
  `stage_to_asset`. Per-stage analyze (vs. pipeline-level) means
  a partial chain still leaves usable artifacts behind.
- **Rewrote `graphids/orchestrate/analyze.py`** — dropped the
  `analyze(actor, stages, chain, …)` driver (inlined into
  `run.py::_run_one_stage`). Kept `run_single_analysis(spec)` as
  the one public function.
- **Rewrote `graphids/orchestrate/__init__.py`** — removed the
  `available()` monarch probe, updated the module-layout docstring.
- **New `graphids/cli/_pipeline.py`** — `pipeline-run` Typer command
  replaces `monarch-run`. Same options, same dry-run preview, plus
  `--lake-root` and `--max-retries`. Dispatches to `run_pipeline`
  directly (no `JobSpec`).
- **`graphids/__main__.py`** — imports `cli._pipeline` instead of
  `cli._monarch`; header command list updated.
- **`tests/test_cli_routing_smoke.py`** — imports `cli._pipeline`,
  expects `pipeline-run` in the registered command set.
- **`configs/resources/submit_profiles.json`** — added `pipeline-run`
  profile (`gpu` partition, 8 cpus, 40G mem, 6:00:00 wall time).
- **Docstring cleanup**: `_otel.py`, `cli/_training.py`,
  `orchestrate/stage.py`, `orchestrate/_setup.py` — removed
  stale Monarch/`actors.py` references.
- **Docs**: rewrote `docs/reference/orchestration.md` to describe the
  in-process loop; updated `docs/reference/write-paths.md` execution
  order diagram; updated Route B in `docs/reference/config-architecture.md`
  and the key-files table; updated `docs/responsibilities.md`; updated
  `.claude/rules/config-system.md` file layout + running examples;
  updated `CLAUDE.md` CLI table + config resolution paragraph; updated
  `.github/copilot-instructions.md`.

### Verified on login node

- `python -m graphids --help` boots cleanly; `pipeline-run` appears
  under Orchestration.
- `python -m graphids pipeline-run --dry-run --dataset hcrl_sa`
  resolves the full 3-stage chain via `build_pipeline_stages` and
  prints `['autoencoder_ff9f9014', 'supervised_48a8e0b3', 'fusion_40579b22']`.
- `pytest tests/test_cli_routing_smoke.py --collect-only` — 5 tests
  collect cleanly.
- `from graphids.orchestrate.run import PipelineConfig, run_pipeline;
  from graphids.orchestrate.analyze import run_single_analysis;
  from graphids.cli._pipeline import pipeline_run; import graphids.__main__`
  — all clean.

### Session diff

1,491 deletions, 436 insertions across 30 files (git diff stat).
The net includes session-42 leftovers (`cli/_orchestrate.py`,
`orchestrate/ops/*`, `scripts/spike_monarch.py`). The monarch delete
alone is roughly -600 lines.

### Additional design work (session 43 follow-up)

Investigated Parsl and Balsam as potential workflow engines, then
designed the full three-layer observability storage architecture
documented in `docs/reference/observability-data-layers.md` (838 lines):

- **Layer 1** (implemented): `{run_dir}/traces.jsonl` + `metrics.jsonl`
  + checkpoints + markers. Source of truth.
- **Layer 2** (designed, not built): `{lake_root}/workflow.db` SQLite
  with `pipeline_runs` + `stage_attempts` tables. Orchestration state —
  retries, skips, mid-flight rows. Synchronous push on stage enter/exit
  via a new `WorkflowDB` class hooked into `_run_one_stage`. ~200 lines.
- **Layer 3** (designed, old builder deleted): redesigned DuckDB catalog
  at `{lake_root}/catalog/kd_gat.duckdb` with `runs` + `epoch_events` +
  `hyperparams` tables + `leaderboard` and `metrics_timeseries` views.
  Stateless `CREATE OR REPLACE` rebuild via `rebuild-catalog` CLI.
  ~250 lines including tests.

Also decided against adopting Parsl or Balsam as execution engines: Parsl
doesn't pay for itself at the current pipeline scale (3-stage linear
chain, 1 GPU); Balsam's central-service architecture is ALCF-gated and
out of scope for OSC. Notes captured in conversation memory; may revisit
if a sweep driver enters the roadmap.

`docs/reference/observability.md` DuckDB section trimmed to a pointer.

### Still outstanding (carried from session 42)

- **Single-stage fit on gpudebug** — still pending. Submit
  `scripts/slurm/submit.sh` autoencoder smoke on `hcrl_sa` and
  confirm preprocessing + mmap + `CANBusSource`/`get_or_build` +
  stage primitives all work end-to-end on GPU.
- **Full pipeline verification on gpudebug** — after single-stage
  smoke passes, run `scripts/slurm/submit.sh pipeline-run --dataset hcrl_sa`
  to confirm `run_pipeline` composes the 3-stage chain correctly
  end-to-end (now strictly in-process).
- **Pre-existing stale test import** —
  `tests/core/preprocessing/test_vram_budget.py` imports a removed
  `_FALLBACK_BYTES_PER_NODE` symbol. Not caused by this session.
- **Layer 2 workflow DB** — build per `observability-data-layers.md`.
  Priority after gpudebug smoke.
- **Layer 3 catalog rebuild** — build per same doc. Lower priority;
  low value until ≥20 completed runs accumulate.

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
