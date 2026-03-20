# Codebase Reduction Investigation

> Generated: 2026-03-19
> Total: ~10,130 lines in graphids/

## Session handoff (2026-03-19)

### What was done this session

**Code changes (already committed-ready, not yet committed):**
- Deleted `graphids/pipeline/cli.py` (backward-compat shim)
- Deleted `graphids/pipeline/stages/utils.py` (re-export facade, 7 callers migrated to direct imports)
- Deleted `graphids/storage/contracts.py` (StageArtifact etc. — zero usage outside own file)
- Removed backward-compat re-exports from `graphids/config/__init__.py` (contracts + lake paths)
- Removed lake path re-exports from `graphids/config/paths.py`
- Migrated `cli.py` + 3 scripts to import from `graphids.storage.paths`
- Moved `cleanup()` into `data_loading.py`
- Renamed `EntityVocabulary.from_legacy_mapping()` → `from_dict()`
- Updated PLAN.md, ml-debugger agent, experiment-tracking rule

**Investigation (this file):**
- Deep read of ALL files in preprocessing/, storage/, pipeline/stages/, orchestration/
- Initial estimates were wildly wrong (57% reduction) → revised to 32% after reading code
- Preprocessing: PyG InMemoryDataset can't replace our mmap/NFS caching, but the
  adapter/schema/engine abstraction stack is over-engineered for one dataset. Polars
  replaces numpy scatter ops in features. Target: 2,150 → 450 lines.
- Orchestration: submitit + hydra-optuna-sweeper replaces custom SLURM scripts + sweep.
  Dagster stays for state-aware retry. Target: 940 → 270 lines.
- Storage: gateway should NOT be flattened to functions (reverses earlier recommendation).
  Gateway as passed-in class is dependency injection — plain functions are easier to bypass.
  I/O boundary tests needed to prevent direct torch.save/json.dump outside gateway.
- Pipeline stages: already use Lightning correctly. No major rewrite needed.
- Identified I/O leaks: mapper.save_embeddings/save_attention bypass atomic writes,
  optuna_sweep writes directly, api.py skips all CLI wrapping.

### Where we left off

**Last discussion: I/O enforcement and the stage executor pattern.**

Key unresolved design points:

1. **Stage executor** — `_run_single_stage()` in cli.py owns ALL cross-cutting concerns
   (validation, logging, manifests, error recovery, Pipes context). `api.py` bypasses all
   of it. This function should move to `pipeline/` and be the single entry point for CLI,
   API, submitit, and Dagster. Not yet written up in this file.

2. **I/O boundary tests** — like `test_layer_boundaries.py` but for file I/O. Ban
   `torch.save`, `torch.load`, `json.dump`, `open(..., 'w')` etc. outside of
   `storage/mapper.py` and `storage/gateway.py`. The only enforcement that actually works.
   Need to add to this file and implement.

3. **Gateway path privacy** — the gateway should be the ONLY way to get a storage path.
   Currently `lake_run_dir()`, `lake_cache_dir()` etc. are public functions that let any
   module construct paths and bypass the gateway. These should become private to the
   storage layer. Not yet written up.

4. **submitit integration design** — if submitit calls `execute_stage(cfg, stage)` directly
   (Python callable, not CLI subprocess), config flows as a Pydantic object (picklable),
   logging configures inside the executor, I/O goes through gateway. But nothing enforces
   that the callable uses the gateway — hence the need for I/O boundary tests.

### Sections still to write
- Revise section H2 (storage) — gateway stays as class, add I/O boundary test design
- Add section on stage executor extraction (cli.py → pipeline/)
- Add section on path privacy (public path functions → gateway-only)
- Finalize overall target architecture diagram

### Meta-observations from this session
- Claude (me) writes custom code instead of using frameworks. Memories and CLAUDE.md
  don't fix this — only structural enforcement (tests, code structure) works.
- Every estimate in this file was initially wrong and had to be revised after actually
  reading the code. Don't trust initial estimates — read first.
- The investigation file itself is the deliverable. Don't start refactoring from it
  without explicit user approval on each item.

---

## File sizes (descending)

| Lines | File | Layer |
|------:|------|-------|
| 632 | `core/models/dqn.py` | core |
| 395 | `core/preprocessing/_cache.py` | core |
| 389 | `core/preprocessing/_features.py` | core |
| 373 | `pipeline/stages/evaluation.py` | pipeline |
| 358 | `storage/mapper.py` | storage |
| 332 | `core/preprocessing/adapters/_can_bus.py` | core |
| 330 | `pipeline/stages/trainer_factory.py` | pipeline |
| 314 | `core/models/vgae.py` | core |
| 302 | `pipeline/orchestration/optuna_sweep.py` | pipeline |
| 286 | `core/preprocessing/_parallel.py` | core |
| 284 | `pipeline/stages/modules.py` | pipeline |
| 281 | `pipeline/stages/temporal.py` | pipeline |
| 276 | `pipeline/stages/eval_inference.py` | pipeline |
| 271 | `pipeline/orchestration/dagster_defs.py` | pipeline |
| 269 | `config/schema.py` | config |
| 256 | `cli.py` | top |
| 248 | `core/preprocessing/_schema.py` | core |
| 245 | `pipeline/orchestration/slurm_primitives.py` | pipeline |
| 232 | `storage/gateway.py` | storage |
| 220 | `pipeline/stages/data_loading.py` | pipeline |
| 213 | `storage/manifest.py` | storage |
| 208 | `pipeline/stages/fusion.py` | pipeline |
| 188 | `core/preprocessing/_engine.py` | core |
| 184 | `config/paths.py` | config |
| 179 | `config/_hydra_bridge.py` | config |
| 178 | `storage/catalog.py` | storage |
| 175 | `core/preprocessing/_dataset.py` | core |
| 166 | `core/preprocessing/_vocabulary.py` | core |
| 161 | `pipeline/stages/training.py` | pipeline |
| 161 | `core/models/gat.py` | core |
| 137 | `core/preprocessing/__init__.py` | core |
| 131 | `core/models/_utils.py` | core |
| 126 | `core/models/fusion_features.py` | core |
| 125 | `core/models/temporal.py` | core |
| 124 | `core/models/registry.py` | core |
| 123 | `pipeline/orchestration/pipes_slurm.py` | pipeline |
| 120 | `core/preprocessing/_cache_metadata.py` | core |
| 101 | `core/preprocessing/adapters/base.py` | core |
| 99 | `pipeline/validate.py` | pipeline |

Remaining small files (~500 lines total): `__init__.py` modules, `logging.py`, `api.py`, `paths.py`, etc.

## By layer

| Layer | Lines | Files | % |
|-------|------:|------:|--:|
| core/preprocessing | ~2,150 | 11 | 21% |
| pipeline/stages | ~2,130 | 8 | 21% |
| core/models | ~1,490 | 8 | 15% |
| pipeline/orchestration | ~940 | 4 | 9% |
| storage | ~970 | 5 | 10% |
| config | ~730 | 5 | 7% |
| top-level (cli, api, logging) | ~350 | 4 | 3% |
| __init__ + glue | ~380 | ~10 | 4% |

---

## A. Custom code that duplicates library features

### A1. Preprocessing caching vs PyG InMemoryDataset (~500 lines)

**Files:** `_cache.py` (395), `_cache_metadata.py` (120), parts of `_dataset.py` (175)

**What we built:** Custom pickle/mmap graph caching with hash-based invalidation, metadata tracking (graph stats, feature dims), and a `CollatedGraphDataset` wrapper.

**What PyG provides:** `InMemoryDataset` has `process()` + `processed_file_names` with automatic cache invalidation via `pre_filter`/`pre_transform` hash. Handles serialization, loading, and re-processing when code changes.

**Gap analysis:** Our caching adds mmap tensor storage for large datasets (MMAP_TENSOR_LIMIT), NFS-safe atomics, and collated storage (zero-copy `__getitem__`). PyG's `InMemoryDataset` doesn't do mmap or NFS-safe writes. The collated storage is a genuine optimization for our graph sizes.

**Verdict:** Partially replaceable. The mmap + collated storage is custom for a reason. The metadata tracking (`_cache_metadata.py`) and hash invalidation could use PyG's built-in mechanism. ~120 lines removable.

### A2. trainer_factory.py — Lightning Trainer wiring (330 lines)

**File:** `pipeline/stages/trainer_factory.py`

**What we built:** `make_trainer()` assembles Lightning Trainer with callbacks (ModelCheckpoint, EarlyStopping, DeviceStatsMonitor), configures MLflow autolog, handles precision, gradient clipping. `load_model()` loads checkpoints. `prepare_kd()` resolves teacher models. `build_optimizer_dict()` builds optimizer configs. `make_projection()` builds projection layers.

**What Lightning provides:** `Trainer` takes all of these as constructor args. `LightningModule.load_from_checkpoint()` handles model loading. `LightningModule.configure_optimizers()` is the standard place for optimizer config.

**Gap analysis:** `make_trainer()` is essentially a config-to-kwargs translator — it maps our Pydantic config fields to Lightning Trainer args. `load_model()` does more than `load_from_checkpoint` (it routes through our model registry). `prepare_kd()` is domain logic (teacher resolution). `build_optimizer_dict()` could move into the LightningModule.

**Verdict:** `make_trainer()` (~80 lines) and `build_optimizer_dict()` (~30 lines) are pure wiring that could be inlined or moved into the LightningModule. ~110 lines removable. `load_model`, `prepare_kd`, `make_projection` are domain logic, keep.

### A3. storage/catalog.py — DuckDB catalog rebuild (178 lines)

**File:** `storage/catalog.py`

**What we built:** Scans all `_manifest.json` files under lake_root, flattens them into rows, writes a DuckDB catalog. Used by `lake --lake-action status` CLI and `push_experiments_to_hf.py`.

**What already exists:** `push_experiments_to_hf.py` pushes the same data to HF Datasets for the dashboard. The DuckDB catalog is a second query layer over the same manifest data.

**Verdict:** If the HF Dataset push is the primary consumer, the DuckDB catalog is redundant. If local CLI queries are valuable, keep it. Decision needed: **one query layer or two?**

### A4. storage/manifest.py — custom artifact tracking (213 lines)

**File:** `storage/manifest.py`

**What we built:** Writes `_manifest.json` per run with config identity, timestamps, git SHA, SLURM job ID, artifact file list with SHA-256 checksums, and metrics. `verify_manifest()` checks integrity. `verify_all()` scans everything.

**What MLflow/Dagster provide:** MLflow `log_artifact()` + `log_metrics()` tracks the same info. Dagster `MaterializeResult(metadata={...})` captures per-asset metadata.

**Gap analysis:** We deliberately replaced MLflow with this (commit `be996b8`). The manifest system works in fire-and-forget mode without a daemon. It's simpler and NFS-native.

**Verdict:** This was a conscious design choice. Keep unless moving fully to Dagster daemon mode.

### A5. validate.py — pre-flight checks (99 lines)

**File:** `pipeline/validate.py`

**What we built:** `validate(cfg, stage)` checks: dataset exists on disk, KD teacher checkpoint exists, prerequisite stage outputs exist (via StorageGateway), evaluation has at least one model.

**What Pydantic provides:** Field-level type/range validation. But NOT filesystem existence checks or cross-stage dependency checks.

**Gap analysis:** These are runtime environment checks, not schema validation. Pydantic can't know if a file exists on NFS. This is genuinely different from config validation.

**Verdict:** Keep. `validate_datasets()` (lines 22-38) is dead — delete that one function.

---

## B. Possibly dead / experimental stages

### B1. Temporal stage (281 + 125 = 406 lines)

**Files:** `pipeline/stages/temporal.py` (281), `core/models/temporal.py` (125)

**Evidence of use:**
- Listed in CLI stages (valid stage name)
- Has a `TemporalConfig` in schema.py
- No `conf/model/temporal_*.yaml` config files exist
- No temporal entries in `pipeline.yaml` stage dependencies
- No temporal entries in any SLURM scripts
- No temporal runs in experimentruns/
- `run_temporal_stage()` is called from `cli.py` dispatch

**Verdict:** Wired up but never run. 406 lines of experimental code. **Decision: keep as future work or delete?**

### B2. Optuna sweep (302 lines)

**File:** `pipeline/orchestration/optuna_sweep.py`

**Evidence of use:**
- Wired into CLI (`tune` and `sweep` commands)
- Has `config/search_spaces/*.yaml` files
- `load_best_config()` (line 223) is dead
- No evidence of sweep runs in experimentruns/

**Verdict:** Infrastructure is built but may not have been run yet. Probably keep (it's on the roadmap). Delete `load_best_config()`.

---

## C. Dead code confirmed

| Lines | Location | What | Status |
|------:|----------|------|--------|
| 130 | `storage/contracts.py` | StageArtifact etc. | **DELETED this session** |
| 17 | `validate.py:22-38` | `validate_datasets()` | Dead, delete |
| 15 | `optuna_sweep.py:223+` | `load_best_config()` | Dead, delete |
| 2 | `_protocols.py:25` | `StageMetrics` TypedDict | Dead, delete |

---

## D. Reduction opportunities summary

| Item | Lines | Action | Confidence |
|------|------:|--------|-----------|
| Temporal stage (B1) | ~406 | Delete if not planned | Needs decision |
| Preprocessing cache metadata (A1) | ~120 | Replace with PyG cache hash | Medium |
| trainer_factory wiring (A2) | ~110 | Inline into LightningModule | High |
| DuckDB catalog (A3) | ~178 | Remove if HF push is sufficient | Needs decision |
| Dead code (C) | ~34 | Delete | Confirmed |
| **Total potential** | **~850** | | |

This would bring graphids from ~10,130 to ~9,280 lines (8% reduction) from direct cuts.

## Decisions

- **Temporal stage:** KEEP. Move to a separate package/plugin so it's out of the main code path but not deleted. Do NOT delete temporal code — it's future work.
- **DuckDB catalog:** Unknown if needed — never got that far in the pipeline. Defer decision until pipeline runs end-to-end.
- **Dagster:** KEEP but simplify with submitit under the hood. Provides state-aware retry + partition tracking that would cost ~50 lines to rebuild from scratch anyway. Drop the bash script generation, keep the asset DAG.
- **Do NOT delete anything from this file without explicit user approval.** This is an investigation doc, not a task list.

---

## K. Preventing custom code drift

### The problem

Claude writes custom code for each problem instead of using the configured framework.
The user then has to abstract out of the custom code. This is why 10k lines exist for
a 3-model pipeline.

Examples from this codebase:
- PyG has `InMemoryDataset` caching → Claude wrote `_cache.py` (395 lines)
- `pd.factorize()` does vocabulary encoding → Claude wrote `_vocabulary.py` (166 lines)
- Lightning `Trainer` takes callbacks as args → Claude wrote `make_trainer()` factory
- `torch.save(path)` saves checkpoints → Claude wrote `StorageGateway` + `ArtifactMapper`
- Optuna has built-in Hydra integration → Claude wrote `optuna_sweep.py` (302 lines)
- submitit submits Python to SLURM → Claude wrote bash script string builder (72 lines)

### The fix: framework-first project structure

Make the project structure enforce tool usage so custom code is structurally difficult:

| Concern | Framework | How to enforce |
|---------|-----------|----------------|
| **Data loading** | PyG `InMemoryDataset` | All datasets subclass it. `process()` is the only entry point. |
| **Training** | Lightning `Trainer` + `LightningModule` | No training code outside `training_step()`/`validation_step()`. |
| **Config** | Hydra Compose + Pydantic | All config flows through `resolve()`. No `os.environ.get()`. |
| **CLI** | Hydra or Typer | One entry point, framework handles dispatch. |
| **HPO** | hydra-optuna-sweeper | Search spaces in YAML. No Python sweep code. |
| **SLURM** | submitit | `executor.submit(fn, args)`. No bash script generation. |
| **Orchestration** | Dagster assets | DAG defined declaratively. Submitit handles submission. |
| **Logging** | structlog | Already enforced. ✅ |
| **Metrics** | torchmetrics | Already used. ✅ |
| **I/O** | `torch.save`/`torch.load` + `atomic_write()` util | One 30-line utility for NFS safety. No mapper/gateway classes. |
| **Experiment tracking** | Manifest files (keep) or MLflow | Already decided. ✅ |

### How to enforce in practice

1. **No wrapper classes for simple operations.** If the operation is `torch.save(obj, path)`,
   do that. Don't create a `Mapper.save_checkpoint()` method that does the same thing with
   extra indirection.

2. **Subclass, don't wrap.** If PyG has `InMemoryDataset`, subclass it. Don't write a
   parallel caching layer and a dataset wrapper and a schema contract on top.

3. **Configure, don't code.** If Hydra can pass a parameter, put it in YAML. Don't write
   Python code to read it from env vars and thread it through constructors.

4. **One entry point per concern.** Training = `trainer.fit()`. Data = `dataset.process()`.
   HPO = `hydra --multirun`. SLURM = `executor.submit()`. If code exists outside these
   entry points doing the same thing, it's drift.

5. **Test against drift.** `test_layer_boundaries.py` already enforces import rules. Add
   similar tests that catch custom code where framework code should be used (e.g., no
   `subprocess.run(["sbatch", ...])` outside submitit, no `torch.save` outside the
   atomic_write utility).

---

## F. Imperative code that should be declarative

### F1. sbatch script generation — string-building (245 lines)

**File:** `pipeline/orchestration/slurm_primitives.py`

`generate_sbatch_script()` (lines 114-185) imperatively builds an sbatch script as a list of strings: appending `#SBATCH` lines one by one, conditionally adding GPU/dependency/exclude directives, then shell commands.

**Declarative alternative:** `submitit` (Facebook Research, ~4k stars) or `simple-slurm` provide Python-native SLURM submission. `submitit` directly launches Python callables on SLURM without generating bash scripts at all — no string templating, no preamble sourcing. Hydra has a `hydra-submitit-launcher` plugin that integrates directly.

With submitit: `executor = submitit.SlurmExecutor(); executor.update_parameters(mem_gb=32, gpus_per_node=1, ...); job = executor.submit(train_fn, cfg)`. The entire `generate_sbatch_script` + `write_script_file` + `submit_sbatch` chain (~100 lines) collapses to ~10 lines.

**Gap:** Our preamble (`_preamble.sh`) loads modules and activates venv. submitit handles this via `setup` commands. The SIGUSR1 handling for Lightning auto-requeue would need testing with submitit's signal forwarding.

**Verdict:** High-value replacement. ~100 lines of imperative string building → ~10 lines of declarative API. Needs spike to verify preamble/signal compat on OSC.

### F2. Trainer assembly — imperative kwargs wiring (65 lines)

**File:** `pipeline/stages/trainer_factory.py:266-330`

`make_trainer()` imperatively constructs callbacks, conditionally adds SLURMEnvironment plugin, sets up CSVLogger, and passes ~15 kwargs to `pl.Trainer()`.

**Declarative alternative:** Lightning CLI (`LightningCLI`) reads trainer config from YAML directly. All callbacks, loggers, plugins, and trainer kwargs can live in a YAML file under `conf/`. No Python wiring needed. `LightningCLI(MyModule, MyDataModule, args=["fit", "--config", "trainer.yaml"])`.

Alternatively, keep `make_trainer()` but drive it from a `trainer:` section in the Hydra config (already partially done — `cfg.training` has most fields). The remaining imperative parts are: callback instantiation (3 objects), plugin selection (SLURM check), logger creation. These could be Hydra `_target_` instantiations.

**Verdict:** Medium value. 65 lines → YAML config. But LightningCLI would be a bigger refactor since we use Hydra for config (two config systems competing). More practical: move callback definitions to YAML using `hydra.utils.instantiate()`.

### F3. Optimizer/scheduler dispatch — if/elif chain (30 lines)

**File:** `pipeline/stages/trainer_factory.py:233-263`

`build_optimizer_dict()` is an if/elif chain on `cfg.training.scheduler_type` to pick between CosineAnnealingLR, StepLR, ReduceLROnPlateau.

**Declarative alternative:** This belongs in `LightningModule.configure_optimizers()` (which is where Lightning expects it). The scheduler selection could be a `_target_` in the training config YAML, instantiated via `hydra.utils.instantiate()`. Or use `torch.optim.lr_scheduler` registry: `getattr(torch.optim.lr_scheduler, cfg.training.scheduler_type)(**params)`.

**Verdict:** Low-hanging fruit. The if/elif → `getattr` pattern is ~5 lines vs 30.

### F4. Model registry — hand-rolled dict + factory functions (124 lines)

**File:** `core/models/registry.py`

Hand-built registry with `register()`, `get()`, factory wrappers that lazy-import model classes. Each model needs a `_*_factory` wrapper just to do `ModelClass.from_config(cfg, num_ids, in_ch)`.

**Declarative alternative:** Hydra's `instantiate()` with `_target_` in YAML. Each model config YAML would have `_target_: graphids.core.models.vgae.GraphAutoencoderNeighborhood` and Hydra instantiates it directly. No registry, no factory wrappers. The fusion feature extractor mapping could be a simple dict literal.

Alternatively: a class decorator `@register("vgae")` on each model class, eliminating the separate registration block.

**Verdict:** Medium value. 124 lines → ~30 lines (dict literal for extractors + Hydra instantiate). But the registry also owns fusion feature layout (offsets, dims), which is domain logic that needs to stay somewhere.

### F5. Optuna search space parsing — YAML → tuples → suggest_* (60 lines)

**File:** `pipeline/orchestration/optuna_sweep.py:43-100`

`_load_search_spaces()` parses YAML into `(type, low, high)` tuples. `_suggest_params()` maps those tuples to `trial.suggest_float()`, `trial.suggest_categorical()`, etc. Two-step translation.

**Declarative alternative:** Optuna has `optuna.distributions.FloatDistribution`, `CategoricalDistribution` etc. which can be loaded directly from YAML. Or use Hydra's Optuna sweeper plugin (`hydra-optuna-sweeper`) which reads search spaces from Hydra config and runs trials — zero custom code.

**Verdict:** If using Hydra for config already, `hydra-optuna-sweeper` eliminates the entire sweep file (302 lines → 0 + a YAML config section). Biggest single win.

### F6. Resource profile parsing — nested dict iteration (25 lines)

**File:** `pipeline/orchestration/slurm_primitives.py:59-73`

`_parse_resource_profiles()` iterates 3 nested levels of `resources.yaml` to build a `(model, scale, stage) → ResourceSpec` dict. `ResourceSpec.from_yaml()` manually parses memory/walltime strings.

**Declarative alternative:** Pydantic `field_validator` on ResourceSpec to accept `"32G"` and `"4:00:00"` directly. Then `resources.yaml` can be loaded with `pydantic.TypeAdapter(dict[tuple[str,str,str], ResourceSpec]).validate_python(data)` — no manual parsing loop.

**Verdict:** Low-hanging fruit. ~25 lines → ~10 lines with Pydantic validators.

### F7. preprocessing/_engine.py — imperative pipeline orchestration (188 lines)

**File:** `core/preprocessing/_engine.py`

Orchestrates preprocessing steps imperatively: load raw data → build vocabulary → construct features → build graphs → cache. Each step is a function call in sequence with manual progress logging.

**Declarative alternative:** This is essentially a small DAG (vocabulary depends on raw data, features depend on vocabulary, graphs depend on features). Could be Dagster assets, or even a simple declarative pipeline definition. But this runs inside a SLURM job, not under Dagster.

**Verdict:** Low value for the effort. The sequential nature is genuine — these steps have real data dependencies. Making it "declarative" would add abstraction without reducing code. Keep.

---

## G. Imperative reduction summary

| Item | Lines | Replacement | Effort |
|------|------:|-------------|--------|
| F5. Optuna sweep → hydra-optuna-sweeper | ~302 | Plugin + YAML config | Medium (verify plugin works with our Hydra setup) |
| F1. sbatch generation → submitit | ~100 | `submitit.SlurmExecutor` | Medium (spike needed for OSC preamble/signals) |
| F3. Scheduler dispatch → getattr | ~25 | `getattr(torch.optim.lr_scheduler, name)` | Trivial |
| F6. Resource parsing → Pydantic validators | ~25 | `field_validator` on ResourceSpec | Trivial |
| F4. Model registry → Hydra instantiate | ~90 | `_target_` in YAML | Medium (fusion layout logic stays) |
| F2. Trainer assembly → YAML config | ~65 | Hydra `instantiate()` for callbacks | Medium |
| **Total** | **~607** | | |

Combined with Section D (~850 lines), total addressable: **~1,450 lines (14% of codebase)**.

---

## E. Root cause

I (Claude) wrote custom solutions for each problem across many sessions instead of using library APIs from the start. The result is 10k lines of custom infrastructure wrapping a 3-model research pipeline. The fix isn't trimming dead code — it's replacing hand-rolled layers with the libraries already in the dependency list.

## H. What each layer should look like (library-first rewrite targets)

### H1. Preprocessing (2,150 → ~800 lines) 🔄 RE-REVISED

**First estimate:** Replace with PyG InMemoryDataset, save 1,450 lines. Wrong about scope.
**Second estimate:** Almost all domain-specific, save only 100 lines. Wrong about abstractions.

**Third (honest) look — the code is domain-specific, but the ARCHITECTURE is wrong:**

The current design forces every dataset through a CAN-bus-shaped pipeline:
adapter ABC (5 methods) → IR schema contract → sliding-window engine → features → cache.
This assumes all future datasets share the same IR columns, the same windowing strategy,
and the same engine. They won't:

| What changes | CAN bus | Network flow (CICIDS) | Water treatment (SWaT) |
|---|---|---|---|
| **Nodes** | Arbitration IDs | IP addresses | Sensors |
| **Edges** | Temporal adjacency (shift-1) | Flows between IPs | Physical connections |
| **Features** | Payload bytes, entropy | Packet size, duration, flags | Sensor readings |
| **Labels** | Directory name convention | Labeled flows | Timestamped annotations |
| **Windowing** | Message-count windows | Flow-based grouping | Time-based intervals |

A water treatment dataset would fight the adapter ABC and IR schema more than benefit
from them. This is how PyG's own built-in datasets work (Cora, PPI, QM9) — each has
independent processing with shared utility functions, no adapter pattern.

**Target architecture:**

```
core/preprocessing/
    datasets/
        can_bus.py      # CANBusDataset(InMemoryDataset) — owns its entire pipeline
        water.py        # future: WaterDataset(InMemoryDataset) — independent
        network.py      # future: NetworkFlowDataset(InMemoryDataset) — independent
    utils.py            # shared: sliding_window(), normalize(), vocab_from_column()
    features.py         # feature computation functions (each dataset picks what it needs)
```

Each dataset is a self-contained `InMemoryDataset` subclass. When adding a new domain,
you write one file — no need to understand adapter ABCs, IR schemas, or engine internals.

**What stays as domain logic inside `can_bus.py`:**
- Hex parsing, temporal adjacency, attack type inference (~150 lines from `_can_bus.py`)
- 26-D node + 11-D edge feature computation (~300 lines from `_features.py`)
- Sliding window graph construction (~80 lines of core logic from `_engine.py`)
- `process()` method ties it together

**What becomes shared utilities in `utils.py`:**
- `sliding_window(df, window_size, stride)` (~20 lines from `_engine.py`)
- `vocab_from_column(series)` → `pd.factorize()` + OOV (~10 lines, replaces 166-line `_vocabulary.py`)
- Feature normalization helpers (~20 lines)

**What gets deleted (abstractions that don't earn their weight for multi-domain):**
- `adapters/base.py` (101) — ABC with 5 methods, only one implementation
- `adapters/_can_bus.py` (332) → merged into `datasets/can_bus.py`
- `_schema.py` (248) — IR column contract between one adapter and one engine
- `_vocabulary.py` (166) → `pd.factorize()` + OOV in ~10 lines
- `_engine.py` (188) → window logic inlined in dataset, shared helper for sliding
- `_cache.py` (395) → PyG InMemoryDataset handles caching (see mmap note below)
- `_cache_metadata.py` (120) → graph stats computed lazily from loaded dataset (~10 lines)
- `_parallel.py` (286) → preprocessing runs once and caches. If sequential is too slow, add `num_workers` to `process()` later.
- `_dataset.py` (175) → `InMemoryDataset` subclass with mmap override (see below)

**mmap concern:** PyG's `InMemoryDataset.load()` doesn't pass `mmap=True`. Fix: override
`load()` in the subclass to call `torch.load(path, mmap=True)`. Also override `get()` to
skip the `copy.copy()`. ~10 lines of overrides, not a blocker.

**NFS atomic write concern:** PyG's `save()` isn't NFS-safe. Fix: override `save()` to use
tmpfile + fsync + rename. ~15 lines. Or call the existing `atomic_write()` utility from
the storage layer.

**NFS locking concern:** Multiple SLURM jobs hitting `process()` concurrently. Fix: wrap
`process()` with `fcntl.flock` on a lockfile. ~10 lines. Currently done in `_cache.py`
with 395 lines of orchestration.

**Savings: ~1,350 lines** (2,150 → ~800). The domain logic survives (~550 lines of
features + CAN bus parsing + windowing). The abstractions and orchestration go away.

### H1a. Preprocessing — tool alternatives investigated

**Question:** Is there a software tool that handles this pipeline, or is custom code necessary?

**Verdict:** The graph construction and feature SELECTION are genuinely custom (research
design decisions). But the feature IMPLEMENTATION can be simplified with Polars.

#### Tools checked

| Tool | What it does | Applicable? |
|------|-------------|-------------|
| **Polars** | Fast DataFrame groupby-agg (Rust backend) | **YES** — replaces numpy scatter ops in `_features.py` |
| **Dask** | Parallel DataFrames, larger-than-memory | No — our CSVs fit in memory, and we cache results |
| **PyG Temporal** (TGN) | Temporal graph networks | No — expects event stream format, not sliding windows |
| **PyG transforms** | `KNNGraph`, `RadiusGraph`, etc. | No — build graphs from point clouds, not tabular events |
| **DGL `from_pandas()`** | Create graph from edge DataFrame | Partial — handles edge list, not windowing or features |
| **torch-frame** | PyG tabular learning | No — tabular features for GNNs, doesn't construct graphs |
| **StellarGraph** | Graph ML library | Discontinued |

#### Polars opportunity in `_features.py`

The 389-line feature file is mostly hand-rolled `np.add.at` scatter operations to compute
grouped statistics per node. This is exactly what `groupby().agg()` does declaratively.

```python
# CURRENT (~50 lines per feature group):
count = np.zeros(num_nodes, dtype=np.float32)
np.add.at(count, node_ids, 1)
byte_sum = np.zeros((num_nodes, 8), dtype=np.float32)
np.add.at(byte_sum, node_ids, byte_values)
byte_mean = byte_sum / np.maximum(count[:, None], 1)
byte_sq_sum = np.zeros((num_nodes, 8), dtype=np.float32)
np.add.at(byte_sq_sum, node_ids, byte_values ** 2)
byte_std = np.sqrt(byte_sq_sum / np.maximum(count[:, None], 1) - byte_mean ** 2)
# ... repeat for min, max, range, entropy, skewness, kurtosis, change_rate ...

# POLARS equivalent (~15 lines for all byte stats):
node_features = window_df.group_by("node_id").agg([
    *[pl.col(f"byte_{i}").mean().alias(f"byte_{i}_mean") for i in range(8)],
    *[pl.col(f"byte_{i}").std().alias(f"byte_{i}_std") for i in range(8)],
    *[pl.col(f"byte_{i}").min().alias(f"byte_{i}_min") for i in range(8)],
    *[pl.col(f"byte_{i}").max().alias(f"byte_{i}_max") for i in range(8)],
    pl.col("payload").map_elements(shannon_entropy).alias("entropy"),
    # skewness, kurtosis via pl.col().skew(), pl.col().kurtosis()
])
```

Polars has built-in `.skew()` and `.kurtosis()` on grouped columns — eliminates the
manual central-moment accumulation (~60 lines of scatter code). Also handles the hex
byte parsing natively via `str.slice()` + `str.to_integer(base=16)`.

**Impact on `_features.py`:** ~389 lines → ~150 lines. The feature CHOICES stay the same,
the implementation becomes declarative groupby expressions instead of imperative scatter
arrays. Polars is also 5-10x faster than pandas+numpy for grouped aggregations.

**Edge features** (11-D) are similar — inter-arrival time stats, frequency, bidirectionality
are all groupby-agg operations on the edge DataFrame.

**What Polars does NOT replace:**
- Clustering coefficient — still needs networkx or PyG (graph structure, not tabular)
- Graph construction (node re-indexing, edge index building) — graph ops, not DataFrame ops
- Sliding window strategy — domain-specific iteration

#### Revised preprocessing target with Polars

```
core/preprocessing/
    datasets/
        can_bus.py      # CANBusDataset(InMemoryDataset) — ~250 lines
                        #   process(): read CSVs, parse hex, slide windows, build graphs
                        #   uses Polars for feature computation
    utils.py            # ~50 lines: sliding_window(), vocab_from_column(), nfs_lock()
    features.py         # ~150 lines: Polars groupby-agg feature expressions
                        #   node_features(window_df) → tensor
                        #   edge_features(edge_df) → tensor
```

**Total: ~450 lines** instead of 2,150. The savings come from:
- Killing abstractions (adapter, schema, engine, cache orchestration): ~1,200 lines
- Polars replacing numpy scatter: ~250 lines
- Remaining: ~250 lines CAN bus parsing + 150 lines features + 50 lines utils

### H2. Storage layer (970 lines) — KEEP GATEWAY, ENFORCE IT 🔄 RE-REVISED

**First claim:** Collapse to 50-line io_utils.py. Wrong — loses NFS safety.
**Second claim:** Flatten gateway to plain functions, save 370 lines. Also wrong — makes leaks worse.

**Third (honest) look: the gateway as a class is the RIGHT pattern.**

The gateway is dependency injection. Stage functions receive a gateway object and route
ALL file I/O through it. This choke point enables:
- NFS atomic writes in one place
- Path validation in one place
- Storage backend swaps (local → ESS → S3) in one place
- Audit/logging of all I/O in one place

**Flattening to plain functions would make leaks WORSE** — any module could call
`torch.save(obj, any_path)` and bypass everything.

**The real problem: existing I/O leaks**

| Leak | Where | What happens |
|------|-------|-------------|
| `save_embeddings()` | mapper.py | `np.savez_compressed` directly — NOT atomic |
| `save_attention()` | mapper.py | `np.savez_compressed` directly — NOT atomic |
| `_export_best_config()` | optuna_sweep.py | `path.write_text()` — bypasses gateway |
| `api.py:train()` | api.py | Calls `STAGE_FNS[stage](cfg)` raw — no gateway |
| Path primitives public | storage/paths.py | Any module can construct paths and write directly |

**Fix: I/O boundary tests (the only enforcement that works)**

```python
# test_io_boundaries.py
ALLOWED_IO_FILES = {
    "storage/mapper.py", "storage/gateway.py",
    "storage/manifest.py", "storage/catalog.py",
}
BANNED = ["torch.save(", "torch.load(", "json.dump(", "np.savez(",
          "pickle.dump(", ".write_text(", ".write_bytes("]

def test_no_direct_io():
    for py_file in all_graphids_py_files():
        if any(a in str(py_file) for a in ALLOWED_IO_FILES):
            continue
        content = py_file.read_text()
        for pattern in BANNED:
            assert pattern not in content, (
                f"{py_file} uses '{pattern}' — route through gateway/mapper"
            )
```

**Path privacy:** Make `lake_run_dir()`, `lake_cache_dir()` etc. private to storage layer.
Gateway is the ONLY public way to get a path. External code uses `gw.resolve()`, never
raw path functions.

**Dead code to remove:** `list_artifacts()` (0 callers), `read_bytes()` (0 callers).

**What to do:**
- Add `test_io_boundaries.py` — hard fail on direct I/O outside storage/
- Fix `save_embeddings`/`save_attention` to use atomic writes
- Extract `save_cka()` compute logic out of mapper
- Make path primitives private (`_lake_run_dir`, not `lake_run_dir`)
- Kill dead methods
- Keep gateway as a class — it's dependency injection, the right pattern

**Savings: ~100 lines** (dead code + CKA extraction). But the architecture gets
ENFORCED, which is worth more than line count.

### H3. Pipeline stages (2,130 → ~1,900 lines) ❌ REVISED

**Initial claim:** Push into LightningModules, save 1,330 lines.

**Reality after reading every function:**

The stages **already use Lightning correctly**:
- VGAEModule/GATModule have proper `training_step`, `validation_step`, `configure_optimizers`
- Eval inference **already uses `trainer.predict()`** via `_GATPredictor`/`_VGAEPredictor` wrappers
- Manual loops for attention capture and VGAE components are **justified** (GATConv return type changes)
- DQN fusion **cannot use `trainer.fit()`** — it's RL, not supervised
- `CurriculumDataModule` properly overrides `train_dataloader()` for per-epoch resampling

**`make_trainer()` cannot be `pl.Trainer(**cfg)`** because it does 7 things:
1. NFS-persistent `default_root_dir` (SLURM checkpoint survival)
2. Conditional `SLURMEnvironment(auto_requeue=True)`
3. CSVLogger with stage-specific path
4. ModelCheckpoint with domain convention `filename="best_model"`
5. EarlyStopping from config
6. DeviceStatsMonitor
7. `cudnn.benchmark` toggle

**`load_data()` is better than LightningDataModule** because each stage needs
different post-processing (curriculum needs difficulty scoring, fusion needs
`cache_predictions`, temporal needs `TemporalGrouper`). A monolithic DataModule
would be more complex.

**Breakdown (2,220 lines total):**
- Domain logic: ~751 lines (KD losses, metrics, RL loop, feature extraction, mmap safety)
- Wiring/glue: ~394 lines
- Imports/docstrings/logging: ~1,075 lines

**What IS removable:**
- `resolve_teacher_path()` (31 lines) — dead, never called outside trainer_factory
- `make_projection()` (21 lines) — dead outside prepare_kd, could inline
- `build_optimizer_dict()` (30 lines) → move into LightningModule `configure_optimizers()`
- `_extract_training_metrics()` (18 lines) → simplify to `trainer.callback_metrics`
- Temporal eval manual loop → `trainer.predict()` (~15 lines)
- `validate_datasets()` (17 lines) — dead
- `StageMetrics` TypedDict (2 lines) — dead

**Savings: ~230 lines** (dead code + inline small functions). Not 1,330.

### H4. Orchestration (940 → ~200 lines) — DETAILED

**Current:** 4 files, two jobs: DAG execution (submit stages in order on SLURM) and HPO (hyperparameter search).

#### H4a. optuna_sweep.py (302 lines) → hydra-optuna-sweeper (0 Python lines)

This file does three things, all replaceable:

1. **Search space loading** (`_load_search_spaces`, `_suggest_params`, lines 43-85): Parses YAML
   into `(type, low, high)` tuples, then maps to `trial.suggest_*` calls. Two-step translation.
   `hydra-optuna-sweeper` reads search spaces directly from Hydra config YAML with zero Python.

2. **Subprocess objective** (`_objective`, lines 93-119): Suggests params → builds CLI command →
   `subprocess.run` → reads val_loss from manifest. With `hydra-submitit-launcher`, each trial
   IS a Hydra run — the launcher handles subprocess dispatch and result collection.

3. **Pipeline sweep** (`run_sweep_pipeline`, `_run_multi_seed_final`, lines 236-302): Sequential
   3-stage sweep + train-best + multi-seed. This is orchestration logic that belongs in the
   DAG definition, not the sweep file. With Hydra multirun + sweeper, this becomes:
   `python -m graphids.cli stage=autoencoder --multirun 'training.lr=interval(1e-4, 1e-2)'`

**What stays:** Nothing. The warm-start logic (`_enqueue_warm_start`, 10 lines) could become
a Hydra plugin callback if needed. The SQLite storage is built into Optuna regardless.

**`load_best_config()` is dead code** — never called outside this file.

**Spike needed:** Verify `hydra-optuna-sweeper` works with our Hydra Compose API setup
(we use `compose_config()`, not `@hydra.main`). May need a thin adapter.

#### H4b. slurm_primitives.py (245 lines) → submitit (~30 lines)

Current file does:

| Function | Lines | submitit equivalent |
|----------|------:|---------------------|
| `generate_sbatch_script()` | 72 | Not needed — submitit submits Python callables directly |
| `write_script_file()` | 14 | Not needed |
| `submit_sbatch()` | 8 | `executor.submit(fn, *args)` |
| `sacct_query()` | 10 | `job.state` / `job.done()` |
| `poll_until_done()` | 9 | `job.result()` (blocking) or `job.done()` (polling) |
| `scale_resources()` | 14 | Keep — domain logic for adaptive retry |
| `get_resources()` | 10 | Keep — resource profile lookup |
| `_parse_resource_profiles()` | 8 | Keep — YAML loading |
| `ResourceSpec` (job.py) | 83 | `submitit.SlurmExecutor.update_parameters()` takes the same kwargs directly |

The big win: **`generate_sbatch_script()` (72 lines of string building) disappears entirely.**
submitit submits Python callables — no bash scripts, no preamble sourcing, no string templating.

**Preamble concern:** Our `_preamble.sh` loads modules (`module load python/3.12 cuda/12.4`)
and activates the venv. submitit has `setup` parameter for pre-execution shell commands:
```python
executor.update_parameters(setup=["source scripts/slurm/_preamble.sh"])
```

**Signal concern:** Lightning's `SLURMEnvironment(auto_requeue=True)` catches SIGUSR1.
submitit forwards signals to the subprocess by default. Should work, needs spike.

**What stays (~60 lines):**
- `scale_resources()` + `FAILURE_REACTIONS` (adaptive retry — domain logic)
- `get_resources()` + `_parse_resource_profiles()` (YAML resource lookup)
- `SlurmJobFailed` exception (used by Dagster retry policy)

**ResourceSpec (job.py, 83 lines):** `from_yaml()` (39 lines) is imperative parsing that
Pydantic `field_validator` handles declaratively. `mem_slurm` and `walltime_slurm` properties
are only needed for bash script generation — with submitit, pass `memory_gb` and `walltime`
directly. **Entire file could collapse to ~20 lines** or be replaced by
`submitit.SlurmExecutor.update_parameters(mem_gb=32, timeout_min=240, ...)`.

#### H4c. pipes_slurm.py (123 lines) → submitit integration

`PipesSlurmClient` is a Dagster `PipesClient` that:
1. Opens a Pipes session (NFS file-based context injection + message reading)
2. Generates + submits sbatch script
3. Polls until done
4. Returns Pipes results

With submitit replacing step 2-3, this becomes:
1. Open Pipes session (keep — Dagster integration)
2. `executor.submit(train_fn, cfg)` (submitit)
3. `job.result()` (submitit blocking wait)
4. Return Pipes results (keep)

The file shrinks from 123 → ~40 lines. `submit_no_poll()` (28 lines) becomes
`executor.submit()` + return job ID.

#### H4d. dagster_defs.py (271 lines) — mostly stays

This file has three parts:

1. **DAG topology** (`build_dag_topology`, `DagNode`, lines 148-195): Builds the pipeline
   graph from `STAGE_DEPENDENCIES` + `PipelineConfig.variants`. This is the actual pipeline
   definition — **must stay** regardless of SLURM tooling. ~80 lines.

2. **Asset factories** (`_make_stage_asset`, lines 65-107): Creates Dagster `@dg.asset` for
   each stage, wires retry policy and adaptive scaling. With submitit, the asset body
   simplifies (no script generation), but the Dagster asset structure stays. ~50 lines → ~30.

3. **Fire-and-forget** (`fire_and_forget`, lines 223-261): Topological sort → submit with
   `--dependency=afterok` chains. With submitit, `--dependency` is passed via
   `executor.update_parameters(slurm_additional_parameters={"dependency": f"afterok:{dep_id}"})`.
   Logic stays the same, just cleaner submission. ~40 lines → ~30.

4. **HF push + catalog rebuild assets** (lines 110-145): Thin wrappers. Keep.

**What stays:** ~150 lines (topology + Dagster asset wiring + fire-and-forget logic).
**What shrinks:** ~120 lines of boilerplate simplified by submitit.

#### H4e. Orchestration summary

| File | Current | Target | Savings | Replacement |
|------|--------:|-------:|--------:|-------------|
| optuna_sweep.py | 302 | 0 | **302** | hydra-optuna-sweeper plugin |
| slurm_primitives.py | 245 | 60 | **185** | submitit (kill script gen, polling) |
| pipes_slurm.py | 123 | 40 | **83** | submitit for submission |
| job.py | 83 | 20 | **63** | submitit params or Pydantic validators |
| dagster_defs.py | 271 | 150 | **121** | simplify asset bodies with submitit |
| **Total** | **940** (+ 83 job.py) | **270** | **754** | |

#### H4f. Alternative: do you even need Dagster?

Dagster adds: UI, asset lineage, partition management, retry policies, scheduling.

**What you actually use from Dagster:**
- Asset DAG with `@dg.asset` + `deps=` (could be `submitit` + `graphlib.TopologicalSorter`)
- Retry via `RetryPolicy` (could be a simple retry loop)
- Partitions for dataset × seed (could be CLI args + a bash loop)
- Pipes protocol for cross-node metrics (could be manifest files — which you already have)

**What you DON'T use:**
- Dagster UI (headless HPC, no web server)
- Dagster scheduling (SLURM is the scheduler)
- Dagster IO managers (you have StorageGateway)
- Dagster sensors / schedules

`fire_and_forget()` already works WITHOUT Dagster — it uses `graphlib.TopologicalSorter`
+ SLURM `--dependency=afterok`. The Dagster layer is infrastructure for a daemon-based
workflow that you don't run.

**If Dagster is dropped entirely:**
- `dagster_defs.py` (271) → `fire_and_forget()` function moves to a standalone module (~60 lines)
- `pipes_slurm.py` (123) → deleted (Pipes protocol not needed without Dagster)
- Remove `dagster` from dependencies

**This would make orchestration = `fire_and_forget.py` (~60 lines) + `slurm.py` (~30 lines with submitit) + resource profiles (~30 lines) = ~120 lines total.**

But this is a bigger decision — Dagster may become useful when running the pipeline
regularly for multiple datasets. **Decision needed: is Dagster earning its place, or is
fire-and-forget sufficient?**

#### H4g. Tools investigated

| Tool | What it does | Applicable? |
|------|-------------|-------------|
| **submitit** | Python-native SLURM submission | **YES** — replaces script gen + submission + polling |
| **hydra-submitit-launcher** | Hydra plugin for submitit | **YES** — each Hydra run becomes a SLURM job |
| **hydra-optuna-sweeper** | Hydra plugin for Optuna | **YES** — replaces entire optuna_sweep.py |
| **Dagster** (already in stack) | Workflow orchestration | **MAYBE** — only fire_and_forget is used, no daemon |
| **simple-slurm** | Python SLURM wrapper | Weaker than submitit — still generates scripts |
| **snakemake** | Workflow manager + SLURM | Overkill — designed for bioinformatics pipelines |
| **Prefect** | Alternative to Dagster | Same daemon problem — no advantage on HPC |
| **dask-jobqueue** | Dask workers on SLURM | Wrong model — we need job submission, not workers |

**Spikes needed:**
1. submitit on OSC: verify `executor.update_parameters(setup=["source _preamble.sh"])` loads modules correctly
2. submitit signal forwarding: verify SIGUSR1 reaches Lightning for auto-requeue
3. hydra-optuna-sweeper: verify it works with `compose_config()` (not `@hydra.main`)

### H5. Config (730 → ~500 lines)

Mostly well-structured already. Minor wins:
- `_hydra_bridge.py` (179) — keep, this is the integration point
- `schema.py` (269) — keep, Pydantic models are the config
- `paths.py` (184) — ~40 lines of lake path functions moved to storage, already done this session
- `constants.py` — keep

**Savings:** ~50 lines (already cleaned this session).

### H6. Models (1,490 — mostly keep)

- `vgae.py` (314), `gat.py` (161) — domain architectures. Keep.
- `dqn.py` (632) — largest single file. Has hand-rolled `TensorReplayBuffer` (~70 lines). `tianshou`, `stable-baselines3`, or even `torchrl.data.ReplayBuffer` provide this. The MLP/WeightedAvg baselines (~130 lines) could be separate files or deleted if not used in experiments yet. Worth investigating but lower priority than infrastructure.
- `registry.py` (124) → Hydra `instantiate()` (see F4)
- `fusion_features.py` (126) — Protocol-based extractors. Keep.

**Savings:** ~200 lines (replay buffer → library, registry → Hydra instantiate).

---

## I. Total reduction potential (REVISED after deep read)

> The initial estimates below were wrong. Deep reads of every file revealed
> that much of the "replaceable" code is genuinely domain-specific, NFS-safe,
> or already using Lightning correctly. Revised numbers follow.

### What was WRONG in the initial estimate

1. **Preprocessing → PyG InMemoryDataset: WRONG.** PyG's InMemoryDataset lacks mmap
   support (`torch.load(mmap=True)`), does `copy.copy()` on every `__getitem__` (doubles
   memory), and doesn't do NFS-safe atomic writes. Our `CollatedGraphDataset` is 90 lines
   and uses PyG's own `collate()`/`separate()` internally. The preprocessing code is almost
   entirely domain-specific: CAN bus hex parsing, 26-D node + 11-D edge feature engineering
   via scatter ops, sliding-window graph construction, attack type inference from directory
   names. None of this has a library replacement.

2. **Pipeline stages → push into LightningModules: WRONG.** VGAEModule and GATModule
   already use Lightning correctly (proper `training_step`, `validation_step`,
   `configure_optimizers`, `self.log(batch_size=...)`). Eval inference already uses
   `trainer.predict()` with justified manual fallbacks for attention capture and VGAE
   component decomposition. `make_trainer()` does 7 environment-specific things (NFS
   persistent root, SLURM auto-requeue plugin, domain checkpoint naming) that
   `pl.Trainer(**cfg)` cannot replace. `load_data()` returning raw lists is more flexible
   than a monolithic LightningDataModule because each stage needs different post-processing.

3. **Storage → inline torch.save: PARTIALLY WRONG.** The atomic write logic (tmpfile +
   fsync + rename with retry) is genuinely needed on NFS — bare `torch.save(path)` produces
   corrupt files if a reader opens during write or node crashes. BUT: the gateway is
   overwhelmingly used as a path resolver (~30 call sites), not for atomic writes (8 sites).
   The path resolution could be plain functions instead of a class.

### What IS still reducible

| Layer | Current | Realistic target | Savings | Replacement |
|-------|--------:|-------:|--------:|-------------|
| Preprocessing | 2,150 | 450 | **1,700** | PyG InMemoryDataset per domain + Polars features |
| Orchestration | 940 | 200 | **740** | submitit + hydra-optuna-sweeper |
| Storage | 970 | 870 | **100** | Enforce gateway (don't flatten), kill dead code, I/O boundary tests |
| Pipeline stages | 2,130 | 1,900 | **230** | Inline wiring, dead functions |
| Config | 730 | 680 | **50** | Already cleaned this session |
| Models | 1,490 | 1,400 | **90** | Registry → Hydra instantiate |
| Top-level + glue | 720 | 680 | **40** | Already cleaned this session |
| **Total** | **~10,130** | **~6,180** | **~2,950** | |

**Honest target: ~6,200 lines** — 29% reduction. Two big wins (preprocessing
restructure + Polars: 1,700 lines; orchestration tooling: 740 lines). Storage stays
roughly the same size but gets ENFORCED via I/O boundary tests. Line count isn't the
only metric — architectural enforcement matters more than shaving lines.

---

## J. Execution order (biggest impact first)

### REVISED execution order

1. **Preprocessing → PyG InMemoryDataset per domain** (1,350 lines) — biggest win. Restructure from adapter/schema/engine pipeline to self-contained dataset classes. Spike: prove InMemoryDataset subclass works with mmap override + NFS atomic save.
2. **Orchestration → submitit + hydra-optuna-sweeper** (740 lines) — second biggest. Spike: prove submitit handles OSC preamble/modules/signals.
3. **Storage → flatten gateway + kill dead code** (370 lines) — straightforward refactor, no spike needed.
4. **Pipeline stages → inline dead functions + small cleanups** (230 lines) — low risk.
5. **Models + config** (180 lines) — registry → Hydra instantiate, minor cleanups.

Each step is independently valuable. No step depends on another.

### Process: spike-first, no excuses

For each item above, BEFORE the refactor:

1. Write a spike script under `tests/spikes/spike_<name>.py` (~20-30 lines)
2. The spike proves the library API handles the specific use case (not a toy example — use real data/config)
3. Run the spike. If it passes → proceed with full refactor. If it fails → update this doc with WHY.
4. Never say "we actually need the custom code" without a failing spike to prove it.

This prevents the pattern of planning big refactors then preserving custom code by finding "complications" during implementation.

---

## E. What's genuinely domain-specific or NFS-required (keep)

**Preprocessing (2,050 lines):**
- CAN bus adapter (332) — hex parsing, temporal adjacency, attack inference
- Feature engineering (389) — 26-D node + 11-D edge via scatter ops, entropy, clustering
- Graph engine (188) — sliding window → PyG Data
- Vocabulary (166) — CAN ID → dense index with OOV
- Schema (248) — IR column contract, feature manifest
- CollatedGraphDataset (90 real lines) — zero-copy mmap, uses PyG collate/separate internally
- Cache orchestration (395) — load-or-rebuild with NFS locking, version validation
- Parallel (286) — Ray fan-out, sequential fallback

**Models (1,490 lines):**
- VGAE (314), GAT (161), DQN (~500 core) — the actual research
- Fusion feature extractors (126) — Protocol-based feature engineering
- Temporal model (125) — GAT encoder + Transformer

**Pipeline stages (1,900 lines after cleanup):**
- Lightning modules (285) — VGAEModule, GATModule, CurriculumDataModule. Proper Lightning usage.
- Evaluation (374) — torchmetrics + domain thresholds, already uses trainer.predict()
- Eval inference (277) — trainer.predict() wrappers + justified manual loops
- Training (162) — thin trainer.fit() wrappers + difficulty scoring
- Fusion (209) — DQN RL loop (can't use Lightning) + MLP/WeightedAvg (uses Lightning)
- Temporal (282) — experimental stage, keep but detach (see Decisions)
- Trainer factory (~230 after cleanup) — make_trainer does 7 env-specific things
- Data loading (221) — mmap safety, DynamicBatchSampler, feature caching

**Config (680 lines after cleanup):**
- Hydra bridge (179) — config composition
- Pydantic schema (269) — config models
- Paths (184) — path derivation, env settings
- Constants — topology, preprocessing params

**Storage (600 lines after cleanup):**
- Atomic write + lock (~80 lines) — NFS-required, no library replacement
- Path layout (~70 lines) — lake directory convention
- Manifest (213) — deliberate MLflow replacement, fire-and-forget mode
- Catalog (178) — deferred decision
