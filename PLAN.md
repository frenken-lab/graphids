# GraphIDS Session Plan

> Last updated: 2026-04-07 (session 37 — config consolidation + shared nodes)

## What this session did (2026-04-07, session 37)

Config consolidation, jsonnet simplification, and Monarch shared-node support.

### Config changes
- **Dataset-aware resources:** `configs/resources/dataset_scaling.json` with
  per-dataset time_scale factors. `hcrl_sa` allocation 21h → 4h.
- **Pipeline defaults single source:** `axes.json` `pipeline_defaults` read
  by both jsonnet TLA defaults and Python CLI. Zero three-way drift.
- **Jsonnet merge chain 8→4 steps:** defaults flattened, model libsonnets
  pre-merge base+scale, 3 analyze configs → 1 with model_type dispatch.
- **Callbacks declared in jsonnet:** 6 forced callbacks moved from Python
  to `defaults.libsonnet` named callbacks object with late-binding
  `trainer.callbacks` list via `$` reference.
- **Fusion configs → `configs/models/fusion/`:** mirrors `graphids/` structure.

### instantiate.py 505→150 lines
- Loss construction → `core/losses/build.py`
- Link arguments → jsonnet stages (cross-links in data/model init_args)
- Callbacks/loggers → config-driven generic instantiation loop

### Monarch improvements
- `--trainer-override` / `-O` CLI flag (wired through to jsonnet TLAs)
- SLURM logs write to data lake via native `log_dir` param
- `exclusive=False` with clusterscope multi-GRES parser patch (rpartition)
- Dagster bloat removed: `slurm/core/submit.py`, `slurm/pipeline.py`
- `orchestrate/definitions.py` → `orchestrate/dagster/definitions.py`

### Partially fixed: autoencoder memory ceiling
The autoencoder profile requests 181G (20 CPUs × 9.2G mem_per_cpu) because
the full dataset was too large to fit in fewer workers' memory. Sessions
34-35 pre-batched the training data to reduce per-worker memory, but the
resource profile was never updated to reflect the reduced footprint.
**Status:** pre-batching landed, profile still oversized. Next step: profile
actual peak memory on hcrl_sa with pre-batched data, then right-size the
autoencoder entry in `job_profiles.json` and add mem scaling to
`dataset_scaling.json`.

### Practice run submitted
Job 46503000 on gpudebug (shared, 32G/4CPU, 3 epochs, hcrl_sa, seed 55).
Pending on Priority — validates checkpoint threading across all 3 stages.

## Next session

### Track 1: Validate 3-stage pipeline
If practice run succeeded, review output. If still queued, resubmit.
Then run full training on `hcrl_sa` with production epochs.

### Track 2: Right-size autoencoder memory
Profile peak memory with pre-batched data on a compute node. Update
`job_profiles.json` autoencoder entry. Add mem scaling to
`dataset_scaling.json`.

### Track 3: Dagster deletion
`rm -rf orchestrate/dagster/` + remove `[tool.dg]` from pyproject.toml.
Gated on successful Monarch sweep validation.

### Known deferred items
- `analyze` command interface: `--tla 'ckpt_path="..."'` (jsonnet TLA)
- Fusion stage absorbs `auxiliaries=[]` and `vgae_ckpt_path=null` as
  ignored TLAs

## What session 36 did (2026-04-06)

Rewired Monarch actor to use `ConfigResolver.resolve()` (same path as
dagster) and validated end-to-end on a GPU compute node.

### ConfigResolver integration (the main event)
- **Replaced hand-rolled `_prepare_stage`** with `ConfigResolver.resolve()`.
  Actor builds a `StageConfig` (via `_build_stage_config`) matching planner
  output, passes it to the resolver. All TLA construction, identity hashing,
  path computation, rendering, and validation now use the canonical path.
- **Fixed 4 bugs** in the old actor:
  1. Path divergence — used `"vgae"/"gat"/method` instead of family names
     (`"unsupervised"/"supervised"/"fusion"`). Checkpoints now land at same
     paths as dagster.
  2. Missing `model_type` in identity dict (crashed autoencoder).
  3. Missing `loss_fn` / `method` (would crash supervised / fusion).
  4. No cross-field validation (skipped `validate_stage_config`).
- **Added `rendered` field to `ResolvedConfig`** — resolver already renders
  internally; actor is in-process so re-rendering is wasted work.
- **Identity + model_type verified** — actor and planner produce identical
  hashes for all 3 stages (autoencoder, supervised, fusion).
- **Deleted `_STAGE_META`** — replaced by topology lookups + `STAGE_FAMILY_MAP`.

### Other changes
- **Extracted `monarch/_setup.py`** — `ensure_spawn`, `touch_marker`,
  `bootstrap_staging` shared by actors and pipeline controller.
- **Stage-aware `pipeline_job_spec`** — accepts `stages` list, avoids
  12h GPU waste when fusion excluded. 2-stage: 9h vs 3-stage: 21h.
- **`__supervise__` verified correct** — absorbs structural failures,
  endpoint errors still reach `_run_with_retry` via `Future.get()`.
- **Fixed `_preamble.sh` eval bug** — rsync progress with parentheses
  broke `eval $(stage-data)`. Fixed with `grep '^export '`.
- **Added `loss_fn`** to `PipelineConfig`, `PipelineActor`, CLI.
- **Spike script** — `scripts/spike_monarch.py` + `spike-monarch` submit
  profile. **ALL 5 STEPS PASSED** on gpudebug (p0255):
  torchmonarch import → env vars → bootstrap_staging → ConfigResolver
  `_prepare_stage` → VGAEModule fast_dev_run fit (GPU, 100K params).
- **Full Monarch pipeline validated** — `run_pipeline` from login node
  with autoencoder `fast_dev_run`. Monarch submitted SLURM job, spawned
  actor, ran Lightning fit, returned checkpoint path at correct location:
  `unsupervised_small_autoencoder_ff9f9014/seed_42/checkpoints/best_model.ckpt`
  (matches dagster planner convention). Eval stage had lenient failure
  (expected — fast_dev_run doesn't write a real checkpoint).
- **Monarch↔OSC compatibility fixes:**
  - `exclusive=True` on `SlurmJob` — bypasses `clusterscope` library
    which can't parse OSC's multi-GRES `sinfo` output (10+ GRES types
    per node cause `ValueError` in comma-split parsing).
  - `scripts/slurm/monarch_python.sh` — worker wrapper that sources
    `.env` + CUDA config before exec'ing venv Python. Monarch's bare
    `srun python -c '...'` skips `_preamble.sh`, so workers were missing
    `KD_GAT_LAKE_WRITE` etc. The wrapper is the `python_exe` for SlurmJob.
  - Fixed `_preamble.sh` eval bug — rsync progress with parentheses
    broke `eval $(stage-data)`. Fixed with `grep '^export '`.
- **Track 2 finding:** `slurm/pipeline.py`, dagster, `ops/entrypoint.py`
  still needed for the dagster path. No code to remove yet.

## Next session — Dagster↔Monarch boundary + multi-stage run

### Track 1: Full 3-stage pipeline
Run `monarch-run` with all 3 stages (autoencoder → supervised → fusion)
on the real `hcrl_ch` dataset. This validates checkpoint threading
between stages and dataset caching on the actor.

### Track 2: Dagster ↔ Monarch boundary decision
Both paths now work end-to-end. Decide:
- **Option A:** Dagster plans sweeps → Monarch executes each pipeline
  (dagster asset calls `run_pipeline` instead of `SubprocessSlurmJobClient`).
  Removes `slurm/pipeline.py` generate_script/SubprocessSlurmJobClient.
- **Option B:** Keep dagster path for sweeps, Monarch for interactive
  single-pipeline runs. Both paths coexist indefinitely.
- **Option C:** Drop dagster entirely for linear pipelines, keep only
  for multi-recipe sweeps.

### Known deferred items
- `instantiate.py` broken imports (`graphids.callbacks`,
  `CurriculumEpochCallback`) — fire at training time, not import time.
- `analyze` command interface: `--tla 'ckpt_path="..."'` (jsonnet TLA).
- Fusion stage absorbs `auxiliaries=[]` and `vgae_ckpt_path=null` as
  ignored TLAs.

## Previous session (2026-04-06, session 35 — Monarch integration)

Added `graphids/monarch/` subpackage for running the 3-stage pipeline
(autoencoder → supervised → fusion) in a single SLURM allocation via
PyTorch Monarch actors. Zero modifications to existing training code.

---

## Previous session (2026-04-06, session 34 — docs audit & compaction)

Audited all docs against the refactored codebase and fixed stale references:

- **Tier 1 (high impact):** Updated CLAUDE.md, config-system.md,
  copilot-instructions.md, config-architecture.md, 3-chain.md — all
  `commands/` → `cli/`, `core/instantiate` → `instantiate`, stage/model
  name renames, callbacks path fixes.
- **Tier 2 (medium):** Fixed kd-pipeline.md, observability.md,
  write-paths.md — stale module paths. Updated migration_plan.md —
  marked all phases complete, deferred PyIceberg.
- **Tier 3 (cleanup):** Deleted `docs/config_reorg.md` (completed
  checklist). Renamed typo'd filenames (`directory_strucuture` →
  `directory_structure`, `responsibilites` → `responsibilities`).
  Added stale-reference notes to ADRs 0001–0006. Compacted PLAN.md
  (dropped sessions 1–25).

## Next session — SLURM smoke test

Verify end-to-end via `scripts/slurm/submit.sh tests`. The Typer CLI,
jsonnet render, Pydantic validation, and instantiate chain are all
wired but only import-tested on login node.

**Known deferred items:**

- `instantiate.py` still has broken imports (`graphids.callbacks`,
  `CurriculumEpochCallback` without import). These fire at training
  time, not import time.
- `orchestrate/ops/entrypoint.py` imports `run_training_from_spec` /
  `run_test_from_spec` from `core.train_entrypoint` — now exists.
- `analyze` command interface changed: `--analyzer.ckpt_path` →
  `--tla 'ckpt_path="..."'` (jsonnet TLA instead of jsonargparse
  dotted override)
- Fusion stage still absorbs `auxiliaries=[]` and `vgae_ckpt_path=null`
  as ignored TLAs.

---

## Recent session history

### Session 33 (2026-04-06) — contract docs cleanup

- Removed remaining `TrainingContract` / `AnalysisContract` references from
  orchestration/analysis docs and rules. Rewrote ADR 0009 for jsonnet +
  `validate_config` pipeline.

### Session 32 (2026-04-06) — SLURM env access

- Centralized SLURM environment reads in `graphids.slurm.env` and replaced
  direct `os.environ` reads in logging, orchestration, callbacks, and budget.

### Session 31 (2026-04-06) — SLURM refactor

- Split `graphids/slurm` into `core/` (accounting + submit), `ops/`
  (profile + staging), and `pipeline.py` for GraphIDS-specific spec plumbing.

### Session 30 (2026-04-06) — Dagster ResourceParam

- Swapped `context.resources.slurm` for `ResourceParam[SlurmTrainingResource]`
  injection in the Dagster asset factory.

### Session 29 (2026-04-06) — Dagster runtime helpers

- Moved Dagster runtime helpers (partition keys, path context, complete marker)
  into `graphids/orchestrate/dagster/runtime.py`.

### Session 28 (2026-04-06) — Orchestrate decomposition

- Reorganized `graphids/orchestrate` into subpackages (`dagster/`, `planning/`,
  `resolve/`, `ops/`, `contracts/`).

### Session 27 (2026-04-06) — Copilot instructions

- Added `.github/copilot-instructions.md`.

### Session 26 (2026-04-05) — Typer CLI + config reorg

- Replaced `graphids/commands/` (12 files, argparse) with `graphids/cli/` (Typer).
- Completed stage name migration (normal/curriculum → supervised) and model
  family migration (vgae/dgi/gat → unsupervised/supervised).
- Fixed ~15 broken imports from earlier incomplete refactors.

## Key references

Work items live in GitHub issues now, not `docs/backlog/` (deleted
wholesale). Use `gh issue list` or the `/gh` skill.
