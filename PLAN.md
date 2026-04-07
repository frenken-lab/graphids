# GraphIDS Session Plan

> Last updated: 2026-04-06 (session 35 — Monarch integration)

## What this session did (2026-04-06, session 35 — Monarch integration)

Added `graphids/monarch/` subpackage for running the 3-stage pipeline
(autoencoder → supervised → fusion) in a single SLURM allocation via
PyTorch Monarch actors. Zero modifications to existing training code.

- **`monarch/__init__.py`** — `available()` import guard
- **`monarch/actors.py`** — `PipelineActor` with `train_autoencoder`,
  `train_supervised`, `train_fusion` endpoints wrapping `_execute()`.
  `__supervise__` fault handler. Falls back to plain class without monarch.
- **`monarch/job.py`** — `MonarchJobSpec` + `pipeline_job_spec()` aggregates
  resource profiles across stages. `create_slurm_job()` / `connect_or_create()`
  for SlurmJob lifecycle.
- **`monarch/pipeline.py`** — `PipelineConfig` + `run_pipeline()` controller
  with retry logic.
- **`cli/_monarch.py`** — `monarch-run` command with `--dry-run` mode.
- Installed `torchmonarch==0.4.0` (PyPI package name, NOT `monarch`).
  Verified: imports, `SlurmJob` construction, `Actor` + `@endpoint` wiring.

## Next session — Monarch substitution + SLURM spike

Two tracks:

### Track 1: SLURM compute-node spike
Submit an actual Monarch job via `salloc` to verify:
1. Worker processes inherit SLURM env vars (TMPDIR, etc.)
2. `bootstrap_staging()` works inside `spawn_procs(bootstrap=)`
3. `PipelineActor` can be spawned and `train_stage` runs Lightning
4. Single-stage autoencoder training completes inside an actor endpoint

### Track 2: Substitute Monarch for obsolete code
With Monarch handling single-allocation lifecycle, identify and remove
code that Monarch now replaces:
- `graphids/slurm/pipeline.py` — `generate_script()`, `SubprocessSlurmJobClient`,
  `write_training_spec()` may be partially obsolete for Monarch-path runs
- `scripts/slurm/submit.sh` — Monarch's `SlurmJob` replaces sbatch generation
  for pipeline runs (keep for standalone jobs like tests, caches)
- `graphids/orchestrate/dagster/` — evaluate what dagster still owns vs what
  the Monarch controller replaces for simple linear pipelines
- `graphids/orchestrate/ops/entrypoint.py` — `from-spec` phases may merge
  with actor endpoints

**Guiding principle:** Monarch is the new execution path for multi-stage
pipelines. Dagster may remain for sweep orchestration (which configs to
run), but job-level lifecycle (submission, polling, env setup, phase
sequencing) moves to Monarch.

**Known items:**
- `actors.py` at 333 lines (target <320 — need to extract utilities)
- Fusion resource profile over-allocates (CPU-only but GPU partition)
- `__supervise__` absorbs failures but doesn't restart — verify this
  doesn't silently swallow real errors

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
