# GraphIDS Session Plan

> Last updated: 2026-04-14 (session 48 — submit-profile auto-sizing)

PLAN.md is current-session work only. Historical session changelogs live in
git log; durable verdicts live in `docs/decisions/README.md`; living architecture
lives in `docs/reference/`.

## Active

- **Validate GPU-first auto-sizing end-to-end** — job 46763851
  (`pipeline-run --dataset hcrl_ch -O trainer.max_epochs=3` on gpudebug/1hr)
  pending. Watch for `probe_done` in `traces.jsonl` + `node_budget.binding
  == "memory"`.
- **Harvest training-time coefficients** — once a few real pipeline runs
  land, write a script that reads `traces.jsonl` + sacct MaxRSS and refits
  the `stage_profiles` scaling blocks (currently placeholders). Today's
  training coefficients are unverified guesses with ~40min/small,
  ~80min/large worst-case budgets.

## Recently landed (this session)

- **Pipeline fragility audit follow-ups** (decouple + resume-from-ckpt):
  `run_pipeline` no longer calls analysis — analysis runs via `python -m
  graphids analyze --ckpt-path <p>` after training. Resume skip-check is
  now authoritative on `best_model.ckpt` existence; the `.complete` marker
  (a Dagster-era workaround) and `.analyze_complete` retired. `evaluate()`
  writes `.test_complete` only on success; OOM/CUDA crashes during test
  no longer silently mark the stage done. `PipelineResult.analyzed_assets`
  removed from the public dataclass. ADR 0003 annotated as partially
  superseded; ADR 0006 updated to reflect the decoupling. Docs swept:
  orchestration.md, write-paths.md, config-architecture.md,
  responsibilities.md, observability-data-layers.md,
  `.claude/rules/config-system.md`.
- **Identity-hash strictness** — plan filed at
  `~/plans/graphids-identity-hash-strictness.md` capturing the 3 options
  (reject None, tagged sentinel, jsonnet default check) for the remaining
  #4 finding. No action now; revisit if a collision surfaces.
- **Config/ Python cleanup** (7-item sweep, 651 → 568 LOC):
  `RunDirIdentity` (37 LOC) and `PathContext.for_checkpoint` (28 LOC)
  deleted as dead code; `data_dir()` fallback path collapsed; `render_config`
  → `render` alias dropped; `schemas.py` inlined one-shot monitor classes
  into a 5-line helper; `AxesConfig` wrapper flattened; `ModelType` Literal
  kept but guarded by an import-time assert against `axes.json` to prevent
  drift.
- **Config/schema cleanup** (audit-driven, no behavior change to callers):
  deleted three dead files (`configs/resources/clusters.json`,
  `job_profiles.json`, `dataset_scaling.json`) and their validator reads;
  inlined the 4-entry GPU VRAM map into `test_budget_matrix.py`. Collapsed
  `render_config` → `render` in `config/jsonnet.py` (two-level alias gone,
  one caller updated). `compute_identity_hash` now raises `KeyError` on
  unknown stage instead of returning `""` silently. New
  `_validate_submit_profiles` runs at package import — catches typos in
  composed `stages` lists, missing `cpus`/`scaling` fields, and
  out-of-range `scale_mult` keys at import time instead of sbatch time.
  Two new regression tests; docs updated (CLAUDE.md, config-system.md,
  CONFIG_REFERENCE.md, config-architecture.md, responsibilities.md,
  README).
- **submit-profile auto-sizing**: `configs/resources/submit_profiles.json`
  gains optional `scaling: {time_min, mem_gb}` blocks and a top-level
  `stage_profiles` map. `submit-profile` CLI accepts `--dataset` / `--scale`,
  reads `cache_metadata.json.aggregate.num_raw_samples`, and composes
  pipeline resources (time=sum, cpus/mem=max across stages). `rebuild-caches`
  coefficients fit from last session's 6 datapoints (OLS + 1.3× safety);
  training-stage coefficients are placeholders pending calibration runs.
  `scripts/slurm/submit.sh` sniffs `--dataset`/`--scale` from args and
  forwards to the CLI — all existing callsites unchanged when no dataset
  given (falls back to `defaults` block). 8-test regression at
  `tests/test_submit_profile.py` (invariants: composition rules,
  monotonicity, static profile unchanged, defaults fallback).

## Recently landed (last session)

- **Preprocessing data-correctness fix (§5.0)**: `CANBusDataset` no longer
  rglobs the dataset root; reads from explicit `source_dirs`. Train scope
  comes from catalog (`train_subdir` + `train_attack_subdir`); each test
  subdir gets its own `data_test_<subdir>.pt` tensor. Closes the
  Sev-1 train↔test contamination + "all test_N eval against test_01"
  regressions.
- **VGAE benign-only training**: `GraphDataModule.label_filter="benign"`
  hparam; `_effective_train_ds` returns a `y == 0` PyG subset view.
  Wired into `configs/stages/autoencoder.jsonnet` only (supervised +
  fusion stages untouched).
- **Cache metadata schema v2** (`graphids/core/data/metadata.py`):
  per-split entries (`splits.<name>` with `num_raw_samples`, `num_graphs`,
  `bytes_on_disk`, `attack_balance`, `graph_stats`, `source_dirs`),
  `aggregate` totals, fcntl flock + atomic rename merge writer. Train
  build emits both `train` and `val` entries; each test subdir merges its
  own. v1 caches now rejected with rebuild instructions.
- **`validate-metadata` CLI** + 7 regression tests covering merge order,
  invariant mismatch, v1 rejection, aggregate consistency, missing
  test-subdir detection.
- **`budget.py`** reads `splits.train.graph_stats.node_count` via
  `load_metadata` (was top-level `graph_stats`).
- **Rebuilt every cache** under v8.0.0 — every prior tensor was
  contaminated. New per-dataset numbers:

  | Dataset | Graphs | Raw rows | Cache size | Splits |
  |---------|--------|----------|------------|--------|
  | hcrl_sa | 19,083 | 1.9M | 200 MB | 6 |
  | hcrl_ch | 185,468 | 18.5M | 1.8 GB | 6 |
  | set_01  | 592,348 | 59.2M | 6.2 GB | 8 |
  | set_02  | 702,090 | 70.2M | 7.7 GB | 8 |
  | set_03  | 601,292 | 60.1M | 7.1 GB | 8 |
  | set_04  | 623,431 | 62.3M | 6.8 GB | 8 |

- **Resource profile right-sized**: `rebuild-caches` SLURM profile
  trimmed from `4cpu/128G/3h` → `6cpu/54G/1h` (peak observed RAM = 38.5 GB
  on set_02; longest elapsed = 3:40).
- **Dead `stage-data` call removed** from `_preamble.sh`. `cpu-raw`
  submit mode collapsed into `cpu`. Old staging command had been
  silently failing inside `eval "$( … | grep '^export ')"` for several
  refactors; rebuild jobs reading from ESS NFS directly took 1–4 min,
  no staging needed for our working set. Docs (`slurm-hpc.md`,
  `CONFIG_REFERENCE.md`, `write-paths.md`) updated.

## Earlier (session 46)

- Replaced `torch.profiler` in `BudgetProfiler.probe` with
  `torch.cuda.max_memory_allocated()` + `time.perf_counter()`.
- Wired `model=` into `GraphDataModule._budget_result` so the probe runs
  inline at fit-start (also fixed the PrefetchLoader-on-host bug from
  unset `_set_device`).
- Retired `probe-budget` CLI; deleted `graphids/plots/` + orphaned
  `core/models/factory.py`. Probe now per-(model, dataset, conv).
- CLI consolidation: dropped leading underscores; folded
  `submit-profile` into `cli/app.py`; deleted `cli/_slurm.py`.

## Open issues

- **#18** Validate GPU-first auto-sizing on SLURM — now unblocked, run a
  training job to confirm
- **#32** Add WaDi dataset module
- **#33** spam, can be closed/blocked

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Issues: `gh issue list`
