# GraphIDS Session Plan

> Last updated: 2026-04-15 (session 49 — ablation tree + NDJSON observability)

PLAN.md is current-session work only. Historical session changelogs live in
git log; durable verdicts live in `docs/decisions/README.md`; living architecture
lives in `docs/reference/`.

## Active

- **Re-run seed=42 unsupervised cells under new format** — in-flight Cardinal
  jobs 8557733 (VGAE), 8557737 (GAE), 8557738 (DGI, PD QOSMax) were submitted
  before two late-session changes: (a) `_otel.py` switch to NDJSON formatters,
  (b) lake_root fix pointing paths at `/fs/ess/PAS1266/graphids/dev/rf15`.
  Those three will finish writing pretty-print JSON to
  `/users/PAS2022/rf15/graphids/experimentruns/...`. Plan: let them finish
  (data still useful during current session), re-run under new format post
  training, move or rsync to lake.
- **Calibrate fit/fit-long time budgets on set_01** — `fit-long` is 4 h/gpu
  static; set_01 small VGAE on H100 took ~23 min. Pitzer V100 will be
  different. Re-fit once a few real set_01 runs land.
- **Harvest training-time coefficients** — previous session's TODO still open.
  Now that `run_io.load_traces`/`load_metrics` exist, the harvester script
  has a parser to build on.

## Recently landed (this session)

- **Campaigns subsystem deleted** (−753 LOC): `graphids/campaigns/`
  + `graphids/cli/campaign.py` + `campaigns/` YAML + `tests/campaigns/` +
  all cross-refs (CLAUDE.md, `.claude/rules/config-system.md`, 4 docs in
  `docs/`, monitoring.py OTel `campaign.*` attrs, `__main__.py` register).
  Superseded by an explicit `configs/ablations/` jsonnet tree.
- **`configs/ablations/` tree** — 16 jsonnets across 5 groups (conv_type,
  unsupervised, gat_sampling, gat_loss, fusion) + `_paths.libsonnet`
  helper + `README.md`. Each ablation locks one axis and auto-computes
  `run_dir` + upstream ckpt paths from `(dataset, seed, lake_root)` TLAs.
  Submit calls collapse to `--tla dataset=... --tla seed=...` — no
  `--set` overrides, no explicit ckpt paths.
- **NDJSON observability** — `graphids/_otel.py::wire_file_exporters`
  now passes `formatter=lambda x: x.to_json(indent=None) + "\n"` to
  `ConsoleSpanExporter` and `ConsoleMetricExporter`. `traces.jsonl` and
  `metrics.jsonl` are now true ndjson (one OTel record per line),
  directly consumable by `polars.read_ndjson` and `duckdb.read_json_auto`.
- **`graphids/core/run_io.py`** — polars parser for `metrics.jsonl` +
  `traces.jsonl`. Accepts rendered config dict / run_dir path / file
  path. Flattens histogram/gauge/sum data points into a long-format
  DataFrame; handles empty-file case (returns empty DF with correct
  schema) for runs still in progress.
- **Submit infra**: `fit` + `fit-long` profiles added to
  `configs/resources/submit_profiles.json` (gpudebug 1h / gpu 4h).
  `scripts/slurm/submit.sh` now passes `SBATCH_DEP` env var through as
  `--dependency=` so dependent stages can chain on upstream `afterok`.
- **Ablation runbook + launcher** — `docs/plans/ablation-set_01.md`
  (54 → 51 jobs after deduping baseline GAT with Stage 1's focal cell)
  + `scripts/ablation/launch_set_01.sh` which loops over
  `(seeds, groups, variants)`, captures per-seed upstream jobids, and
  submits dependent stages with `SBATCH_DEP=afterok:<jid>[:<jid>]`.
  Supports `--dry-run` and `--seed <N>` for seed-wave execution.
- **Fair-share diagnosis** — first bulk launch (54 jobs Pitzer gpu) sat
  `PD Reason=Priority` >7 hours. Hypothesized per-user deprioritization,
  cancelled, resubmitted single job — also sat PD. Actual cause: queue
  saturation (test-only start estimate ~2.5 days Pitzer, ~3 days
  Ascend, ~20 days Cardinal batch). Moved to Cardinal debug (1 h
  walltime, `QOSMaxJobsPerUserLimit=2`): jobs start in seconds. Use
  Cardinal debug for any ablation cell that fits under 1 h.

## Recently landed (session 48)

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

## Recently landed (session 47)

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
