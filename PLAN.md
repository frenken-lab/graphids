# GraphIDS Session Plan

> Last updated: 2026-04-15 (session 50 — pipeline route deleted, single route remains)

## This session (50)

- **Deleted the pipeline route.** `orchestrate/run.py`, `orchestrate/planning.py`,
  `orchestrate/resolve.py`, `cli/pipeline.py` gone. `orchestrate/config.py`
  trimmed to `ResolvedConfig` + `InstantiatedRun` only (PipelineConfig,
  StageConfig, TrainingRunConfig, KDEntry, PipelineResult removed).
  `config/topology.py` reduced to import-time jsonnet-tree validation +
  dataset catalog + path helpers (PathContext, compute_identity_hash,
  StageDef/Topology model deleted). `configs/matrix/topology.json`
  shrunk to just a stage-name list. `docs/reference/kd-pipeline.md` and
  `observability-data-layers.md` deleted as stale. ADRs 0003 / 0006 /
  0007 / 0009 annotated as (partially) superseded.
- **Cluster-aware single `fit` profile.** `fit` + `fit-long` in
  `submit_profiles.json` collapsed to one `fit` with
  `partitions.<cluster>.{short,long}` + `time_short` / `time_long`.
  `submit-profile` CLI gained `--cluster` (default `pitzer`) and
  `--length` (default `long`); patches `profile.partition` / `profile.time`
  from the partitions map before falling through to `_resolve_resources`.
  `scripts/run` default `--cluster=pitzer` (or `$GRAPHIDS_CLUSTER`);
  `--smoke` sets `length=short`; always passes `--clusters=$CLUSTER`.
  `--walltime HH:MM:SS` added so submissions can right-size below the
  profile's default. `stage_profiles` composition block and the
  pipeline-only completion helpers (`_literal_field_values`,
  `_complete_conv_type`, `_complete_loss_fn`, `_complete_fusion_method`)
  removed with the pipeline route.
- **Two-point budget probe.** `graphids/core/data/budget.py::probe`
  rewritten to run fwd+bwd at two batch sizes (2k + 20k nodes) and take
  the slope `(bwd_big - bwd_small) / (nodes_big - nodes_small)` as
  `bpn_node`. Fixed overhead (cuDNN workspaces, optimizer state, KD
  teacher, allocator baseline) drops out into the implicit intercept —
  no more `resident` subtraction. Validated on set_01/vgae/H100:
  `bpn_node` fell from 25,677 B/node to 9,019 B/node (−65%); budget
  rose from 3.29 M nodes to 10.47 M (+218%). Safety margin bumped
  0.85 → 0.95 in `settings.py` (two-point slope's intercept already
  covers fixed costs). Dead `batch_size: 8192` removed from
  `configs/stages/{autoencoder,supervised}.jsonnet` (unused on the
  dynamic-batching path — sampler uses probe budget). `critical-constraints.md`
  gained a "Two-point probe" bullet.
- **Fusion CPU path.** New `fit-cpu` submit profile (cpus=16, mem=32G,
  mode=cpu) with cluster-indexed partitions (Pitzer `debug-cpu`/`cpu`,
  Cardinal `debug`/`cpu`, Ascend `debug`/`cpu`). `scripts/run --cpu`
  flag selects it. New `graphids/_cpu.py::configure_cpu_threads()`
  reads `SLURM_CPUS_PER_TASK` and pins `torch.set_num_threads(N)` +
  `torch.set_num_interop_threads(1)` + `OMP_NUM_THREADS` + `MKL_NUM_THREADS`
  at process start (called from `cli/training.py::_prepare` right after
  `ensure_spawn()`). Idempotent via module flag.
- **Tests** — `test_config.py` shed TrainingRunConfig/KDEntry cases;
  `test_submit_profile.py` shed pipeline-composition regressions;
  `test_cli_routing_smoke.py` dropped the `pipeline-run` expectation;
  `test_budget_matrix.py` fixed the stale `cache_dir` patch (budget.py
  no longer reads cache metadata). All 26 non-training tests green.
- **Docs swept** — `CLAUDE.md`, `.claude/rules/config-system.md`,
  `.claude/rules/slurm-hpc.md`, `docs/responsibilities.md`,
  `docs/reference/orchestration.md`, `docs/reference/config-architecture.md`,
  `docs/reference/write-paths.md`, `configs/CONFIG_REFERENCE.md`,
  `graphids/config/VALIDATION_CHECKLIST.md`,
  `.github/copilot-instructions.md` rewritten to describe the single route.

Multi-stage chains are now a bash loop over `scripts/run <preset>` with
`SBATCH_DEP=afterok:<jid>` deps — `scripts/ablation/launch_set_01.sh`
already does this. Orchestration to rebuild: TBD, next session.

## Handoff — next-session triage

**Running at commit time** (Cardinal `debug`, `--smoke --walltime 0:30:00`):

| Job ID | Preset | Dataset | Seed |
|---:|---|---|---:|
| 8568397 | `configs/ablations/unsupervised/gae.jsonnet` | set_01 | 42 |
| 8568398 | `configs/ablations/unsupervised/dgi.jsonnet` | set_01 | 42 |

**First things to check next session**:

1. `squeue --me -M cardinal` / `sacct -M cardinal -j 8568397,8568398
   --format=JobID,State,ExitCode,Elapsed,MaxRSS -P` — expect COMPLETED
   0:0, elapsed ~13 min (matches prior VGAE calibration).
2. Pull probe numbers from stderr:
   `grep budget_probed /fs/ess/PAS1266/graphids/slurm_logs/graphids-{gae,dgi}_{8568397,8568398}.err`.
   Expect similar correction ratio to VGAE (bpn_node falls ~3×;
   budget rises ~3×).
3. GPU/VRAM summary via `graphids.core.run_io.load_metrics(run_dir)` on
   `/fs/ess/PAS1266/graphids/dev/rf15/set_01/ablations/unsupervised/{gae,dgi}/seed_42`.
   `ml.gpu.utilization_pct`, `ml.cuda.allocated_peak_mb`, `ml.batch.duration_s`
   are the columns of interest. Tag each query with a time cutoff — metrics.jsonl
   is append-only, so prior runs accumulate in the same file.
4. After both finish: compare train/val loss curves across VGAE vs GAE vs DGI
   on set_01 (`train_loss`, `val_loss` metric rows). That's the ablation's
   actual output.

**Reference for VGAE calibration pair** (same dataset, same seed, pre- vs
post-probe change):

| | 8567784 (single-point, 0.85) | 8568190 (two-point, 0.95) |
|---|---:|---:|
| bpn_node (B/node) | 25,677 | 9,019 |
| budget_nodes | 3,290,292 | 10,469,479 |
| Wall | 12:53 | 12:46 |
| Allocated peak VRAM | 16,649 MB | 16,649 MB |
| GPU util mean | 99.26% | 98.89% |

VRAM didn't move because the benign train split (2.64 M nodes) already
fit in one batch under *both* budgets — set_01 is dataset-capped, not
hardware-capped. The probe-cal win will show up on GAT (unfiltered train,
~5× more nodes) and on set_02/03/04.

PLAN.md is current-session work only. Historical session changelogs live in
git log; durable verdicts live in `docs/decisions/README.md`; living architecture
lives in `docs/reference/`.

## Active (carried from prior sessions)

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
