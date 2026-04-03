# KD-GAT Session Plan

> Last updated: 2026-04-03 (session 15 — orchestrator observability)

## Current State

Pipeline converges at LightningCLI (`train_entrypoint.py` → `run_lightning()`). ConfigResolver
handles cross-field validation + audit trail. SLURM submission via `scripts/submit.sh` works.
Dagster orchestrator runs as CPU SLURM job (not login node).

Each model config is **one dagster asset = one SLURM job** running train→test→analyze
sequentially. Training runs under `set -euo pipefail`; test/analyze are best-effort
(`set +euo pipefail`). Per-phase marker files (`.train_complete`, `.test_complete`,
`.analyze_complete`) written on success.

Asset checks are split: `checkpoint_complete` (blocking) gates downstream assets,
`analysis_complete` (non-blocking) is informational only. Checkpoint checks report
per-phase status (`phase_train`, `phase_test`, `phase_analyze`) in dagster metadata.

Observability: three-layer stack.
- **turm** — live SLURM queue + log tailing (now real-time via `PYTHONUNBUFFERED=1`)
- **orchestrator JSONL** — structured event log per run at
  `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl` (materialize/skip/retry/scale/submit/fail)
- **pipeline-status** — aggregated asset status table (dagster + sacct + phase markers)
  with `--log [FILTER]` and `--follow` for orchestrator event queries
- DeviceStatsMonitor logs CUDA memory stats per step.
  nvitop, reportseff, SlurmTUI installed as complementary tools.

## What this session did (2026-04-02, session 8 — smoke test + ablation launch)

### hcrl_sa smoke test — passed

Ran full pipeline (vgae→gat_normal→gat_curriculum→4 fusion methods), all 7 SLURM
jobs exit 0 in ~25 min total. GAT AUC=0.9995. Found and fixed 4 bugs:
stale test assertions (3), missing `FusionDataModule.test_dataloader` (1).

### set_01 ablation — launched, in progress

Updated `ablation.yaml` with claims 5 (conv type: gatv2 vs gat vs gps) and
6 (unsupervised method: VGAE vs GAE vs DGI). 24 recipe configs expand to 32
unique dagster assets after dedup. Stage-sharing DAG avoids combinatorial explosion.

**Identity hash fix:** Added `model_type` to autoencoder + curriculum identity_keys
so DGI/GAE/VGAE autoencoders get distinct hashes. Added `model_keys` (subset of
identity_keys) to control which keys become CLI overrides vs just hash inputs.

**Ablation status** (2026-04-02 10:15 EDT):

| Status | Count | Assets |
|--------|-------|--------|
| COMPLETED | 3 | `autoencoder_288aba35` (vgae small/GAE), `autoencoder_9ffb88b1` (vgae large), `autoencoder_ff9f9014` (vgae small) |
| RUNNING | 3 | `normal_2bca8cb0` (large), `normal_46ee23eb` (large), `normal_789ca533` (large) |
| PENDING retry | 3 | `normal_2af9d630` (small/focal), `normal_56cc5893` (small/ce), `normal_ab6a75a4` (small/wce) — stale ckpts cleared |
| PENDING new | 2 | `curriculum_bf2a5575` (small), `curriculum_e9354ccd` (large) — upstream done, queued |
| MANUAL submit | 1 | `autoencoder_c479d625` (DGI) — job 46265877, exhausted dagster retries |
| NOT YET SUBMITTED | 20 | 4 autoencoders (ready), 8 curricula (blocked on autoencoders), 8 fusions (blocked on curricula) |

### Bugs found: 8 total

3 patterns identified — see `docs/backlog/ablation-bug-patterns.md` for full log.

| Pattern | Bugs | Root cause |
|---------|------|------------|
| Stale tests after code changes | 1-3 | Test assertions not updated after sessions 6-7 |
| Stale artifacts from prior code | 5, 6 | Copied YAML fields, old checkpoint class_paths |
| Identity keys ≠ model keys | 7, 8 | Identity keys for hashing blindly passed as CLI overrides |

Bug 4 (missing test_dataloader) was standalone.

### Config/override tests — 84/84 passed

All config, merge parity, recipe expansion, CLI routing, and submit.sh tests green.

### Issues updated

| Issue | Status |
|-------|--------|
| `docs/backlog/ablation-bug-patterns.md` | **New** — 8 bugs logged with patterns |

## What this session did (2026-04-01, session 7b — observability)

### Observability layer — existing tools + thin glue code

Researched SLURM/HPC monitoring ecosystem. Decided against turm fork in favor of
installing existing tools + ~200 lines of custom glue.

**Tools installed:**
- nvitop 1.6.2 — real-time GPU monitoring TUI + Python API (in .venv)
- reportseff 2.10.3 — multi-job efficiency reports, replaces seff (in .venv)
- SlurmTUI 0.6.3 — sacct history TUI with Textual (standalone tool)

**Phase markers** (`generate_script()` in `slurm/slurm.py`):
- `.train_complete`, `.test_complete`, `.analyze_complete` written on phase success
- Uses `if cmd; then touch; fi` pattern (not `$?` — immune to line insertions)
- `PHASE_MARKERS` constant in `config/runtime.py`, re-exported via `__init__.py`
- Checkpoint check in `checks.py` reports `phase_train/test/analyze` in dagster metadata
- Legacy runs (no markers) show `-` instead of false failures

**DeviceStatsMonitor** (`_lightning.py`):
- Added as forced callback (like checkpoint/early_stopping)
- Logs CUDA memory stats per step to WandB/CSV automatically
- nvitop's Lightning callback was deprecated + incompatible with pytorch_lightning 2.6.1

**Pipeline status CLI** (`graphids/commands/pipeline_status.py`):
- `python -m graphids pipeline-status [--limit N] [--json]`
- Uses `DagsterInstance.get_asset_records()` batch API (no server needed)
- Deduplicates `get_run_by_id()` calls by run_id
- Derives `run_dir` from checkpoint path using `CKPT_SUBPATH` depth (not hardcoded)
- Rich table output with per-asset status + phase markers + wall time + job ID
- JSON output mode for scripting

**Plan:** `~/plans/observability-plan.md` — full research + architecture + deferred items

### Issues updated

| Issue | Status |
|-------|--------|
| `docs/backlog/observability-remaining.md` | **Partially resolved** — CLI status done, alerting deferred |
| `docs/backlog/slurm-phase-reporting.md (deleted)` | **Resolved** — phase markers implemented (session 7) |

## What this session did (2026-04-01, session 7)

### Wiring audit and contract hardening

Categorized 17 bugs from sessions 5-6: config resolution (6), device management (3),
checkpoint/IO (2), SLURM/infra (3), model contract (1), dagster orchestration (2).
Identified the config→checkpoint→IO wiring path as the dominant bug source.

Audited the wiring path and found 12 fragility points. Fixed 9 (3 critical, 3 high,
3 medium):

**Critical fixes:**
- `merge_yaml_chain` raises `FileNotFoundError` on missing config files (was silent skip)
- `to_override_dict` warns on unmapped upstream assets, raises `KeyError` on unmapped
  model families (was silent drop of upstream checkpoints)
- `checkpoint_path()` delegates to `PathContext` (was parallel path derivation that could diverge)

**High fixes:**
- `to_override_dict` warns on `runtime_overrides` key conflicts (was silent last-write-wins)
- `touch_complete` uses `os.open` + `fsync` on file and directory for NFS durability (was bare `touch`)
- `_flatten_dict` rejects non-scalar values with `TypeError` (was silent `str()` on lists/dicts)

**Medium fixes:**
- `generate_script` uses `set -euo pipefail` before preamble + training
- `lake_root`/`user` read at materialization time, not `build_defs` time (was baked at import)
- Config snapshot applies `LINK_TARGETS` for manual replay reproducibility

### Fusion checkpoint fallback

`best_model.ckpt` never created for fusion RL (bandit/DQN) — only `last.ckpt` exists.
Fixed `run_test_from_spec`, `checks.py`, and `assets.py` to prefer `best_model.ckpt`,
fall back to `last.ckpt`.

### Issues updated

| Issue | Status |
|-------|--------|
| `docs/backlog/slurm-phase-reporting.md (deleted)` | **Resolved** — phase markers implemented |
| `docs/backlog/config-overhaul-remaining.md` | W4 (collision detection) resolved, validation hardened |
| `docs/backlog/override-chain.md` | 5 mitigations added, `OverrideChain` proposal still open |

## Blocking — done

1. ~~Run clean smoke test~~ — **done** (session 8, hcrl_sa, all 7 stages pass)
2. ~~Run config/override tests~~ — **done** (session 8, 84/84 green)
3. ~~Launch ablation~~ — **done** (session 8, set_01 seed 42, 32 assets)

## Active — set_01 ablation in progress

Second orchestrator (job 46276742) running. 3 curriculum jobs running on GPU
(4c0b42ae, d5987709, 24c093ae). Most stages complete — see session 16 audit below.

## What this session did (2026-04-03, session 16 — lake audit + fusion CPU pipeline)

### Lake artifact audit

Audited all 49 run directories under `set_01`. Key findings:

| Finding | Scope | Action |
|---------|-------|--------|
| Train val_acc=96% but test acc=17% | All GAT normal/curriculum | Not a bug — test aggregates 6 subdirs including OOD + excluded attack types. Backlogged: `per-testset-metrics.md` |
| No analysis artifacts for fusion/DGI | All fusion + DGI runs | `ANALYSIS_SUPPORTED_MODELS` only had vgae/gat. Added `dgi`. Fusion blocked by deeper issues (backlogged) |
| No `best_model.ckpt` for Bandit/DQN | 2 RL fusion runs | `automatic_optimization=False` breaks ModelCheckpoint silently. Backlogged in `fusion-analysis-support.md` |
| Fusion only got 50 epochs | All fusion runs | Was `max_epochs: 50`, old tests used 1500. Fixed to 1500, patience 200 |
| ~12 stale orphan directories | Pre-DeviceStatsMonitor runs | From prior config hash changes. Safe to clean up |
| `resource_profile.csv` in ~13/30 | Older runs | Callback added session 14, expected |

### Fusion CPU training pipeline

Fusion models are tiny (35K params) operating on pre-cached 15-D state vectors.
GPU was wasted on 10+ hours of small tensor ops. Split into two phases:

1. **Extract** (GPU, ~2 min): `python -m graphids extract-fusion-states` loads
   VGAE+GAT, runs inference on 150K graphs, saves `{train,val}_states.pt` to disk
2. **Train** (CPU, hours): `FusionDataModule` loads cached states from disk via
   `cached_states_dir` param. No GPU, no upstream model loading.

Changes:
- New command `extract-fusion-states` in `graphids/commands/extract_fusion_states.py`
- `FusionDataModule`: added `cached_states_dir` param, `_load_cached_states()` method
- `fusion.yaml`: `max_epochs: 1500`, `patience: 200`, `accelerator: cpu`
- `fusion.yaml` resources: all methods `gpu_train→cpu_train`, mem 8G/4G
- `clusters.yaml`: added `cpu_train` mode for all 3 clusters
- `topology.py`: fusion stage mode `cpu_train`
- `slurm.py`: CPU jobs get `SKIP_CUDA_CONF=1 SKIP_STAGE_DATA=1`
- `scripts/slurm/extract_fusion_states.sh` + `run_fusion_cpu.sh`

### Extraction job submitted

Job **46311413** — extracts both small and large fusion states:
- Small: `vgae_ff9f9014` + `gat_bf2a5575` → `fusion_states/small_ff9f9014_bf2a5575/`
- Large: `vgae_9ffb88b1` + `gat_e9354ccd` → `fusion_states/large_9ffb88b1_e9354ccd/`

### Issues updated

| Issue | Status |
|-------|--------|
| `docs/backlog/per-testset-metrics.md` | **New** — test_step pools 6 test sets |
| `docs/backlog/fusion-analysis-support.md` | **New** — 3 blockers for fusion analysis |

## Next session — submit fusion CPU training

**Prerequisite:** Verify extraction job 46311413 completed:
```bash
sacct -j 46311413 --format=JobID,State,Elapsed
ls /fs/ess/PAS1266/kd-gat/dev/rf15/set_01/fusion_states/*/fusion_states/
```

**Submit all 8 fusion methods on CPU** (zero GPU, all parallel):
```bash
LAKE=/fs/ess/PAS1266/kd-gat/dev/rf15/set_01
SCRIPT=scripts/slurm/run_fusion_cpu.sh

# Small scale
sbatch --job-name=fusion_bandit_sm   $SCRIPT bandit       small "$LAKE/fusion_small_fusion_82437173/seed_42"
sbatch --job-name=fusion_dqn_sm      $SCRIPT dqn          small "$LAKE/fusion_small_fusion_a7bdee36/seed_42"
sbatch --job-name=fusion_mlp_sm      $SCRIPT mlp          small "$LAKE/fusion_small_fusion_d64ae7a5/seed_42"
sbatch --job-name=fusion_wavg_sm     $SCRIPT weighted_avg small "$LAKE/fusion_small_fusion_79a8454f/seed_42"

# Large scale
sbatch --job-name=fusion_bandit_lg   $SCRIPT bandit       large "$LAKE/fusion_large_fusion_e6219e33/seed_42"
sbatch --job-name=fusion_dqn_lg      $SCRIPT dqn          large "$LAKE/fusion_large_fusion_377a40f4/seed_42"
sbatch --job-name=fusion_mlp_lg      $SCRIPT mlp          large "$LAKE/fusion_large_fusion_0afb6d08/seed_42"
sbatch --job-name=fusion_wavg_lg     $SCRIPT weighted_avg large "$LAKE/fusion_large_fusion_e11ff23f/seed_42"
```

Note: these write into existing run dirs. `lightning_logs/version_N` will increment
past the old 50-epoch runs. Old checkpoints will be overwritten by new training.

**Also check:**
- 3 curriculum GPU jobs still running (4c0b42ae, d5987709, 24c093ae) — should complete soon
- DGI autoencoder c479d625 now gets analysis (embeddings) on next run
- Remaining ablation assets not yet submitted (conv-type variants, loss variants)

## What this session did (2026-04-02, session 9 — training efficiency)

### Diagnosis: GPU utilization 5-22% across graph stages

Synthesized `docs/reference/ablation-resource-profile.md` (profiled from running
ablation) with `docs/reference/cpu_gpu_gnn_training_reference.md` (browser session
research). Root cause: CPU-side `Batch.from_data_list()` collation dominates GPU
compute by 2:1 (GAT) to 16:1 (VGAE). Old budget system filled VRAM regardless
of pipeline capacity.

### Changes made

**Config (applied, affects new jobs):**
- `num_workers: 2` → `6` in all 4 stage YAMLs
- SLURM profiles: `cpus: 4` → `8`, memory bumped for worker RSS
- `clusters.yaml`: added `mem_per_cpu` per partition (all 3 clusters), fixed
  Cardinal partition `batch` → `gpu`
- `resources.py`: validates `mem ≤ cpus × mem_per_cpu` at profile resolution

**budget.py (wired up, needs GPU validation):**
- New module `graphids/core/preprocessing/budget.py` replaces `vram_node_budget`
- Two-point probe measures collation rate (γ), GPU per-node rate (β), and
  kernel overhead (α) to classify regime
- Affine GPU model: `T_gpu = α + β·N` (not constant)
- Throughput budget exists only in collation-dominated regime with α > 0:
  `N_optimal = α / (γ/W - β·mean_nodes)`
- `datamodule.py` and `curriculum.py` wired to call `node_budget()`

**Key correction during session:**
Original design treated T_gpu as constant (25ms regardless of batch size).
User caught the flaw: if utilization is 100% at all worker counts, throughput
can't vary. Corrected to affine model where both T_collation and T_gpu scale
with batch size. The regime (collation vs compute dominated) is batch-size-
independent — determined by per-node rates and worker count.

### What is NOT validated

- **Two-point probe on GPU.** The affine model (α, β) has not been measured on
  real hardware. Needs a SLURM job running the probe at two batch sizes.
- **Throughput budget accuracy.** The `_throughput_budget_nodes` formula is
  derived correctly from the affine model but depends on accurate α, β, γ.
- **Actual GPU utilization improvement.** Regime classification works on CPU
  (tested). Real utilization improvement needs before/after profiling.
- **Tests.** Updated `test_vram_budget.py` parses and imports resolve, but
  can't run on login node. Needs `scripts/submit.sh tests -k test_vram_budget`.

## What this session did (2026-04-02, session 10 — budget audit + simplification)

### Budget module rewrite

Audited budget.py against PyG/Lightning/PyTorch deps. Two library replacements:
- `torch.utils.benchmark.Timer` replaces manual `time.perf_counter` for GPU timing
  (multi-sample median, proper CUDA sync)
- `PrefetchLoader` (PyG) wraps DataLoaders for async H2D via CUDA streams

Then audited the module internals:
- Deleted `regime()` — its 2.0/0.5 thresholds were arbitrary and didn't control
  the budget decision. Replaced with raw `cg_ratio` logged as a continuous value.
- Deleted `CostCoefficients`, `collation_time()`, `gpu_time()`,
  `collation_gpu_ratio()`, `_throughput_budget_nodes()` — dead code or decorative
  wrappers. None were called in the budget decision path.
- Collapsed 8 exports to 3: `BudgetResult`, `_probe`, `node_budget`.
- Tagged every constant as DERIVED / HEURISTIC / FALLBACK with provenance.
- Inlined throughput ceiling math into `node_budget()` with full derivation visible.

### Tests

- Rewrote `test_vram_budget.py` to test through public API with mocked `_probe`
- Added `test_budget_matrix.py`: 611 parametrized tests across 4 datasets ×
  8 model configs × 3 GPU types × 3 worker counts. Uses realistic values from
  actual config files (model YAMLs, cache_metadata.json, resource profiles).
- Tests verify: budget within VRAM, reasonable batch counts, monotonicity
  (more VRAM → bigger budget, bigger model → smaller budget, more workers →
  budget doesn't shrink), regime properties, GPS quadratic path.

### Ablation diagnosis

- Bug 9: DGI `torch.compile` inductor crash. Fix: `compile_model: false` in
  `config/models/dgi/base.yaml`. Blocks entire DGI ablation branch.
- Bug 10: phantom `resume_ckpt` from stale orchestrator code (already fixed in
  session 9 commit ebd7e1f, but running orchestrator uses old code).
- Both logged to `docs/backlog/ablation-bug-patterns.md`.

### Ablation progress

Orchestrator further along than expected — fusion jobs now submitting.
Running: `normal_56cc5893`, `fusion_0afb6d08`, `fusion_d64ae7a5`.
Pending: 3 more fusions. Blocked: DGI branch (bug 9) + ab6a75a4 (bug 10).

### Graphcore source material

Researched 14 pages across PyG tutorials, popXL tutorials, supplementary docs.
Key finding: Graphcore docs provided the conceptual framework (fixed budgets,
packing vs padding, efficiency metrics). The affine cost model, regime
classification, and two-point probe are original to our design.

## What this session did (2026-04-02, session 11 — CLI command audit + hygiene)

### Command audit

Audited all 14 commands in `graphids/commands/` by functional logic and
dependencies. Identified 4 natural clusters:

| Cluster | Commands | Pattern |
|---------|----------|---------|
| Spec runners | train-from-spec, test-from-spec, analyze-from-spec | Thin delegates, dagster→SLURM transport |
| Core / GPU compute | analyze, profile-training, probe-budget | Model instantiation, GPU work, artifacts |
| Data / preprocessing | stage-data, rebuild-caches, test-preprocessing | CPU-only dataset I/O |
| Ops / observability | pipeline-status, job-stats, submit-profile | Query SLURM/dagster, format output |

### Landscape folded into analyze

`landscape.py` (48 lines) was a wrapper around `analyze` with fixed flags.
Folded into `analyze.py` as a subcommand: `python -m graphids analyze landscape ...`.
Deleted `landscape.py`, removed from `_COMMAND_MODULES`. Updated
`submit_profiles.yaml` landscape command to route through analyze.

### profile-budget rewritten to follow convention

Original version did bespoke YAML merge + `importlib._import_class` — broke
project convention. Rewritten to use `jsonargparse.add_subclass_arguments(
pl.LightningModule)` + `instantiate_classes()` — same path as LightningCLI.

### Command renames (proposed — applied in session 12)

- `profile` → `job-stats` (sacct resource report, not profiling)
- `profile-training` → `profile` (actual PyTorch Profiler)
- `profile-budget` → `probe-budget` (hardware cost model measurement)

### Handrolled code fixes (6 items)

| File | Fix |
|------|-----|
| `rebuild_caches.py` | Hardcoded `ALL_DATASETS` tuple → `dataset_names()` from config |
| `rebuild_caches.py` | Hardcoded `/fs/scratch/PAS1266/...` marker → derived from `KD_GAT_SCRATCH` env var |
| `submit_profile.py` | `_load_submit_profiles()` → `read_yaml()` from `yaml_utils` |
| `submit_profile.py` | Removed `ResourceSpec` misuse (constructed+discarded for time validation) |
| `stage_data.py` | Hardcoded fallback paths → fail fast if `KD_GAT_SCRATCH`/`KD_GAT_DATA_ROOT` unset |
| `stage_data.py` | Hand-rolled `set(argv)` parsing → `argparse.ArgumentParser` |

### SLURM log path leak fixed

`scripts/submit.sh`, `_preamble.sh`, `_epilog.sh` all hardcoded `slurm_logs/`
relative to project root (NFS). Fixed to derive `SLURM_LOG_DIR` from env vars
(`KD_GAT_SLURM_LOG_DIR` → `KD_GAT_LAKE_ROOT/slurm` → `experimentruns/slurm`),
matching the Python `SLURM_LOG_DIR` constant in `runtime.py`. Verified shell
and Python produce identical paths.

## What this session did (2026-04-02, session 12 — bug fixes + backlog cleanup)

### Ablation status (from sacct, not pipeline-status)

Native `pipeline-status` showed only 2 SUCCESS — **wrong**. Dagster never
updates asset status after SLURM completion. sacct shows the real picture:

| Stage | Completed | Failed |
|-------|-----------|--------|
| autoencoder | 3 (GAE small, VGAE large, VGAE small) | 1 (DGI — torch.compile crash) |
| normal | 4 (all 4 hashes) | 1 (ab6a75a4 — phantom resume_ckpt) |
| curriculum | 2 (small, large) | 0 |
| fusion | 5 (all 5) | 0 |
| **Total** | **14** | **2** |

Orchestrator (job 46260678) still RUNNING at 18h but idle — no child jobs
in queue. All completable work finished.

### Bug fixes (3 categorical + 3 defensive)

**Categorical: unguarded `torch.compile`** — copy-pasted bare call in 3
model `_build()` methods with no error handling. Inductor backend can fail
on unusual FX graphs (DGI's dual-encoder structure).

| Fix | File |
|-----|------|
| `try_compile()` helper | `_training.py` — catches exception, logs warning, falls back to eager |
| Replace 3 inline copies | `dgi.py`, `vgae.py`, `gat.py` — all call `try_compile()` |
| DGI compile disabled | `config/models/dgi/base.yaml` — `compile_model: false` (24.6K params, no benefit) |

**Defensive: unguarded external calls** (from subagent audit of codebase):

| Fix | File |
|-----|------|
| CUDA guards on `_probe()` | `budget.py` — `torch.cuda.*` calls guarded by `model.device.type == "cuda"` |
| ImportError handling | `__main__.py` — `importlib.import_module` → clean `SystemExit` on failure |
| CUDA guard on `empty_cache` | `tasks.py` — consistent with defensive pattern elsewhere in file |

### Command renames applied

Session 11 proposed, session 12 applied:
- `profile` → `job-stats`, `profile-training` → `profile`, `profile-budget` → `probe-budget`
- Updated: `__main__.py`, `submit_profiles.yaml`, `submit.sh`, `CLAUDE.md`

### Backlog cleanup (7 subagents, 0 failures)

| Task | Result |
|------|--------|
| Delete resolved backlog files | -2 files (ablation-bug-patterns → `reference/`, profile-budget-command) |
| Delete `edge_to_tensor` | Confirmed 0 callers, deleted from `features.py` |
| Fix broken test_features imports | Already clean (prior session) |
| Orphaned YAML audit | -9 files, -2 dirs (`schema/`, `overrides/`). `matrix/axes.yaml` kept. 4 docs updated |
| Dead lr/weight_decay audit | Not in GAT/DGI `__init__` — consumed by base class `getattr`. No-op |
| open-items.md audit | 20→12 items. 4 resolved in code (curriculum DataLoader, GPS budget, dataset staging, T_max) |
| CLAUDE.md command table | Updated with renames + landscape folded into analyze |

### New backlog item

`docs/backlog/pipeline-status-stale.md` — `pipeline-status` queries dagster
`AssetRecord` which never updates after SLURM completion. Fix: reconcile
with sacct (option A, ~20 lines) or fix dagster polling (option B).

## What this session did (2026-04-03, session 15 — orchestrator observability)

### Structured orchestrator event log

Dagster orchestrator was a black box — decisions (materialize, skip, retry,
scale resources, submit, poll, succeed, fail) produced no visible output.

**structlog JSONL** — configured in `definitions.py` at module scope. Under
SLURM: `JSONRenderer` → `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`. Without
SLURM: `ConsoleRenderer` (validation, `dg list defs`). Process-global — all
structlog loggers in `resolve.py` and `slurm.py` automatically route to JSONL.

**Events added (8 total):**

| Event | File | Level | Key Fields |
|-------|------|-------|------------|
| `orchestrator_init` | `component.py` | info | recipe, num_assets, datasets, seeds |
| `asset_start` | `assets.py` | info | asset, dataset, seed, retry |
| `asset_skip` | `assets.py` | info | asset, ckpt |
| `resource_scaled` | `assets.py` | info | asset, retry, old/new mem+time |
| `asset_complete` | `assets.py` | info | asset, state, job_id, wall_time |
| `asset_failed` | `assets.py` | error | asset, state, job_id |
| `submitted` (enriched) | `slurm.py` | info | job_id, partition, time, mem, gres |
| `slurm_poll` | `slurm.py` | info | job_id, state, prev |

(Plus existing `config_resolved` in `resolve.py` — unchanged.)

### pipeline-status --log CLI

Added `--log [FILTER]` mode to `pipeline-status` for querying orchestrator JSONL.
Filters: `all`, `failures`, `retries`, `completions`, `submissions`, `polls`.
`--follow` / `-f` for live tailing. `--log-file` for specific file (default: latest).

### turm fix

`export PYTHONUNBUFFERED=1` in `_preamble.sh`. Fixes Python stdout buffering
under SLURM — turm and `tail -f` now see output in real-time instead of at exit.

### Spec file preservation

Removed spec file deletion from `slurm.py` `finally` block. Specs persist in
`{SLURM_LOG_DIR}/specs/` as audit trail of what was submitted to SLURM.

### Issues updated

| Issue | Status |
|-------|--------|
| `docs/backlog/observability-remaining.md` | **Resolved** — all high items done |

## Next

### ~~Budget module audit~~ — DONE (session 16)

Full equation-by-equation audit against research docs + first principles.
See `docs/reference/budget-cost-model-audit.md`.

**5 bugs found, all fixed:**

| Bug | Fix |
|-----|-----|
| Stale docstrings (module + node_budget) | Rewritten to match code |
| γ measurement contaminated by GPU state | `torch.cuda.synchronize()` + `gc.collect()` + 3-sample median |
| cg_ratio uses forward-only β | Now uses β × backward_multiplier (training-adjusted) |
| num_steps truncates (drops 10-15% of data/epoch) | `math.ceil` in datamodule.py + curriculum.py |
| No throughput floor guard | Added: `budget = clamp(floor, mem_ceiling)` |

**Key mathematical result:** budget = mem_budget is correct for all
current configs. Throughput floor (minimum batch to amortize α) exists
but is always << mem_budget (max floor ≈ 87K nodes vs min ceiling ≈ 54K
nodes for GAT large on V100). The floor is now computed and stored in
`BudgetResult.throughput_floor` as a guard.

**Plotting script:** `scripts/plot_budget.py --example` generates 14
plots: throughput curves, regime heatmap, budget comparison, 3-panel
deep dives (throughput + VRAM + GPU util) per model.

**807 tests pass**, 2 skipped (GPU-only). Probe-budget SLURM job
submitted (46275772) to validate on GPU with new γ measurement.

### Ablation follow-up — 15/32 completed, 17 remaining

**Status (2026-04-02, session 14):**

| Stage | Done | Remaining | Blockers |
|-------|------|-----------|----------|
| Autoencoders | 3 (vgae small, vgae large, GAE) | 1 DGI + ~2 conv-type variants | DGI: compile fix committed (85a7f1c) |
| Normals | 5 | 1 (ab6a75a4) | phantom resume_ckpt: fix committed (ebd7e1f) |
| Curricula | 2 (small/large ce, used by fusion pipeline) | ~6 ({focal,wce} × {small,large} + 2 conv-type) | Upstream done — never submitted |
| Fusions | 5 (4 small + 1 large) | ~3 (remaining large) | Upstream done — never submitted |

Orchestrator (46260678) went idle after completing fusion pipeline branch
(autoencoder→normal→curriculum→fusion with default ce). Standalone
curriculum/normal ablation variants and large-scale fusions never queued.

**Resource changes (this session):**
- `mem` removed from GPU training profiles — derived as `cpus × mem_per_cpu`
  at resolution time (72G on Pitzer, 31G on Ascend, 72G on Cardinal).
  Was hardcoded 40-52G, wasting 20-30G per job.
- DGI time limit 2h → 4h (matches VGAE, no runtime evidence for DGI on set_01).
- Fixed 3 invalid fusion profiles: large bandit/dqn `cpus: 2→3` (24G exceeded
  2×9292=18G), large weighted_avg `cpus: 1→2` (10G exceeded 1×9292=9G).
- Fusion keeps explicit `mem` (workers: 0, needs far less than CPU-proportional).

**To finish:**
1. Relaunch orchestrator — `scripts/submit.sh ablation`. All bug fixes and
   resource changes are committed. Should submit remaining 17 assets.
2. Or manual sbatch for DGI (c479d625) and normal (ab6a75a4) first, then
   relaunch for the rest.

### probe-budget on GPU

Command is built and renamed (`probe-budget`). Needs GPU run.

1. **Run on GPU** — `scripts/submit.sh probe-budget`. 32 probes, ~2 min.
2. **Replace test estimates** — update `MODEL_PROBES` in `test_budget_matrix.py`
   with measured values.
3. **Decision gate:** if α ≈ 0 for all models → delete throughput ceiling code.
   Budget becomes 5 lines.

### SLURM validation of all session 11-12 changes

```bash
scripts/submit.sh tests -k test_overrides
scripts/submit.sh tests -k test_config
scripts/submit.sh tests -k test_budget
scripts/submit.sh tests -k test_smoke
```

### ~~Fix pipeline-status~~ — DONE (session 13)

sacct reconciliation implemented. Phase markers now work via filesystem fallback.

### ~~Resource profiling system (7 dimensions)~~ — Phase 1+2 DONE (session 14)

Phase 1+2 implemented. Phase 3+4 deferred until after first campaign with callback active.

**Phase 1 — ResourceProfileCallback + edge-aware margin:**
- `ResourceProfileCallback` in `_lightning.py` — forced callback, logs per-step
  VRAM + batch stats (num_nodes, num_edges, cuda_peak_mb, host_rss_mb, step_time_ms)
  to `{run_dir}/resource_profile.csv` every 50 steps
- Edge-aware budget: `_probe()` now solves 2×2 system (vram = A·N + B·E) at two
  batch sizes for per-node (A) and per-edge (B) VRAM cost. `node_budget()` reads
  `edge_count.p95` from `cache_metadata.json` and computes
  `effective_bpn = bytes_per_node + bytes_per_edge × edges_per_node_p95`

**Phase 2 — backward multiplier + KD + fusion + compile:**
- Backward multiplier: `_probe()` runs one forward+backward step via
  `torch.autograd.backward()` to measure real gradient memory ratio. Falls back
  to `_GRAD_MULTIPLIER=2` when `_step` unavailable
- KD teacher reservation: `node_budget()` subtracts estimated teacher VRAM
  (params × 2.5) from free VRAM before sizing budget
- Fusion pre-flight: `FusionDataModule.setup()` warns if combined VGAE+GAT
  models use >85% of VRAM after loading
- Compile status: `BudgetResult.is_compiled` records `torch.compile` state

**New BudgetResult fields:** `bytes_per_edge`, `edges_per_node_p95`,
`backward_multiplier`, `teacher_vram_bytes`, `is_compiled`

**New _probe() return:** 6-tuple `(bytes_per_node, bytes_per_edge,
backward_multiplier, gamma, alpha, beta)`. All callers updated (budget.py,
profile_budget.py, test_vram_budget.py, test_budget_matrix.py).

**Test fixes:** budget matrix test bounds corrected (graphs_per_batch
upper bound too tight for small graphs on large GPUs), monotonicity test
now compares `mem_budget` not `budget` (regime switches break final-budget
monotonicity but VRAM ceiling is always monotonic with model size).
808 tests pass, 2 skipped (GPU-only).

**Phase 3+4 deferred** — calibration analyzer + auto-feedback. Needs one
campaign with callback active to produce `resource_profile.csv` data.

### Training efficiency (next campaign)

1. Add `prefetch_factor` parameter (~10 lines + YAML)
2. Per-model worker count (YAML only, after profiling — informed by resource profile data)
3. CPU training spike for autoencoders (deferred)

### KD pipeline E2E test

Minimal wiring test (option C from `kd-untested.md`) before writing paper
claims. Then add KD to ablation recipe.

## Key References

| Doc | Purpose |
|-----|---------|
| `docs/reference/throughput-optimal-batching.md` | Throughput model, sources, corrected design |
| `docs/reference/gnn_throughput_equations.md` | Formal cost model with epistemic status |
| `docs/reference/budget-pipeline-analysis.md` | Old vs new pipeline walkthrough, worker scaling |
| `docs/reference/osc-cluster-memory-limits.md` | Per-partition mem_per_cpu for all 3 clusters |
| `docs/reference/ablation-bug-patterns.md` | Bug patterns from smoke test + ablation — prevention guide |
| `docs/backlog/training-efficiency.md` | Backlog: remaining tiers (prefetch_factor, CPU training) |
| `docs/backlog/resource-profiling-plan.md` | 7-dimension resource profiling: callback + probe + calibration |
| `docs/decisions/0003-slurm-job-consolidation.md` | **Implemented** — bundle train+test+analyze in one SLURM job |
| `docs/backlog/config-overhaul-remaining.md` | Config overhaul tracker — open items |
| `docs/backlog/per-stage-overrides.md` | Global vs stage-specific overrides (open) |
| `docs/backlog/analyzer-manifest.md` | Manifest ownership (open) |
| `docs/backlog/override-chain.md` | 4-hop override flow — 5 mitigations applied, architecture open |
| `docs/reference/experiment-plan.md` | 17-config ablation matrix |
| `docs/backlog/open-items.md` | All deferred items (12 remaining) |
