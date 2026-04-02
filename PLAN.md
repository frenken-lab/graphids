# KD-GAT Session Plan

> Last updated: 2026-04-02 (session 10 — budget audit + simplification)

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

Observability: `python -m graphids pipeline-status` shows aggregated asset status
(dagster + phase markers + sacct). DeviceStatsMonitor logs CUDA memory stats per step.
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

Orchestrator: SLURM job 46260678. Monitor with `squeue -u $USER` or `sacct`.

**When current run completes, verify:**
1. All 32 assets have `.complete` markers
2. DGI autoencoder (manual job 46265877) completed
3. 3 normal retries succeeded after checkpoint cleanup
4. Curriculum + fusion stages submitted and completed downstream

**Follow-up if needed:**
- Re-run any failed assets: `scripts/submit.sh ablation --assets '<name>' --partition 'set_01|42'`
- Leaf nodes can use direct `sbatch` (no dagster needed)

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

## Next — profile-budget command

The budget module is built on probe values that have never been measured on real
hardware. The critical unknown: **is α > 0?** If not, the throughput ceiling
never exists and the module reduces to `budget = VRAM / bytes_per_node`.

1. **Build `profile-budget` command** — `python -m graphids profile-budget`
   instantiates each model (random weights) × dataset, runs `_probe()`, outputs
   measured γ, α, β, bytes_per_node. See `docs/backlog/profile-budget-command.md`.
2. **Run on GPU** — `scripts/submit.sh profile-budget`. 32 probes, ~2 min.
3. **Replace test estimates** — update `MODEL_PROBES` in `test_budget_matrix.py`
   with measured values.
4. **Decision gate:** if α ≈ 0 for all models → delete throughput ceiling code.
   Budget becomes 5 lines.
5. **Run budget tests** — `scripts/submit.sh tests -k test_budget`
6. **Fix DGI** — set `compile_model: false`, resubmit DGI autoencoder.
7. **Resubmit ab6a75a4** — manual sbatch (dagster retries exhausted).
8. **Verify ablation completion** — all 32 assets.

## Key References

| Doc | Purpose |
|-----|---------|
| `docs/reference/throughput-optimal-batching.md` | Throughput model, sources, corrected design |
| `docs/reference/gnn_throughput_equations.md` | Formal cost model with epistemic status |
| `docs/reference/budget-pipeline-analysis.md` | Old vs new pipeline walkthrough, worker scaling |
| `docs/reference/osc-cluster-memory-limits.md` | Per-partition mem_per_cpu for all 3 clusters |
| `docs/backlog/training-efficiency.md` | Backlog: remaining tiers (prefetch_factor, CPU training) |
| `docs/backlog/profile-budget-command.md` | **New** — profile-budget command spec |
| `docs/backlog/ablation-bug-patterns.md` | 10 bugs from smoke test + ablation, with pattern analysis |
| `docs/decisions/0003-slurm-job-consolidation.md` | **Implemented** — bundle train+test+analyze in one SLURM job |
| `docs/backlog/config-overhaul-remaining.md` | Config overhaul tracker — open items |
| `docs/backlog/per-stage-overrides.md` | Global vs stage-specific overrides (open) |
| `docs/backlog/analyzer-manifest.md` | Manifest ownership (open) |
| `docs/backlog/override-chain.md` | 4-hop override flow — 5 mitigations applied, architecture open |
| `docs/reference/experiment-plan.md` | 17-config ablation matrix |
| `docs/backlog/open-items.md` | All deferred items |
