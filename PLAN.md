# GraphIDS Session Plan

> Last updated: 2026-04-04 (session 18 — fusion refactor + reward bake-in)

## What this session did (2026-04-04, session 18 — fusion refactor)

Audited `graphids/core/models/` with a focus on fusion complexity. Found a
runtime-crash bug in KD training, 4 tiers of dead code, and 5 reward-shaping
coefficients exposed as kwargs that the paper fixes as methodological
constants. Executed all cuts in one pass. **Net −200 LOC across source,
tests, configs, and docs.**

### Bug fix (Tier 1)

**`teacher_on_device` was missing `@contextlib.contextmanager`** — plain
generator function used as `with teacher_on_device(self, device):` in
`vgae.py:427` and `gat.py:269`. Any VGAE/GAT KD training crashes on the
first step with `TypeError: 'generator' object does not support the context
manager protocol`. Confirmed at runtime. One-line fix. Explains part of
issue #25 (KD pipeline never tested end-to-end).

### Dead code cuts

| Tier | Cuts | Source |
|---|---|---|
| 1 | `teacher_on_device` decorator fix; delete `BanditFusionModule.regret_stats`, `_total_reward`/`_total_steps` trackers, `_store_buffer` wrapper; delete `DQNFusionModule.store_experiences_batch` wrapper, `buffer_size_current` property | −25 |
| 2 | Collapse `fusion_features.py` registry: drop `FusionFeatureExtractor` Protocol + `@runtime_checkable`, `FeatureLayout` getter functions. Replace with module-level `STATE_DIM`, `LAYOUT`, `EXTRACTORS` constants. Update 5 source call sites + 12 test references | −65 |
| 3 | Delete DQN `target_network`, `target_update_freq`, `update_counter`, `ReduceLROnPlateau` scheduler, `gamma`/`scheduler_patience` init args — γ=0 makes the target network unread and `configure_optimizers()` never returned the scheduler. Replace `loss.backward()` with Lightning-idiomatic `self.manual_backward(loss)` | −45 |
| 4 | Delete `safe_load_checkpoint` pre-flatten migration guard — `scripts/migrate_checkpoints.py` doesn't exist; error message was a stale reference | −15 |
| 5 | Bake reward shaping coefficients into module-level constants (see below) | −57 |

### Reward-coefficient bake-in (Tier 5)

Audited the 7 reward-shaping kwargs in `FusionRewardCalculator`:
`reward_correct`, `reward_incorrect`, `confidence_weight`,
`combined_conf_weight`, `disagreement_penalty`, `overconf_penalty`,
`balance_weight`. Evidence:

- **0 YAML sites** set them (bandit.yaml/dqn.yaml only pass `vgae_weights`)
- **0 recipes** sweep them (ablation.yaml sweeps `loss_fn`, `conv_type`, `variational`, `fusion_method` only)
- **0 issues** or plans mention reward shaping as a future ablation
- **Paper (`kd-gat-paper/methodology.md §Stage 3`)** presents the reward
  equation with ±3.0 as inline constants and says *"Both DQN and bandit use
  this identical reward"* — reward is a fixed methodological choice, not
  an ablation axis
- **Ablation chapter** lists three axes: KD, GAT training, fusion method
  comparison — nothing on reward shaping
- **Only consumer:** one differential test that passed non-default values
  purely to verify they affect output

Replaced kwargs with 7 module-level `_CONSTANT` names in
`fusion_reward.py`, pointing at the paper's equation. Unknown kwargs now
rejected at construction time. Deleted the no-longer-viable
`test_reward_coefficients_actually_used` test; coverage of the one
surviving tunable (`vgae_weights`) is preserved by
`test_derive_scores_uses_vgae_weights`.

### Files touched (session 18)

**Source:** `core/models/fusion/{bandit,dqn,fusion_baselines,fusion_features,fusion_reward,__init__}.py`, `core/models/{_training,__init__}.py`, `core/__init__.py`, `core/preprocessing/datamodule/fusion.py`

**Tests:** `tests/core/models/test_fusion.py`, `tests/test_integration.py`

**Config/docs:** `config/fusion/methods/dqn.yaml`, `config/CONFIG_REFERENCE.md`, `PLAN.md`

### Verification

- AST parse clean on all 12 touched Python files
- Import chain resolves end-to-end; all 4 fusion modules instantiate with the new constants
- `pytest tests/core/models/test_fusion.py tests/test_integration.py::TestDecisionThreshold` → **17/17 passing**
- Attribute assertions verify removed names (`dqn.target_network`, `dqn.scheduler`, `dqn.gamma`, `bandit.regret_stats`, etc.) no longer exist
- `teacher_on_device` context manager usable as `with` block (verified at runtime)
- Full test suite on SLURM: not run (login node — run via `scripts/submit.sh tests`)

## Current state

Pipeline converges at LightningCLI (`train_entrypoint.py` → `run_lightning()`).
`ConfigResolver` handles cross-field validation + audit trail; override chain
collapsed 9→3 per ADR 0009 (commit `0837c04`). SLURM submission via
`scripts/submit.sh`. Dagster orchestrator runs as a CPU SLURM job, not login.

Each model config is **one dagster asset = one SLURM job** running
train → test → analyze sequentially. Training under `set -euo pipefail`;
test/analyze best-effort. Per-phase marker files (`.train_complete`,
`.test_complete`, `.analyze_complete`) written on success.

Asset checks split: `checkpoint_complete` (blocking) gates downstream,
`analysis_complete` (non-blocking) informational. Phase status surfaced in
dagster metadata (`phase_train`/`phase_test`/`phase_analyze`).

**Observability:** three layers.
- `turm` — live SLURM queue + log tailing (`PYTHONUNBUFFERED=1` gives real-time)
- Orchestrator JSONL — structured events per run at
  `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`
- `pipeline-status` — dagster + sacct + phase markers aggregate, with
  `--log [FILTER]` and `--follow`

Run records: `run_record.json` sidecar per run (atomic write, Pydantic
schema) → DuckDB catalog rebuildable via `rebuild-catalog`.

## Active

### Ablation follow-up — 15/32 completed, ~17 remaining

| Stage | Done | Remaining | Blocker |
|---|---|---|---|
| Autoencoders | 3 (VGAE small/large, GAE) | 1 DGI + ~2 conv-type variants | DGI compile fix committed (`85a7f1c`) |
| Normals | 5 | 1 (`ab6a75a4`) | phantom `resume_ckpt` fix committed (`ebd7e1f`) |
| Curricula | 2 (small/large ce — drives fusion) | ~6 (`{focal,wce} × {small,large}` + 2 conv-type) | upstream done, not submitted |
| Fusions | 5 (4 small + 1 large) | ~3 (remaining large) | upstream done, not submitted |

Previous orchestrator (`46260678`) went idle after completing the fusion
branch. Standalone curriculum/normal ablation variants and large-scale
fusions never queued.

**To finish:**
1. Relaunch orchestrator: `scripts/submit.sh ablation`. All fixes and
   resource changes are committed — should submit remaining 17 assets.
2. Or manual `sbatch` for DGI (`c479d625`) and normal (`ab6a75a4`) first,
   then relaunch for the rest.

### Fusion CPU training pipeline

Fusion models are tiny (35K params) on pre-cached 15-D state vectors; GPU
was wasted. Split into two phases:
1. **Extract** (GPU, ~2 min): `python -m graphids extract-fusion-states`
   loads VGAE+GAT, runs inference on 150K graphs, saves
   `{train,val}_states.pt` to disk.
2. **Train** (CPU, hours): `FusionDataModule` loads cached states via
   `cached_states_dir`.

Extraction job **46311413** submitted for small + large states.
Verification command:

```bash
sacct -j 46311413 --format=JobID,State,Elapsed
ls /fs/ess/PAS1266/kd-gat/dev/rf15/set_01/fusion_states/*/fusion_states/
```

Once extraction is verified, submit 8 fusion methods on CPU (all parallel,
zero GPU) via `scripts/slurm/run_fusion_cpu.sh`. Note: these write into
existing run dirs; `lightning_logs/version_N` will increment past the old
50-epoch runs.

### probe-budget on GPU

Command built + renamed. Needs one GPU run (`scripts/submit.sh probe-budget`,
32 probes, ~2 min). **Decision gate:** if α ≈ 0 for all models → delete
throughput ceiling code, budget becomes 5 lines.

### SLURM test validation

```bash
scripts/submit.sh tests -k test_resolver
scripts/submit.sh tests -k test_config
scripts/submit.sh tests -k test_budget
```

## Next (not blocked)

- **KD pipeline E2E test** — minimal wiring test before paper claims, then
  add KD to ablation recipe.
- **Training efficiency tier 2** — `prefetch_factor` parameter (~10 lines +
  YAML), per-model worker count (YAML only, after resource profile data).
- **CPU training spike for autoencoders** — deferred; needs evidence from
  resource profile before committing.
- **Fix deferred test issues** from session 17 audit:
  - `TestDecisionThreshold` (3 tests in `test_integration.py`) — fragile on
    unseeded random state; rewrite as monkeypatched deterministic test.
  - `test_fusion.py` lazy in-function imports + invalid `# noqa:` directive
    — mechanical cleanup.

## What this session did (2026-04-04, session 17 — tests audit)

### Tests audit, compaction, and rules

Audited all 18 test files. Major cleanup in 4 passes:

| Pass | Result |
|---|---|
| Drift fixes | Deleted `tests/orchestrate/__init__.py`; removed stale `graphids/pipeline/*` ruff ignores; collapsed 4 markers → 2 (`slow`, `slurm`); dropped spurious `slurm` marker from 7 CPU-only tests; renamed `graphids/commands/test_from_spec.py` → `run_test_from_spec.py` (prevents pytest collection); fixed `CLAUDE.md` CLI architecture doc drift (`_lightning.py` not `cli.py`) |
| Duplication delete | Deleted `tests/test_smoke.py` (subset of `test_gat.py`/`test_vgae.py`); dropped 2 meta-tests; trimmed `conftest.py::base_cfg` from 30 → 11 fields; replaced 14 bare `Exception` with `ValidationError` in `test_config.py`; fixed silent `pytest.skip` on missing recipe files |
| test_overrides split | 717-line `tests/orchestrate/test_overrides.py` split into 5 focused files: `test_yaml_utils.py`, `test_recipe_expand.py` (consolidated with `test_recipe_expand_kd.py`), `test_resolver.py`, `test_validation_rules.py`, `test_kd_teachers.py` |
| Harmful-test delete | Rewrote `test_budget_matrix.py` — deleted 72-instance formula-mirror test that re-implemented `budget.py`'s math; replaced `MODEL_PROBES` (stale hardware measurements from 2026-04-03) with generic archetypes; kept monotonicity + memory-bound property tests. Rewrote `test_edge_aware_margin` in `test_vram_budget.py` as a differential test. Deleted 15 Pydantic-semantic tests from `test_config.py` (tests the framework, not the code). Deleted 6 file-existence loop tests (duplicates `topology.py` import-time assertions). |

**Result:** 391 → 204 tests collected cleanly. All touched files ruff-clean.
Test suite is now resistant to budget-formula refactors, schema churn, and
config tree reorganizations.

### test-writing rules file

New `.claude/rules/test-writing.md` — auto-loaded into every future session
in this project. Codifies the three-question framework (project-vs-framework,
formula-mirror-vs-property, cite-the-bug), marker discipline, and
anti-patterns from this audit.

### Files touched

**Renamed:** `graphids/commands/test_from_spec.py` → `run_test_from_spec.py`
(and `__main__.py:44` router entry).

**New tests:** `tests/config/test_yaml_utils.py`, `tests/config/test_recipe_expand.py`,
`tests/orchestrate/test_resolver.py`, `tests/orchestrate/test_validation_rules.py`,
`tests/orchestrate/test_kd_teachers.py`.

**Deleted tests:** `tests/test_smoke.py`, `tests/orchestrate/test_overrides.py`,
`tests/orchestrate/__init__.py`, `tests/config/test_recipe_expand_kd.py`.

**Modified:** `tests/conftest.py`, `tests/config/test_config.py`,
`tests/core/preprocessing/test_budget_matrix.py`,
`tests/core/preprocessing/test_vram_budget.py`, `tests/test_integration.py`,
`tests/core/models/test_{gat,vgae,fusion}.py`, `pyproject.toml`, `CLAUDE.md`,
`graphids/__main__.py`.

## What this session did (2026-04-03, session 16 — lake audit + fusion CPU pipeline)

### Lake artifact audit (set_01)

Audited all 49 run directories under `set_01`. Key findings:

| Finding | Scope | Action |
|---|---|---|
| Train val_acc 96% but test acc 17% | All GAT normal/curriculum | Not a bug — test aggregates 6 subdirs including OOD + excluded attack types. Tracked as GH issue. |
| No analysis artifacts for fusion/DGI | All fusion + DGI runs | `ANALYSIS_SUPPORTED_MODELS` had only vgae/gat. Added `dgi`. Fusion blocked on deeper issues. |
| No `best_model.ckpt` for Bandit/DQN | 2 RL fusion runs | `automatic_optimization=False` breaks `ModelCheckpoint` silently. |
| Fusion only got 50 epochs | All fusion runs | Was `max_epochs: 50`; fixed to 1500, patience 200. |
| ~12 stale orphan directories | Pre-`DeviceStatsMonitor` runs | Safe to clean up. |

### Budget module audit (DONE)

Full equation-by-equation audit. 5 bugs fixed:

| Bug | Fix |
|---|---|
| Stale module + `node_budget` docstrings | Rewritten |
| γ measurement contaminated by GPU state | `torch.cuda.synchronize()` + `gc.collect()` + 3-sample median |
| `cg_ratio` used forward-only β | Now uses `β × backward_multiplier` |
| `num_steps` truncates, drops 10–15% of data/epoch | `math.ceil` in `datamodule.py` + `curriculum.py` |
| No throughput floor guard | `budget = clamp(floor, mem_ceiling)` |

**Mathematical result:** `budget = mem_budget` for all current configs.
Throughput floor exists but is always ≪ mem_budget (max floor ≈ 87K nodes
vs min ceiling ≈ 54K nodes for GAT large on V100). Stored in
`BudgetResult.throughput_floor` as a guard.

## Key decisions (committed)

See `docs/decisions/` for the full ADR history (0001–0009). Highlights:

| ADR | Topic |
|---|---|
| 0001 | Reject Hydra/OmegaConf — jsonargparse + plain YAML |
| 0002 | Forced callbacks (checkpoint, early stopping, DeviceStats, RunRecord, ResourceProfile) |
| 0003 | SLURM job consolidation — train+test+analyze in one job |
| 0004 | Keep custom VRAM probe (not PyTorch Profiler) |
| 0005 | WandB decisions |
| 0006 | Dagster integration — Component + IOManager + Resource + checks |
| 0007 | Config system architecture — ConfigResolver as exclusive merge path |
| 0008 | DataLoader: no custom collation, use PyG primitives |
| 0009 | Collapse override handoff chain 9 → 3 (DONE, session 16) |

## Key references

| Doc | Purpose |
|---|---|
| `docs/reference/config-architecture.md` | Config tree + resolver + merge chain |
| `docs/reference/orchestration.md` | Dagster → SLURM flow |
| `docs/reference/orchestration-risks.md` | Known risks + mitigations |
| `docs/reference/data-flow.md` | Data pipeline end-to-end |
| `docs/reference/kd-pipeline.md` | KD teacher/student wiring |
| `docs/reference/throughput-model.md` | Budget cost model |
| `docs/reference/ablation-resource-profile.md` | Measured resource profile from ablation |
| `docs/reference/osc-cluster-memory-limits.md` | Per-partition mem_per_cpu (all 3 clusters) |
| `docs/reference/observability.md` | JSONL events + pipeline-status |
| `docs/reference/write-paths.md` | Lake layout + sidecars |
| `docs/reference/dataloader-performance.md` | Worker/prefetch notes |
| `docs/reference/3-chain.md` | 3-stage KD pipeline overview |

Work items live in GitHub issues now, not `docs/backlog/` (deleted
wholesale). Use `gh issue list` or the `/gh` skill.
