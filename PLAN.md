# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

## This session — ablation analysis substrate + eval demarcation

Phases from `~/plans/ablation-analysis-substrate.md`, plus a follow-up
profile collapse for the SLURM surface.

- **Phase 0 — Bouthillier 2021 §5 review** (no code). Agent extracted the
  paper; summary at `~/plans/bouthillier-2021-section-5.md`. Recommended
  N = 29 per variant under the paper's γ=0.75 / α=β=0.05 decision rule.
  Researcher constraint: N ≤ 3 (OSC allocation) — documented in
  `~/plans/phase-3-revised.md`.
- **Phase 1 — per-class test metrics.** Replaced `binary_test_metrics`
  for classifier-flavor models (GAT + all fusion) with unified
  `classification_test_metrics` (Multiclass* + ClasswiseWrapper). Test
  step buffers `(N, K)` probabilities; VGAE/DGI keep binary + threshold.
  New keys: `accuracy`, `mcc`, `ece`, `{f1,precision,recall,specificity,
  auc,ap}_{macro,weighted,per_class/<name>}`.
- **Phase 2 — parent/child MLflow runs.** Added
  `start_parent_run(group, variant, dataset)` in `_mlflow.py`; children
  link via `MLFLOW_PARENT_RUN_ID` env var → `mlflow.parentRunId` tag. CLI:
  `python -m graphids mlflow-start-parent`. Launcher opens 16 parents
  upfront; `_fit()` injects the env var per-variant.
- **Phase 3 — `graphids.analysis.compare`.** Four MLflow-driven functions:
  `leaderboard`, `tie_candidates`, `effect_size` (Cohen's d + bootstrap
  CI, no p-values at N≤3), `expected_max`. CLI: `python -m graphids compare
  {leaderboard|ties|effect-size|expected-max} <group> <dataset>`.
- **Phase 5 — SLURM profile collapse + CLI tighten.** Two profiles only
  (`gpu`, `cpu`). `scripts/run` absorbs `scripts/slurm/submit.sh`; takes
  `<preset.jsonnet>` (training) or `--mode {gpu|cpu} --command "..."`
  (ops). `submit-profile` CLI deleted; MLflow walltime-history moved to
  `graphids.slurm.sizing`. Eval demarcation falls out naturally: test
  commands submit with `--mode cpu` — no GPU allocation, no separate
  profile needed.

## 2026-04-22 session — Stages 1+2 OOV handling + pluggable IdEncoder

Driven by the 2026-04-21 Cardinal DAG failure: all 10 fit-then-test
jobs crashed with `IndexError: index out of range in self` at the
CAN-ID embedding lookup. Diagnosed as per-split vocab drift — each
split built its own `arb_id → index` mapping, so test subdirs with
novel arb_ids over-flowed the train-sized embedding table. Research
pass, design decision, and implementation in one session.

- **Research.** `~/plans/oov-embedding-handling.md` (v2, 2021–2026
  source gate, >20 cites per source). Verdict: industrial recsys has
  converged on hash-bucketed embeddings for dynamic/OOV vocabs; CAN
  IDS literature has no standard treatment. Three-stage plan.
- **Stage 1 — shared vocab (fix the crash).** New
  `graphids/core/data/vocab.py` + `CANBusSource.build()` scans union
  of all splits' source_dirs, persists `vocab.json`, digest stamped
  into `cache_metadata.json`. `METADATA_SCHEMA_VERSION` bumped 2→3.
  Rebuild verified jid 47041299 (14 min, 6 datasets; set_01 has 2046
  unique arb_ids + UNK@0 = 2047).
- **Pluggable `IdEncoder` architecture.** New
  `graphids/core/models/id_encoding/` with `IdEncoder` base,
  `LookupIdEncoder` (Stage 3 `p_unk_drop` plumbing), `HashIdEncoder`
  (Stage 2 primary). `InputEncoder` takes `id_encoder: IdEncoder`;
  VGAE/GAT/DGI Module wrappers gained `id_encoder_class_path` +
  `id_encoder_kwargs` config surface. `from_vocab_size` classmethod
  bridges `datamodule.num_ids` → encoder-native params.
- **Stage 2 ablation presets.** Three jsonnet presets under
  `configs/ablations/id_encoding/`: `lookup.jsonnet` (baseline),
  `learned_unk.jsonnet` (Stage 3 `p_unk_drop=0.1`), `hash.jsonnet`
  (Stage 2 HashIdEncoder, `k=2`, `num_buckets_factor=4`).
  All three validate through the Pydantic gate.
- **Paper.** `~/kd-gat-paper/paper/content/methodology.md` gained a
  new §"Handling Out-of-Vocabulary Arbitration IDs" subsection with
  9 new 2021–2026 bib entries (Monolith, Unified Embedding, Yan hash,
  recsys surveys, CAN-IDS survey, ROAD, CAN-MIRGU, CAN-BERT).
- **State-dict break.** `input_encoder.id_embedding.weight` →
  `input_encoder.id_encoder.embedding.weight`. Ckpts from the 2026-04-21
  Cardinal DAG are formally unloadable by design.

## 2026-04-23 session — launcher generalization + seed 42 launch

Preconditions from the 2026-04-22 session were all met before launch:
caches rebuilt (jid 47041299), state-dict refactor + pluggable
IdEncoder landed, ckpt compat guard in place, stale 2026-04-21 ckpts
deleted. SLURM sanity pytest (jid 47045029) green; Pitzer V100 VGAE
smoke (jid 47045030) walltimed at gpudebug's 1h limit with
checkpoints written (expected compute-tiny behavior per commit
6522722).

- **Launcher generalized.** `scripts/ablation/launch_set_01.sh` →
  `launch_ofat.sh` (git mv). Added `--dataset <name>` flag (default
  `set_01`). DAG shape unchanged.
- **id_encoding folded into Stage 1.** Three variants
  (`lookup`/`learned_unk`/`hash`) added to the parallel standalone
  pool. Stage 1 count 10 → 13. Removes the manual for-loop from
  PLAN's prior Step 2.
- **Cardinal fit wall bumped to 1:30:00.** `configs/resources/
  submit_profiles.json` — the prior 1:15:00 cut one pre-rebuild gat
  fit to 6 minutes of margin (69.3 min elapsed). 90 min gives 20+
  min safety margin across all 19 historical variants; VGAE on
  Cardinal has zero prior FINISHED data so this is the one-line
  insurance.
- **Docs synced.** 6 doc references updated to the new launcher
  filename + Stage 1 variant list (README, config-system rule,
  config-architecture reference, copilot-instructions,
  PLAN).

### Seed 42 outcome — 11/19 fits succeeded

| Category | Count | Detail |
|---|---:|---|
| COMPLETED | 11 | gat, gatv2, gps, none, curriculum_random, ce, weighted_ce, lookup, learned_unk, hash, focal (22–37 min each) |
| FAILED (walltime) | 3 | **vgae, gae, dgi** — max_epochs=1200 × ~9 s/ep ≈ 180 min, ran into the 1:30 wall |
| CASCADE-CANCELLED | 6 | curriculum_vgae, extract-states, bandit, dqn, mlp, weighted_avg (afterok on failed vgae/focal) |
| OUT_OF_MEMORY | 1 | gat test job at default 16G (fixed forward via 32G) |

Root cause identified: unsupervised models (VGAE/GAE/DGI) have
`max_epochs=1200` which at ~9 s/epoch on H100 needs ~3 h, not 90 min.

### Phase A simplification (executed 2026-04-23)

Driven by user push-back on the accumulating custom orchestration
burden. Seed 42 validated the complaint: dep-race fired, MLflow
parents silently failed, tests OOM'd, unsupervised walltimed, stale
`.train_complete` markers drifted.

- **Python DAG driver.** `launch_ofat.sh` (203 LOC) replaced by
  `launch_ofat.py` (431 LOC). Jids held in memory ⇒ Stage 3 dep-race
  (open #12, fired twice) **eliminated by design**.
- **Per-group walltime override.** Unsupervised group gets
  `--time 3:30:00`; others use profile default. No more walltime on
  VGAE/GAE/DGI re-launches.
- **MLflow-backed idempotent skip.** Queries latest fit status per
  `(variant, seed)`; skips re-submission if FINISHED. Filesystem
  `.train_complete` markers retired — they drifted across refactors.
- **Venv guard.** Launcher fails fast if `VIRTUAL_ENV` isn't
  `graphids/.venv` (symlinked `python` binary made site-packages
  isolation depend on the env var alone).
- **Stage-1 test OOM fixed.** `_chain_test` now passes `--mem 32G`
  (was inheriting 16G profile default).
- **`backfill_mlflow.py` deleted** (148 LOC). One-shot recovery
  script that kept being re-invoked — accept occasional historical
  data gaps.

### Phase B simplification (executed 2026-04-23)

Honest deletion of obsolete custom code:

- **`mlflow_reap_zombies.py` deleted** (127 LOC). RUNNING zombies
  now surface in MLflow UI until next submit overwrites them. User
  can run `sacct` manually if they care. The alternative was a cron
  cross-referencing SLURM with MLflow — exactly the "custom
  orchestration that breaks" the user flagged.
- **`start_parent_run` deleted** (54 LOC) from `_mlflow.py` plus
  `_parent_run_tag` helper (10 LOC) and `MLFLOW_PARENT_RUN_ID`
  constant. Launcher no longer creates parents; children are
  filterable by `graphids.*` tags without a parent_run_id.
- **`mlflow-start-parent` CLI deleted** (36 LOC) from `cli/app.py`.
- **`_call_with_retry_on_schema_race` deleted** (20 LOC). MLflow 3
  handles this better; the rare "table already exists" race wasn't
  worth the wrapper.

**Cumulative LOC delta: −184** (deleted: `mlflow_reap_zombies.py`
−127, `backfill_mlflow.py` −148, `launch_ofat.sh` −203, parent-run
plumbing −101; added: `launch_ofat.py` +431, minor docs +4).

The 428-LOC `_mlflow.py` remaining is doing real work: 5 short
lifecycle functions + helpers for tag/param/metric shaping that
MLflow can't do natively. Further inlining would move complexity to
callsites without deleting it — stopped here.

## 2026-04-24 session — DGI bug RCA, cluster-tag fix, DAG into the library

Triage of the Cardinal re-launch surfaced three bugs + one architectural
cleanup.

### DGI test uncalibrated centroid (Cardinal jid 8772115)

DGI test FAILED at 41s with `DGIModule.svdd_center is uncalibrated`.
Root cause: `SVDDCalibrationCallback.on_fit_end` calibrated the
in-memory buffers but never re-saved the ckpt. `ModelCheckpoint` only
writes in `on_train_epoch_end`, so both `best_model.ckpt` and
`last.ckpt` shipped with `svdd_calibrated=False` baked into state_dict.
Latent since commit `9c35c47` (2026-04-16) — first DGI test to actually
run because the earlier Cardinal runs all walltimed or crashed before
test.

**Fix:** delete the persistence path entirely. Centroid is a
deterministic function of `(encoder, benign train data)` — re-fit fresh
at `Trainer.test()` start, don't persist in state_dict. Deleted
`SVDDCalibrationCallback`, `svdd_calibrated` buffer, jsonnet
registration, and `scripts/recalibrate_dgi.py`. Added `strict=False`
load with a log line in `Trainer._load_model_weights` so old-format
ckpts still load. Regression verified end-to-end: seed 42 DGI test (jid
8815229) FINISHED, 28 metric keys populated.

### gh#40 — empty `graphids.cluster` MLflow tag

`GraphIDSSettings.cluster` now falls back to `SLURM_CLUSTER_NAME` when
`GRAPHIDS_CLUSTER` isn't exported. Works from any submission path
without per-script coordination. Historical rows stay empty; fix is
forward-only.

### Launcher into the library (Option A)

`scripts/ablation/launch_ofat.py` (432 LOC) migrated to
`graphids/slurm/dag.py` + `graphids/cli/ablation.py`. Topology is now a
declarative `OFAT_DAG` tuple of `FitNode`/`ExtractStatesNode` consumed
by a topological executor. CLI: `python -m graphids launch-ablation
[--dataset X --seed N --cluster c --dry-run]`.

Pathway to the declarative future already set up: lift `OFAT_DAG` into
`configs/ablation_dag.py` (step 1) → load from jsonnet (step 2) → emit
Mermaid/Graphviz from the loaded DAG (step 3).

### Session net

-117 LOC on bugs + -70 LOC on launcher move. Full test suite green
through all three changes (121 passed).

## 2026-04-24 evening session — drift fix + library consolidation

Started with a Pitzer fusion-fit crash (`ModuleNotFoundError: No module
named '_jsonnet'`): commit `8e14e06`'s `uv sync` had pruned the
undeclared C-binding dep. Cascaded into a full audit pass.

### Drift / settings fixes

- **`jsonnet>=0.20`** declared in `pyproject.toml`; `uv.lock` pins
  `jsonnet==0.22.0` (manylinux wheel, no compile). ADR 0010 reversed —
  PyPI binding is now load-bearing-correct, hand-installed go-jsonnet
  binary at `~/.local/bin/jsonnet` deprecated.
- **`filelock>=3.13`** declared; `core/data/io.py` deleted (42 LOC) —
  `nfs_lock` → `filelock.FileLock`, duplicate `atomic_save` consolidated
  to `_fs.atomic_save`.
- **Fail-fast settings**: `lake_root` no longer defaults to relative
  `"experimentruns"` (silent CWD pollution gone). `GraphIDSSettings`
  auto-loads `./.env` via pydantic-settings `env_file=...`, so login-node
  invocations don't need `set -a; source ./.env`. `extra="ignore"` so
  shell-only `GRAPHIDS_*` vars in `.env` don't break validation.
- **`GRAPHIDS_RUN_ROOT` introduced** as a separate env var distinct from
  `GRAPHIDS_LAKE_ROOT`. The two were silently conflated under one name —
  `LAKE_ROOT=/fs/ess/PAS1266/graphids` (shared: mlflow.db, cache,
  mlartifacts) was being used as the root for run_dirs as well, but the
  jsonnet preset defaults hard-coded `/fs/ess/PAS1266/graphids/dev/rf15`
  (per-user). Naive consolidation would have orphaned every existing
  ckpt. Now they're independent vars.

### `_mlflow.py` defensive-wrapper purge

668 → 614 LOC; 16 try/except blocks → 4. Module philosophy reversed
from "every MLflow call try/swallow" to "MLflow is required, failures
propagate." Removed: ImportError guards (mlflow is a hard dep), broad
`except Exception` outer wrappers in `start_training_run`, `log_test_run`,
`log_epoch_metrics`, `log_final_fit`, `_register_logged_model`. Narrowed:
`_git_sha_tag` (`CalledProcessError, FileNotFoundError`),
`log_params` resume conflict (`MlflowException`), `end_training_run`
cleanup (`MlflowException`, kept logged-not-raised so secondary failure
doesn't shadow primary training exception via `__context__`). Aligns
with `feedback_no_backward_compat_fallbacks` rule.

### Three jsonnet consolidations (all landed)

1. **`std.extVar('run_root')`**: `lake_root` TLA default removed from
   all 19 ablation presets (was duplicated 19× and drifting from
   settings). Set once in `render()` from `RUN_ROOT`.
2. **`std.native('paths.X')(...)`**: `configs/ablations/_paths.libsonnet`
   deleted. `graphids/config/paths.py` is canonical for run_dir /
   vgae_ckpt / states_dir; jsonnet calls into it via `native_callbacks`
   registered by `render()`. `slurm/dag.py:_run_dir` is a Path-typed
   wrapper, drops its `lake_root` parameter and the duplicate scheme.
   **Upstream `_jsonnet` bug worked around**: native callback param names
   of length ≥2 raise "binding parameter a second time" when called
   positionally with mixed local/literal args; single-letter names dodge
   it.
3. **`std.mergePatch` + `ext_code`**: `configs/_lib/helpers.libsonnet`
   deleted; `cli/app.py:apply_overrides` deleted (replaced with
   `dotted_to_nested`). `--set a.b.c=v` flags now flow as
   `std.extVar('overrides')` and apply via `std.mergePatch` at each
   preset's apex. Stages no longer take `trainer_overrides` /
   `stage_overrides` TLAs; presets express group defaults as nested
   objects directly. One mechanism replaces three duplications.

### Files touched

| Concern | Files |
|---|---|
| Created | `graphids/config/paths.py` |
| Deleted | `core/data/io.py`, `configs/ablations/_paths.libsonnet`, `configs/_lib/helpers.libsonnet`, `cli/app.py:apply_overrides` |
| Modified Python | `pyproject.toml`, `uv.lock`, `.env`, `settings.py`, `constants.py`, `jsonnet.py`, `cli/app.py`, `cli/training.py`, `slurm/dag.py`, `_mlflow.py`, `core/data/datasets/can_bus.py` |
| Modified jsonnet | All 3 stages + all 19 ablation presets |
| Modified docs | `docs/decisions/README.md` (ADR 0010), `.claude/rules/{config-system,data-layout}.md`, `docs/reference/{config-architecture,observability,data-flow,orchestration,write-paths}.md`, `configs/CONFIG_REFERENCE.md`, `configs/ablations/README.md`, `CLAUDE.md` |

### Render snapshots verified identical

Rendered four representative presets pre/post each stage and diff'd —
zero output drift across all 3 consolidations. 133 tests pass.

### Pitzer seed 42 status (live)

| Tier | Fits | Tests |
|---|---|---|
| Stage 0 unsupervised (3) | ✓ all FINISHED on Cardinal | vgae ✓, gae ✓, dgi resubmitted (CUDA-on-CPU budget probe pre-existing bug) |
| Stage 1 (13: conv_type/gat_loss/gat_sampling/id_encoding) | ✓ all FINISHED on Cardinal | 7 ✓; 6 failed at 21:36 ET on transient settings.py state — **resubmitted at 22:50 ET** |
| Stage 2 curriculum_vgae | ✓ FINISHED | ✓ FINISHED |
| Stage 3 extract-fusion-states | ✓ FINISHED on Pitzer | n/a |
| Stage 4 fusion (4) | ✓ all FINISHED on Pitzer | 3 ✓; bandit failed on stale `lake_root` TLA (pre-Stage-A) — **resubmitted** |

**All 20 fits FINISHED, 19 tests in flight on Pitzer at session end.**

## Next session — first order of business: triage seed 42 test re-run

**Tests resubmitted at 22:50 ET — jids 47079986–47080005.** Most should
finish overnight; 3 known concerns to verify:

1. **dgi-test specifically**: previous failure was
   `RuntimeError: budget probe prerequisites missing: CUDA (conv_type=gatv2)`
   on a CPU test. Pre-existing bug — commit `a224f8c` ("Skip budget probe
   for val/test dataloaders") presumably doesn't cover the test path DGI
   takes. If it fails again: investigate `core/data/budget.py` test-path
   skip logic. Workaround: set `GRAPHIDS_ALLOW_FALLBACK_BUDGET=1`.
2. **18 cosmetic zombie RUNNING rows** in MLflow from the earlier failed
   tests (those tests opened MLflow rows that the new no-swallow
   `end_run` would have closed FAILED, but they crashed before reaching
   that). Cosmetic only — `compare.py` filters to FINISHED. To clean:
   manually `UPDATE runs SET status='FAILED' WHERE status='RUNNING' AND
   end_time IS NULL` on `/fs/ess/PAS1266/graphids/mlflow.db`, or accept
   them.
3. **Stage A cluster context**: SLURM jobs are pickled at submission;
   the in-flight 19 tests will use whatever `graphids/` source tree is
   on disk at job-start time. With Stages A/B/C all landed and
   semantics-preserving, results should match what Cardinal produced
   for the same fit ckpts.

### First command to run next session

```bash
source ~/graphids/.venv/bin/activate
set -a && source ./.env && set +a
sacct -u $USER --starttime=2026-04-24T22:00 \
    --format=JobID,JobName%32,State,Elapsed,ExitCode -P \
    | grep -v '\.ba\|\.ex\|\.0' | column -t -s '|'
```

### Triage decision tree

| State | Action |
|---|---|
| All 19 tests FINISHED | Proceed to N=3: `python -m graphids launch-ablation --dataset set_01 --seed 123 --cluster pitzer` then `--seed 777` |
| dgi-test fails again | Debug `core/data/budget.py` val/test skip path; the budget probe shouldn't fire on CPU-mode test dataloaders |
| Bandit-test fails | Stale TLA shouldn't be possible post-Stage-A; if it fails, the launcher's `chain_test` call needs investigation — submitted via current code, not pickled state |
| Other test fails | Likely real bug. Inspect stderr at `/fs/ess/PAS1266/graphids/slurm_logs/<jid>_0_log.err`; resubmit just that test |

### Step 2 — scale to N=3 + compare

Once seed 42 is green (or failures are understood), mirror launches
for `--seed 123` and `--seed 777`:

```bash
scripts/ablation/launch_ofat.py --dataset set_01 --seed 123 --cluster cardinal
scripts/ablation/launch_ofat.py --dataset set_01 --seed 777 --cluster cardinal
```

Then:
```bash
python -m graphids compare effect-size id_encoding set_01
python -m graphids compare leaderboard id_encoding set_01
```

### Step 3 — paper data pipeline

Once N=3 is in, run `graphids/analysis/export_hf.py` (design at
`~/plans/hf-dataset-design.md`) to push the leaderboard +
effect-size tables to HF, then `~/kd-gat-paper/data/pull_data.py`
pulls them so paper rendering is device-agnostic.

## Still-open follow-ups

None blocking the next session. All Stage 1+2 correctness work is
landed; the remaining items are verification (SLURM jobs above) and
paper-pipeline integration.
- **Retroactive eval on existing fit-only ckpts.** One-shot sweep script
  to populate MLflow test rows for historical runs. Sizable compute
  (~20 fit-only ckpts × 5 min CPU each) but straightforward.
- **Phase 4 — seed-expansion launcher wrapper.** Bash: take `--seeds
  1,2,3` and loop. Low priority given N ≤ 3 screening workflow already
  uses the existing `--seed` loop.
- **HF dataset export pipeline.** Design drafted at
  `~/plans/hf-dataset-design.md`. Implementation starts once seed 42
  completes cleanly: `graphids/analysis/export_hf.py` reads MLflow +
  run_dirs, assembles the bucket tree, pushes versioned revision.
  Paper build (`~/kd-gat-paper/data/pull_data.py`) consumes from HF
  so rendering is device-agnostic.
- **fp16 overflow — hcrl_sa only.** Confirmed NOT a regression:
  set_01 fp16 ran clean on V100 past the crash window (job 46974907).
  hcrl_sa has outlier features under the train-only scaler (cache v9)
  that push something in VGAE's forward past fp16 range. Low
  priority — smoke tests on hcrl_sa can use `--set
  trainer.precision="32-true"`; real ablation uses set_01.
- ~~Launcher dep-race (task #12)~~ — **RESOLVED** by Python DAG
  driver (Phase A 2026-04-23). Jids held in memory; no scheduler
  query between stages.
- ~~MLflow parent-run creation silently fails~~ — **RESOLVED** by
  deletion (Phase B 2026-04-23). Parent runs removed from the
  design; `mlflow-start-parent` CLI + `start_parent_run` gone.
- **MLflow tracking — three-plan decision set**
  (drafted 2026-04-23, all gated on seed 42 finishing cleanly):
  - **`~/plans/mlflow-tracking-simplification.md`** (574 lines) —
    moderate plan. One experiment per `(dataset, axis)`, no
    parent/child, status-gated resume (5 statuses), upstream
    lineage via `graphids.upstream.{vgae,gat}_run_id` tags. All 5
    open questions resolved with principles. Net ~-80 LOC.
  - **`~/plans/mlflow-tracking-plan-review.md`** (228 lines) —
    neutral ML-engineer review informed by SOTA (MLtraq, Aim, SEML,
    Hydra+submitit, DuckDB+Parquet). Flags 6 plan gaps: no
    alternatives considered, no tests for resume matrix, no SQLite
    contention verification, upstream tag couples to MLflow
    internals (should be `run_dir` not `run_id`), missing
    git-SHA-mid-resume scenario (Q6), reversibility of
    parent-run deletion.
  - **`~/plans/mlflow-maximalist.md`** (941 lines) — inverse
    direction: use MLflow to the hilt, cut custom wrapper code.
    Eight features evaluated + rejected (autolog = Lightning-only;
    evaluate = tabular-only; projects = no SLURM backend; nested
    runs = wrong semantic for OFAT; UI-based compare lacks
    bootstrap CI; `log_inputs(models=...)` is Databricks-only;
    no batch system-metrics API; `search_logged_models` no tag
    filter). Two worth adopting: `mlflow.data.Dataset` +
    `log_input` (-23 LOC, UI lineage), `MlflowClient.create_logged_model()`
    metadata-only (+10 LOC, first-class upstream entity without
    triggering the artifact-store ban).
  - **Recommended synthesis (Option C):** adopt moderate plan as-is,
    then layer MetaDataset (clear win) + LoggedModel (queryability)
    from the maximalist. Net ~-97 LOC. Also adopt review's #2 (resume
    matrix tests), #3 (swap to `upstream_run_dir` tag), and #7 (stamp
    `uv.lock` hash + python version).
  - **`~/plans/ablation-orchestration-evaluation.md`** (716 lines) —
    evaluates Optuna / Hydra+submitit+optuna-sweeper / Ray Tune /
    Metaflow / NNI / SEML / Dagster as ablation-framework candidates.
    Verdict: **don't adopt any.** Evidence-based findings: Optuna
    alone is hyperopt-only (no DAG); MLflow's `MlflowStorage` for
    Optuna (v2.22.0+) actually *restores* parent/child the moderate
    plan deliberately deletes; Hydra+submitit doesn't express
    `afterok` natively (the real pain point); NNI archived Sept 2024;
    SEML replaces MLflow (wrong direction); dagster-slurm +
    metaflow-slurm require login-node drivers (violates OSC policy).
    **The Stage 3 dep-race is a 5–15 LOC bash retry**, not a
    framework problem. Keep jsonnet + `launch_ofat.sh`.

## Open issues

- **#32** Add WaDi dataset module.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
