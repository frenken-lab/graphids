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

## Next session — triage the seed 42 re-launch (SLURM jobs live)

**Re-launch submitted 2026-04-23 via `launch_ofat.py` — 22 jobs on Cardinal.**
Python launcher held jids in memory so Stage 3 submitted in one shot
(no dep-race). Idempotent skip correctly skipped 11 completed Stage 1
variants. 3:30 wall applied to unsupervised group.

### Queued jids

```
Stage 0 (re-run, walltimed previously, 3:30:00 wall):
  8772107  graphids-fit-vgae
  8772112  graphids-fit-gae
  8772114  graphids-fit-dgi
Stage 2 (afterok VGAE):
  8772125  graphids-fit-curriculum_vgae
Stage 3 (afterok VGAE only — focal already FINISHED so no focal dep):
  8772127  graphids-extract-fusion-states
Stage 4 (afterok states):
  8772128  graphids-fit-bandit
  8772130  graphids-fit-dqn
  8772214  graphids-fit-mlp
  8772216  graphids-fit-weighted_avg
Test chains (13 for completed Stage 1 variants, --mem 32G):
  8772108-8772126 and 8772129/8772211/8772215/8772217 (fusion tests)
```

### First command to run next session

```bash
source ~/graphids/.venv/bin/activate
sacct -M cardinal -u $USER --starttime=2026-04-23 \
    --format=JobID,JobName%30,State,Elapsed,ExitCode -P \
    | grep -v '\.ba\|\.ex' | column -t -s '|'
```

### Success criteria

- **All 3 unsupervised fits FINISHED** at 3:30 wall (VGAE should need ~3h
  at 9 s/epoch × 1200 epochs; earlier runs walltimed on 1:30 wall).
  If any walltime again, bump `LONG_WALL_TIME` in `launch_ofat.py` to
  `4:00:00`.
- **curriculum_vgae FINISHED** afterok vgae.
- **extract-fusion-states FINISHED** afterok vgae (focal ckpt from
  2026-04-23 already on disk).
- **4 fusion methods FINISHED** afterok states.
- **Test chains complete** for all variants — the 13 Stage 1 tests now
  run with `--mem 32G` (the 16G default OOM'd gat test 8724434).

### Known gotchas

- **`QOSMaxJobsPerUserLimit`** on Cardinal throttled several test jobs
  to PD (concurrency cap, not failure). They'll drain as other jobs
  finish. Not actionable.
- **Zombie RUNNING MLflow rows are possible.** `mlflow_reap_zombies.py`
  was deleted 2026-04-23; if SLURM kills a fit and MLflow status
  doesn't flip to FAILED, the row stays RUNNING in the UI. Next submit
  (re-run) will replace it correctly because idempotent skip checks
  `status == FINISHED`, not "any non-zero status." Zombies are
  cosmetic, not operational.

### Triage decision tree

| Seed 42 state | Action |
|---|---|
| All 9 jobs FINISHED | Proceed to seed 123 / 777 launches: `scripts/ablation/launch_ofat.py --dataset set_01 --seed 123 --cluster cardinal` |
| Unsupervised walltimed again | Bump `LONG_WALL_TIME` → `4:00:00` in `launch_ofat.py:34`, resubmit via same command (idempotent skip handles already-done variants) |
| Stage 3 or 4 FAILED | Check logs at `/fs/ess/PAS1266/graphids/slurm_logs/graphids-<name>_<jid>.err`; resubmit via same launcher command |
| Test jobs FAILED | Likely a real bug (OOM at 32G would be new, or classification_test_metrics regression). Inspect one stderr; resubmit test manually |

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
