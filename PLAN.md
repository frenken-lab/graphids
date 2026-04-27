# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

## This session — ablation analysis substrate + eval demarcation


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
