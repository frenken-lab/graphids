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

## Next session — Cardinal seed 42 in flight

Full `set_01` × seed 42 DAG submitted to Cardinal (2026-04-21). Jids
`8691643`–`8691677` (33 jobs: 16 GPU fit + 16 CPU test + 1
extract-fusion-states). Stage 3 hit a one-shot dep race on first
submission → retried successfully as jid `8691669`. See
`.claude/rules/` + `~/lab-setup-guide/docs/ml-workflows/hpc-training-nuances.md`
for the cluster-selection + race-condition lessons.

**Resume checklist when jobs finish**:

1. `squeue -u $USER -M cardinal` — confirm all 33 done or in expected state
2. `sacct -M cardinal -j 8691643-8691677 -o JobID,State,Elapsed,ExitCode` — scan for FAILED
3. Verify ckpts: `find /fs/ess/PAS1266/graphids/dev/rf15/set_01/ablations -name best_model.ckpt -newer /tmp` (16 expected; VGAE + 10 standalone + curriculum_vgae + 4 fusion)
4. MLflow parent linkage: `sqlite3 /fs/ess/PAS1266/graphids/mlflow.db "SELECT count(*) FROM tags WHERE key='mlflow.parentRunId' AND value IN (SELECT run_uuid FROM runs WHERE name LIKE '%_set_01_seed42_cardinal')"`
5. If all green: launch seeds 123 + 777 via `scripts/ablation/launch_set_01.sh --seed 123 --cluster cardinal` (and 777). N=3 screening set complete.
6. Run `python -m graphids compare {leaderboard|effect-size|expected-max} <group> set_01` over each axis to verify Phase 3 analysis pipeline works on real multi-seed data.

**Note**: Pitzer run earlier today produced an under-trained VGAE ckpt
(walltime hit at epoch ~700/1200); Cardinal's fresh run will overwrite it.
No data to reconcile.

## Still-open follow-ups

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
- **Launcher retry on cross-stage dep race (task #12).** Stage 3's
  first submission failed with "Job dependency problem" because
  Cardinal's scheduler hadn't registered the upstream focal jid yet.
  Either sleep 2-5s between stage boundaries or retry with backoff
  on that specific error.

## Open issues

- **#32** Add WaDi dataset module.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
