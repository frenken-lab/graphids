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

## Still-open follow-ups

- **Retroactive eval on existing fit-only ckpts.** One-shot sweep script
  to populate MLflow test rows for historical runs. Sizable compute
  (~20 fit-only ckpts × 5 min CPU each) but straightforward.
- **Phase 4 — seed-expansion launcher wrapper.** Bash: take `--seeds
  1,2,3` and loop. Low priority given N ≤ 3 screening workflow already
  uses the existing `--seed` loop.
- **Stage 3 states-jid capture broken in dry-run (pre-existing).**
  `STATES_JID[$SEED]="${line##* }"` extracts the last whitespace token;
  in dry-run this is the tail of the sbatch `--wrap` arg, not the jid.
  Real-run behavior is correct (last line is
  `"Submitted batch job NNN"`). Fix by matching `sbatch` output
  explicitly or dropping dry-run display of downstream dep fields.
- **`_paths.libsonnet` uses `best.ckpt` but ModelCheckpoint writes
  `best_model.ckpt`.** Affects `vgae_ckpt()` / `gat_ckpt()` helpers and
  Stage 3 `extract-fusion-states` invocation. Latent bug — curriculum_vgae
  and fusion chains will fail to find the upstream ckpt. Fix in one pass
  across `_paths.libsonnet:11,14`, `scripts/ablation/launch_set_01.sh:134-135`,
  `CLAUDE.md:22,31,34`, `configs/ablations/README.md`.

## Open issues

- **#18** Validate GPU-first auto-sizing on SLURM — unblocked by probe
  rewrite; run one training job to confirm.
- **#32** Add WaDi dataset module.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
