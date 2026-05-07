# Experiment-framework evaluation

**Status:** 2026-05-05, decision shipped. Spike still pending. Compressed
2026-05-07 — full per-framework comparison archived in
`git log --follow` history.

## Decision

**Keep custom `graphids/plan/`.** Drift resistance (config snapshot frozen
in sbatch at submit time, see `chassis-invariants.md` §2) is load-bearing
for paper-deadline research; no external framework provides it without a
~30 LOC overlay. The "static JSON in git" property is *partly* real
(self-contained sbatch line, login-node Pydantic validation across the
full rendered tree) and *partly* sunk-cost (stash/replay/diff/handover
all reduce to `git SHA + plan_module + plan_args`).

Core (compose / primitives / paths / orchestrate / MLflow / SIGUSR2) is
framework-agnostic and survives any choice. Chassis (schema / plan_id /
cli / plans) is what would migrate.

## Frameworks rejected (one line each)

- **Optuna** (4.x, in `pyproject.toml`, unused) — best fit on paper.
  Replaces schema + plan_id + dashboard + retry, gives TPE / Hyperband /
  NSGA-II / `optuna-dashboard` / `enqueue_trial` for ~50 LOC objective.
  Loses drift resistance: exec re-runs `objective(trial)` against
  current code, so a stray `compose()` edit reaches queued jobs.
  **Outcome:** held pending spike.
- **Ray Tune** — owns the cluster (1 sbatch, N internal trials), breaks
  the SLURM-per-row chassis. PBT is uniquely good but we don't need it.
  Skip unless we head toward FSDP/DDP-per-trial.
- **Ray Core** — wrong abstraction layer; we'd rebuild Optuna or Tune on
  top. Skip.
- **Flambe** — last release 2020, no Lightning support, YAML-first
  (the layer we just deleted). Architecturally regressive. Skip.

## What survives interrogation of the static-JSON claim

Real wins:

1. **Drift resistance** — composer edits don't reach queued jobs;
   model/data class edits do. Asymmetric by design.
2. **Self-contained sbatch line** — `python -m graphids exec --row '<json>'`
   pasted from `*.err` is its own reproducer, no DB needed.
3. **Login-node validation breadth** — Pydantic walks the entire rendered
   class_path tree before SLURM ingest.

Use cases that evaporate on inspection (= `git SHA + plan_module + args`
covers them either way): stash-and-replay, hand-to-co-author, diff
between sweeps, audit log.

## Spike (still planned)

Half-day, login-node-feasible up to SLURM submit.

**Scope:** reimplement `ablations/ofat.py` GAT loss-fn axis (3 values × 3
seeds = 9 trials, all `fit`) as an Optuna study.

**Deliverables:**
1. `graphids/plan/plans/ablations/ofat_optuna.py` alongside `ofat.py`.
   `objective(trial)` reuses `compose.compose()`, suggests
   `loss_fn ∈ {focal, ce, weighted_ce}`, returns val_auroc.
2. `graphids submit --study <name> --n-trials 1` flag; sbatch body
   becomes `python -m graphids exec --study <name> --n-trials 1`.
3. Storage at `${RUN_ROOT}/studies/<name>.db` (sqlite for spike;
   journal-file or psql if NFS concurrency surfaces).
4. Comparison artifact: chassis LOC delta, time-to-9-jobs-queued,
   `optuna-dashboard` usability vs drafted `plans show`, retry
   semantics under killed worker.

**Out of scope:** fusion plan (multi-action), replacing Parsl, pruning
(OFAT too small).

## Decision gate

Revisit if the spike shows **all three**: chassis LOC delta negative,
`optuna-dashboard` usable, `study.enqueue_trial` retry works cleanly. In
that case migrate: plans → Optuna studies; delete `Plan` / `TrainRow` /
`plan_id` / `cli/plans.py`. Keep `compose.py` / `primitives.py` /
`paths.py` / `orchestrate.run_row` / MLflow callback / Parsl `submit_row`
unchanged.

If sqlite-on-NFS or concurrency issues block migration → evaluate
journal-file storage or postgres before rejecting Optuna outright.

If a workflow surfaces that depends on drift resistance more than
documented today → keep custom, document the workflow.

Otherwise (spike inconclusive or chassis LOC delta non-negative): stay
custom indefinitely.

## Hold list

Pause until spike resolves:

- `docs/drafts/chassis-followons.md` do-now / do-later items mooted by
  Optuna migration.
- `docs/drafts/plan-chassis-reorg.md` — same.
- TUI direction. Optuna has a real dashboard; static-HTML over TUI even
  if we keep custom (see `chassis-design-lessons.md` Lesson 5).
- Further work on `cli/plans.py`.

The 2026-05-04 readability refactor (lib→primitives, blueprint→schema,
row folded into compose) stays — strict improvement, survives Optuna
adoption.

## What survives any choice

- `plan/compose.py` — model/data/loss block assembly. Becomes the body
  of whatever objective function the framework wants.
- `plan/primitives.py` — class-path catalog + `spec()` helper.
- `paths.py` — `run_dir`, `best_ckpt`, `states_dir`.
- `orchestrate.run_row` — instantiates Lightning + dispatches on action.
- MLflow logging callback.
- SIGUSR2 preempt-resume in `orchestrate._trainer_kwargs`.
- Pydantic schema for action-dispatch (`fit`/`test`/`extract`/`analyze`/
  `cache`) — Optuna doesn't replace this; lives at a different layer.

## Citations

- Optuna 4.x docs: https://optuna.readthedocs.io
- Ray Tune key concepts: https://docs.ray.io/en/latest/tune/key-concepts.html
- Ray Core tasks: https://docs.ray.io/en/latest/ray-core/tasks.html
- Flambe (archival): https://github.com/asappresearch/flambe — last
  release 0.4.17 (2020).
- Drift resistance property: `.claude/rules/chassis-invariants.md` §2.
- `feedback_submitit_pickle.md` (the asymmetry: source edits reach
  queued jobs, config edits don't).
- `chassis-design-lessons.md` — seven lessons distilled from this eval.
