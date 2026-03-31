# KD-GAT Session Plan

> Last updated: 2026-03-31 (end of context audit session)

## Current State

Run 005 completed 22/36 jobs on Ascend A100. All 3 immediate failures fixed (fusion wiring, preprocessing OOM, wall times). **NOT ready to relaunch** — systemic issues remain.

## Handoff — What the last session did (2026-03-31)

Full audit and consolidation of plans/, docs, and context. No code changes.

**Deleted 9 completed/superseded plan files** (-1855 net lines across 24 files):
- `run-005-fixes.md`, `flatten-model-config.md`, `trainer-yaml-wiring.md`, `models-consolidation.md`, `preprocessing-consolidation.md`, `gpu_vram_usage.md`, `ablation-001-training-efficiency.md`, `ablation-run-004-failures.md`, `tier-priority-and-implementation.md`
- All open items extracted to `plans/open_issues.md` before deletion

**Deduplicated GitNexus** — 3 copies (CLAUDE.md, AGENTS.md, rules/gitnexus.md) collapsed to 1 (rules file). CLAUDE.md and AGENTS.md now have one-line pointers.

**Trimmed 6 files** — PLAN.md, forced-callbacks.md, ablation_and_main_005.md, dagster-history.md, dagster-native-orchestration.md, write-paths.md. Cut resolved history, stale cross-refs, verbose fix proposals.

**Audited scripts/** — found 4 stale scripts (old CLI), 1 calling nonexistent Python file, ghost `.pyc`, missing `run_tests_slurm.sh`. Wrote refactor plan: `plans/scripts-refactor-option-c.md`.

**Stale memories identified but NOT cleaned** — `project_hydra_config_refactor.md` (Hydra was rejected), `feedback_yaml_only_config.md`, `feedback_never_run_tests_login.md`, `feedback_slurm_partition.md` all duplicate rules files. Delete these.

## Blocking — Must fix before relaunch

1. **Forced callbacks** — ModelCheckpoint silently dropped by jsonargparse list replacement. Curriculum runs trained 300 epochs with no checkpoint. Fix spec ready: `plans/architecture/forced-callbacks.md`
2. **Open issues triage** — `plans/open_issues.md` has 25+ deferred items across 6 categories. Need to prioritize: which block correct results vs which are cleanup.
3. **Scripts refactor** — 4 stale SLURM scripts use deleted CLI. Plan ready: `plans/scripts-refactor-option-c.md`

## In Progress

- HF Spaces dashboard (`buckeyeguy/kd-gat-dashboard`)

## Next (after relaunch succeeds)

1. Evaluation + analysis as dagster assets (`plans/architecture/evaluation-analysis-assets.md`)
2. HPO sweep with Optuna (Phase 2 of `plans/experiment-sweep-plan.md`)
3. Final evaluation — best config, all 6 datasets, 3+ seeds (Phase 3)

## Key References

| Doc | Purpose |
|-----|---------|
| `plans/open_issues.md` | All deferred items, consolidated |
| `plans/experiment-sweep-plan.md` | 17-config ablation matrix, stage-sharing DAG, phased HPO |
| `plans/ablation_and_main_005.md` | Run 005 job summary + failure post-mortem |
| `plans/architecture/forced-callbacks.md` | Checkpoint loss fix — ready to implement |
| `plans/scripts-refactor-option-c.md` | scripts/ cleanup — thin shells, Python does the work |
| `plans/architecture/write-paths.md` | Filesystem layout, write path inventory |
| `plans/architecture/dagster-native-orchestration.md` | Component architecture reference |
| `plans/research/profiling-and-observability.md` | What's wired, what's not, tool decisions |
