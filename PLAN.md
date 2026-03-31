# KD-GAT Session Plan

> Last updated: 2026-03-31

## Current State

Run 005 completed 22/36 jobs on Ascend A100. All 3 failures fixed (fusion wiring, preprocessing OOM, wall times). Ready to relaunch — but systemic issues in `plans/open_issues.md` should be addressed first.

## Blocking — Must fix before relaunch

1. **Forced callbacks** — ModelCheckpoint silently dropped by jsonargparse list replacement. Curriculum runs trained 300 epochs with no checkpoint. Fix spec: `plans/architecture/forced-callbacks.md`
2. **Open issues triage** — `plans/open_issues.md` has 25+ deferred items across 6 categories. Need to prioritize: which block correct results vs which are cleanup.

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
| `plans/architecture/write-paths.md` | Filesystem layout, write path inventory |
| `plans/architecture/dagster-native-orchestration.md` | Component architecture reference |
| `plans/research/profiling-and-observability.md` | What's wired, what's not, tool decisions |
