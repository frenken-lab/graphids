# KD-GAT Session Plan

> Last updated: 2026-03-17

## Active Plan

No active plan. Pipeline consolidation complete. Next: run training.

## In Progress

- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces. Shows 181 experiment runs + 37 sweep trials.

## Blocked

(none)

## Open Questions

### Orchestration architecture
- Should Ray be kept at all, or fully replaced? (Path A vs C)
- When to revisit Flyte? (trigger: OSC gets K8s)
- How to handle HPO if Ray is dropped? (standalone Optuna loses ASHA multi-fidelity)
- Database consolidation: SQLite (single-writer) + PostgreSQL (multi-writer) + DuckDB (analytics) — three engines?

### Is RL justified for fusion?
See `~/plans/fusion-redesign.md` for full analysis.

## Next Up (after orchestration)

- Fusion method comparison experiment
- Evaluate research questions R1–R3
- Research visualization Space (`buckeyeguy/kd-gat-research`) — see previous PLAN.md for detailed requirements

## Key Reference Documents

| Document | Purpose |
|----------|---------|
| `~/plans/ecosystem-component-registry.md` | 24-component grocery list with interfaces and gaps |
| `~/plans/orchestration-tool-evaluation.md` | 6 tools scored against 14 requirements |
| `~/plans/code-state-report-template.md` | Reusable 10-step codebase analysis methodology |
| `~/plans/2026-03-11-architecture-session.md` | Full session decisions and findings |
| `~/plans/fusion-redesign.md` | RL fusion analysis |
| `~/plans/slurm-orchestration-redesign.md` | Original orchestration design rationale |

## Completed

- **Pipeline layer consolidation v2** — 4 phases: bugs+config, torchmetrics, batched eval (10-50x speedup), god function decomposition. See `plans/pipeline-consolidation.md`. (2026-03-17)
- **Preprocessing module hardening** — 6 fixes: ghost config param, adapter serialization, IR validation, feature manifest, SRP split. (2026-03-17)
- **Models layer hardening** — decouple extractor, consolidate conv, typed layout. (2026-03-17)
- **Architecture review & ecosystem mapping** — 50 files / 9,540 lines inventoried. 24 ecosystem components defined with interfaces, gaps, priorities. 4 critical gaps identified. Orchestration tool evaluation (6 tools × 14 requirements). Session doc written. (2026-03-11)
- Codebase consolidation: 12,511→9,537 lines (-24%), 55→50 files. Deleted unused orchestration (executor, driver, planner), moved loss_landscape to scripts/, trimmed tune_config. (2026-03-10)
- MLOps tools catalog: 437 tools compiled. 12 dimensions. (2026-03-10)
- Shared PostgreSQL backend: Apptainer PG 16 on SLURM, on-demand launcher, dual-backend store. (2026-03-10)
- Scheduler-agnostic orchestration: built and deleted same day — 5-component system was over-engineered. Code preserved in git history (commits 123845f, 971dd6e, c697b63). (2026-03-10)
- Memory/batch sizing simplification: ~600 lines removed. (2026-03-07)
- MLflow migration: replaces W&B + lakehouse + CSVLogger. (2026-03-06)
- Feature engineering v2.0.0: 11→26-D node features, GATv2. (2026-03-03)
