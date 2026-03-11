# KD-GAT Session Plan

> Last updated: 2026-03-11

## Priority: Ecosystem Architecture & Tool Selection

Completed a full architectural review and ecosystem component mapping. Before training or building new orchestration, we need to finalize tool selections against the 24-component registry.

### Next Steps

1. **Review ecosystem component registry** — `~/plans/ecosystem-component-registry.md` defines all 24 components with interfaces, gaps, and priorities. Use this as the "grocery list" for tool shopping.
2. **Fix critical gaps (low-effort):**
   - Commit `uv.lock` to git (reproducibility)
   - Add git SHA + DVC rev to MLflow tags (reproducibility + data lineage)
   - Add per-stage SU tracking via `sacct` integration
3. **Orchestration decision** — choose between:
   - Path A: Keep Ray + add Submitit for per-stage SLURM dispatch (~200 lines)
   - Path B: Replace Ray with Parsl (if Ray overhead is a blocker)
   - Path C: Minimal custom coordinator + Submitit (if dropping Ray entirely)
   - See `~/plans/orchestration-tool-evaluation.md` for full analysis
4. **Implement plan-then-execute pattern** — separate DAG spec (YAML templates) from execution (pluggable backends). See session doc section 7.
5. **Add dry-run mode** — `cli plan --dry-run` to preview what a pipeline run will do before committing SUs
6. **Statistical significance testing** — add bootstrap CI + paired t-test to evaluation stage for multi-seed runs
7. **Run training** — full sweep once orchestration is settled

## In Progress

- **Ecosystem component registry** — 24 components mapped with interfaces, gaps, priorities (`~/plans/ecosystem-component-registry.md`)
- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces. Shows 181 experiment runs + 37 sweep trials.
- **tool-landscape** (`~/tool-landscape`) — DuckDB-backed evaluation framework, 1,157 tools. Used for orchestration tool evaluation.

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

- **Architecture review & ecosystem mapping** — 50 files / 9,540 lines inventoried. 24 ecosystem components defined with interfaces, gaps, priorities. 4 critical gaps identified. Orchestration tool evaluation (6 tools × 14 requirements). Session doc written. (2026-03-11)
- Codebase consolidation: 12,511→9,537 lines (-24%), 55→50 files. Deleted unused orchestration (executor, driver, planner), moved loss_landscape to scripts/, trimmed tune_config. (2026-03-10)
- MLOps tools catalog: 437 tools compiled. 12 dimensions. (2026-03-10)
- Shared PostgreSQL backend: Apptainer PG 16 on SLURM, on-demand launcher, dual-backend store. (2026-03-10)
- Scheduler-agnostic orchestration: built and deleted same day — 5-component system was over-engineered. Code preserved in git history (commits 123845f, 971dd6e, c697b63). (2026-03-10)
- Memory/batch sizing simplification: ~600 lines removed. (2026-03-07)
- MLflow migration: replaces W&B + lakehouse + CSVLogger. (2026-03-06)
- Feature engineering v2.0.0: 11→26-D node features, GATv2. (2026-03-03)
