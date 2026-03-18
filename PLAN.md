# KD-GAT Session Plan

> Last updated: 2026-03-18

## Active Plan

**Pipeline toolchain migration** — replace custom management layer (84% of pipeline/) with Hydra + Dagster + Lightning.

- Decision: `~/plans/pipeline-toolchain-decision.md`
- Research: `~/plans/pipeline-decoupling-analysis.md`
- Phase 1 design: `~/plans/phase1-hydra-config.md`

## In Progress

- **Toolchain migration implementation** — executing tool-agnostic structural changes
- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces

## Blocked

(none)

## Migration Phases

| Phase | What | Status |
|-------|------|--------|
| 1 | Config framework migration (LightningCLI evaluated, spike pending) | **Designing** |
| 2 | Hydra Optuna sweeper (replace sweep_pipeline/tune_config/store) | Pending P1 |
| 3 | Extract shared slurm_client module | Pending |
| 4 | AdaptiveSlurmLauncher Hydra plugin | Pending P1+P3 |
| 5 | Lightning predict_step + BasePredictionWriter for eval | Pending |
| 6 | SLURMEnvironment auto-requeue | Pending |
| 7 | Simplify artifacts.py (ESS primary, drop MLflow fallback) | Pending |
| 8 | dagster-slurm eval (WSL → OSC SSH) | Pending |
| 9 | Dagster partitions (multi-seed × multi-dataset) | Pending P8 |

## Open Questions

- Hydra maintenance risk — last stable 1.3.2 (Feb 2023). hydra-zen compensates. Compose API is escape hatch.
- submitit pickles in-process — conflicts with CUDA isolation. Need subprocess inside trainable.
- Manifest convergence timing — do before or after Hydra? (~1 day, independent)

## Next Up (after toolchain migration)

- Fusion method comparison experiment
- Evaluate research questions R1–R3

## Key Reference Documents

| Document | Purpose |
|----------|---------|
| `~/plans/pipeline-toolchain-decision.md` | Chosen toolchain + migration order + database convergence |
| `~/plans/pipeline-decoupling-analysis.md` | 5,852-line decomposition: 16% ML / 84% management |
| `~/plans/phase1-hydra-config.md` | Phase 1 detailed design |
| `~/plans/fusion-redesign.md` | RL fusion analysis |
| `~/plans/ecosystem-component-registry.md` | 24-component grocery list with interfaces and gaps |

## Completed

- **Structural migration (5 tasks)** — (2026-03-18)
  - Dead code removal: `list_models()`, `list_auxiliaries()`, `sweep_searcher_path()`, Optuna pickle save
  - Manifest convergence: `metrics` field on `Manifest` (SoT), catalog reads from manifest with recursive flattening (fixes eval metrics gap)
  - Split `pipeline.yaml`: topology-only (removed preprocessing constants, defaults, paths → module-level constants in handler.py)
  - Plan artifact schema: `PlanJob`, `ArtifactDependency`, `Plan` Pydantic models + `build_plan()` in `orchestration/plan.py`. Verified: 12 jobs for 3-seed large variant, correct deps + resources, JSON round-trip.
  - Config concern separation: preprocessing constants elevated to module-level in handler.py (no longer read from YAML at runtime). Path extraction deferred — too many callers for a no-behavior-change refactor.
- **Pipeline decoupling analysis** — 5,852 lines classified (932 ML, 4,920 management). 180 functions: 20 ML, 55 replaceable, 105 glue. Toolchain decision made. (2026-03-18)
- **Pipeline layer consolidation v2** — 4 phases: bugs+config, torchmetrics, batched eval (10-50x speedup), god function decomposition. (2026-03-17)
- **Preprocessing module hardening** — 6 fixes: ghost config param, adapter serialization, IR validation, feature manifest, SRP split. (2026-03-17)
- **Models layer hardening** — decouple extractor, consolidate conv, typed layout. (2026-03-17)
- **Architecture review & ecosystem mapping** — 50 files / 9,540 lines inventoried. 24 ecosystem components. (2026-03-11)
- Codebase consolidation: 12,511→9,537 lines (-24%), 55→50 files. (2026-03-10)
- Memory/batch sizing simplification: ~600 lines removed. (2026-03-07)
- MLflow migration: replaces W&B + lakehouse + CSVLogger. (2026-03-06)
- Feature engineering v2.0.0: 11→26-D node features, GATv2. (2026-03-03)
