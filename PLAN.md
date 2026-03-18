# KD-GAT Session Plan

> Last updated: 2026-03-18

## Active Plan

**Pipeline toolchain migration** — replace custom management layer (84% of pipeline/) with Hydra + Dagster + Lightning.

- Decision: `~/plans/pipeline-toolchain-decision.md`
- Research: `~/plans/pipeline-decoupling-analysis.md`

## In Progress

- **Toolchain migration** — config layer complete, CLI + orchestration + ML decoupling next
- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces

## Blocked

(none)

## Migration Phases

| Phase | What | Status |
|-------|------|--------|
| 1a | Hydra Compose API config composition (replace ConfigHandler.resolve) | **Done** |
| 1b | Decompose config layer (constants.py, paths.py, _hydra_bridge.py) | **Done** |
| 1c | Collapse to lake_root-only paths, dissolve LakeConfig | **Done** |
| 1d | Dissolve lake/ package (manifest→pipeline, catalog→pipeline, locking→core) | **Done** |
| 1e | Hydra CLI routing (replace argparse in cli.py) | **Next** |
| 2 | Hydra Optuna sweeper (replace sweep_pipeline/tune_config/store) | Pending P1e |
| 3 | dagster-slurm (replaces bulk of pipes_slurm.py) | **Decided** (Option A) |
| 4 | ~~AdaptiveSlurmLauncher~~ — eliminated by flattened orchestration | **Eliminated** |
| 5 | Lightning predict_step + BasePredictionWriter for eval | Pending |
| 6 | SLURMEnvironment auto-requeue | Pending |
| 7 | Simplify artifacts.py (ESS primary, drop MLflow fallback) | Pending |
| 8 | dagster-slurm + pixi-pack (WSL daemon → OSC SSH → pixi env) | **Decided** (Option A) |
| 9 | Dagster partitions (multi-seed × multi-dataset + HPO trials) | Pending P8 |

## 3-Pillar Architecture (target)

| Pillar | Owner | Current state |
|--------|-------|---------------|
| **Config** | Hydra Compose + Pydantic | **Done** — 5-file config layer, Hydra config groups, lake_root-only |
| **Orchestration** | Dagster + dagster-slurm | Partial — fire_and_forget works, dagster-slurm integration pending |
| **ML Training** | Lightning modules + stages | Coupled — 932 lines ML in 3,000 lines of glue. Needs predict_step, CLI simplification |
| **I/O** | pipeline/manifest + catalog + artifacts | Reorganized — manifest+catalog in pipeline/, locking in core/ |

## Open Questions

- **Adaptive retry hooks**: dagster-slurm doesn't classify SLURM failures (OOM vs TIMEOUT). How to hook `scale_resources()` into Dagster's `RetryPolicy`?
- **pixi-pack + OSC CUDA**: Verify PyTorch conda packages with bundled cudatoolkit work on OSC GPU nodes.

## Decisions Made

- **Hydra Compose API** (2026-03-18): hydra-core + omegaconf (no hydra-zen). Config groups in conf/model/, conf/auxiliary/, conf/dataset/. Same resolve() signature, Hydra internals.
- **lake_root-only paths** (2026-03-18): Single storage root. experiment_root, data_root, cache_root, stage_dir_override eliminated. KD_GAT_LAKE_ROOT (default: experimentruns).
- **lake/ dissolved** (2026-03-18): manifest.py → pipeline/, catalog.py → pipeline/, locking.py → core/preprocessing/_locking.py.
- **Flattened orchestration** (2026-03-18): Dagster owns all SLURM submission. Phase 4 eliminated.
- **dagster-slurm Option A** (2026-03-18): Bash launcher + pixi-pack + custom preamble.
- **Manifest as metrics SoT** (2026-03-18): _manifest.json is sole source of truth.

## Next Up (after toolchain migration)

- Fusion method comparison experiment
- Evaluate research questions R1–R3

## Key Reference Documents

| Document | Purpose |
|----------|---------|
| `~/plans/pipeline-toolchain-decision.md` | Chosen toolchain + migration order |
| `~/plans/pipeline-decoupling-analysis.md` | 5,852-line decomposition: 16% ML / 84% management |
| `~/plans/phase1-hydra-spike.md` | Phase 1 detailed design (spike portion complete) |
| `~/plans/fusion-redesign.md` | RL fusion analysis |
| `~/plans/ecosystem-component-registry.md` | 24-component grocery list |

## Completed

- **Hydra config migration + config layer refactor** — (2026-03-18)
  - Replaced ConfigHandler with Hydra Compose API (_hydra_bridge.py)
  - Decomposed handler.py → constants.py + paths.py (no overlap, clear ownership)
  - Unified env vars: path vars via Hydra oc.env, SLURM/MLflow via pydantic-settings
  - Moved stages/variants from pipeline.yaml shadow load to Hydra config.yaml
  - Literal-validated model_type/scale (replaces runtime _check_cross_field validator)
  - Collapsed to lake_root-only paths (removed experiment_root, data_root, cache_root, stage_dir_override)
  - Dissolved LakeConfig class → standalone lake_* functions in config/paths.py
  - Dissolved lake/ package → manifest+catalog to pipeline/, locking to core/
  - Deleted: handler.py, lake/config.py, lake/__init__.py, old models/ + auxiliaries/ dirs
  - Added: hydra-core, omegaconf, pydantic-settings deps. Removed hydra-zen.
  - Updated rules files (architecture.md, config-system.md, project-structure.md, code-style.md)
  - 5 commits, 161 tests passing, zero regressions
- **Plan CLI + manifest convergence phase B** — (2026-03-18)
- **Structural migration (5 tasks)** — (2026-03-18)
- **Pipeline decoupling analysis** — (2026-03-18)
- **Pipeline layer consolidation v2** — (2026-03-17)
- **Preprocessing module hardening** — (2026-03-17)
- **Models layer hardening** — (2026-03-17)
- **Architecture review & ecosystem mapping** — (2026-03-11)
- Codebase consolidation: 12,511→9,537 lines (-24%), 55→50 files. (2026-03-10)
- Memory/batch sizing simplification: ~600 lines removed. (2026-03-07)
- MLflow migration: replaces W&B + lakehouse + CSVLogger. (2026-03-06)
- Feature engineering v2.0.0: 11→26-D node features, GATv2. (2026-03-03)
