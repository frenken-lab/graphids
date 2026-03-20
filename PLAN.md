# KD-GAT Session Plan

> Last updated: 2026-03-20

## Active Plan

**Framework consolidation: Hydra-as-framework + Lightning experiment management** — see `plans/framework-consolidation.research.md`

- Phase A (done): Lightning save_hyperparameters, CSVLogger fix, EvalArtifactCallback, RunMetadataCallback
- Phase B (done): Deleted cli.py, optuna_sweep.py, subprocess_utils.py, search_spaces/. Added `__main__.py` with @hydra.main. -651 net lines.
- Phase C (done): Deleted `graphids/storage/` (1,107 lines). All I/O through Lightning + Hydra + stdlib. -1,220 net lines.
- Phase D (done): `hydra.utils.instantiate()` for callbacks + scheduler dispatch. -5 net lines.
- Phase E (done): Fixed broken scripts, updated rules files, removed ray dep, cleaned stale references. -110 net lines.

## Recently Completed

- **Framework consolidation Phase A+B** (2026-03-20) — Lightning experiment management + Hydra-as-framework. Deleted sweep code + Typer CLI (-919 lines), added @hydra.main entry points + callbacks (+348 lines). See `plans/framework-consolidation.research.md`.
- **Stage executor + submitit orchestration** (2026-03-20) — extracted `execute_stage()` as single entry point for all pipeline paths (CLI, API, notebook). Replaced Dagster + custom SLURM script generation with submitit + graphlib. Deleted dagster_defs.py, pipes_slurm.py, slurm_primitives.py (-687 production lines, -43% of pipeline/orchestration). `api.py` now has full guarantees (validation, manifest, logging, archive). See `plans/stage-executor-and-launcher.research.md`.
- **CLI move + facade enforcement + I/O leak fixes** (2026-03-19)
- **structlog integration** (2026-03-19)
- **StorageGateway + ArtifactMapper** (2026-03-19)

## In Progress

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
| 1e | Hydra CLI routing (replace argparse in cli.py) | **Done** |
| 2 | Optuna direct HPO (replace Ray Tune + sweep_pipeline + store) | **Done** |
| 3 | Extract shared slurm_client module from pipes_slurm.py | **Done** |
| 4 | ~~AdaptiveSlurmLauncher~~ — eliminated by flattened orchestration | **Eliminated** |
| 5 | Typed eval decomposition + torchmetrics condensation | **Done** |
| 5b | Lightning predict_step for eval inference | **Done** |
| 6 | SLURMEnvironment auto-requeue | **Done** |
| 7 | Simplify artifacts.py (ESS primary, drop MLflow fallback) | **Done** |
| 8 | Dagster daemon as SLURM job (CPU job + SSH tunnel to UI) | **Done** |
| 9 | Dagster partitions (multi-seed × multi-dataset + HPO trials) | Pending P8 |

## 3-Pillar Architecture (target)

| Pillar | Owner | Current state |
|--------|-------|---------------|
| **Config** | Hydra Compose + Pydantic | **Done** — 5-file config layer, Hydra config groups, lake_root-only |
| **Orchestration** | Dagster + dagster-slurm | Partial — fire_and_forget works, dagster-slurm integration pending |
| **ML Training** | Lightning modules + stages | Eval decomposed (Phase 5). CLI at `graphids/cli.py` (Phase 1e). |
| **I/O** | Lightning CSVLogger + ModelCheckpoint + callbacks | **Done** — No custom storage layer. CSVLogger for metrics, ModelCheckpoint for checkpoints, EvalArtifactCallback for eval artifacts. |

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

- **CLI move + facade enforcement + I/O leaks** — (2026-03-19)
  - Moved `graphids/pipeline/cli.py` → `graphids/cli.py` (shim removed)
  - Updated pyproject.toml entry point, subprocess_utils module string, all sbatch scripts
  - Added `compose_config` + `verify_all` to package facade re-exports
  - Fixed 20 cross-package deep imports to use facade modules
  - Routed `load_model`, temporal/fusion checkpoint loads through `ArtifactMapper`
  - DQN `load_checkpoint` accepts `dict | str | Path` (mapper-friendly)
  - Extracted `DEFAULT_LAKE_ROOT` constant, replaced hardcoded `"experimentruns"` in schema + slurm_client
  - Normalized 2 intra-package absolute imports to relative
  - 38 files changed, zero regressions
- **structlog integration** — (2026-03-19)
  - Replaced stdlib logging + MLflow with structlog processor pipeline
  - `graphids/logging.py`: configure_logging(), JSON/console renderers, stdlib bridge
  - Context binding via `structlog.contextvars` at stage entry points
  - `pipeline_run_id` correlation for cross-job tracing
- **StorageGateway + ArtifactMapper** — (2026-03-19)
  - New `graphids/storage/` layer (7 files): gateway, mapper, paths, manifest, catalog, contracts
  - NFS-safe atomic writes (tmpfile+fsync+rename), advisory locking (fcntl.flock)
  - Domain-aware serialization: checkpoints, configs, eval artifacts, cache, pickle
  - Deleted: `artifacts.py`, `eval_writers.py`, `_atomic_io.py`, `_locking.py`
  - 215 tests pass
- **Typed eval decomposition (Phase 5)** — (2026-03-18)
  - Decomposed evaluation.py (743 lines, 1 file) → 4 files (784 lines total)
  - eval_types.py: frozen dataclasses (GATResult, VGAEResult, FusionResult)
  - eval_inference.py: typed inference functions (run_gat/vgae/fusion_inference)
  - eval_writers.py: artifact writers (write_embeddings/attention/dqn_policy/cka)
  - evaluation.py: slim orchestrator + compute_metrics + probe_embedding_dim
  - MetricCollection replaces 11 individual torchmetrics instantiations
  - binary_roc replaces sklearn roc_curve (sklearn dependency eliminated)
  - Removed dead per-class manual computation (derivable from confusion matrix)
  - Fixed E2E test return contract (evaluate() returns {"metrics": ...})
  - compute_metrics and probe_embedding_dim are now public API (no underscore)
  - All 158 non-SLURM tests pass, zero regressions
- **Hydra CLI routing (Phase 1e)** — (2026-03-18)
  - Replaced argparse _build_parser() with Hydra override grammar via Compose API
  - Training: `stage=autoencoder model=vgae_large training.lr=0.001` (Hydra key=value)
  - Non-training subcommands keep per-command argparse (orchestrate, lake, tune, etc.)
  - Added show-config subcommand (replaces --cfg job)
  - sweep_id/tags/ckpt_path → EnvironmentSettings (not PipelineConfig)
  - build_cli_cmd() emits Hydra grammar for subprocess dispatch
  - Deleted ~200 lines (_build_parser, _parse_dot_overrides, multi-seed loop)
  - Fixed _hydra_bridge.py: tuple serialization + unknown dataset handling
  - 3 commits, 174 tests passing, zero regressions
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
