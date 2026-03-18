# KD-GAT Architecture Decisions

> Import hierarchy: See code-style.md (enforced by tests/test_layer_boundaries.py).

## Config Architecture

3 files, single API entry point:

| File | Role |
|------|------|
| `handler.py` | `ConfigHandler` class — YAML loading, resolution, path derivation. `EnvironmentSettings` (pydantic-settings) for env var overrides. |
| `schema.py` | All Pydantic models — pipeline config, architecture sub-configs, dataset catalog entries, artifact validation contracts. |
| `__init__.py` | Singleton `_api = ConfigHandler()` + re-exports. All external code uses `from graphids.config import X`. |

- Pydantic v2 frozen BaseModels + YAML composition + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access, never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: all data lives in YAML (`pipeline.yaml`, `resources.yaml`, `datasets.yaml`). Python loads and exposes via `ConfigHandler`.
- Env vars: declared in `EnvironmentSettings(BaseSettings)` with `env_prefix="KD_GAT_"`. pydantic-settings handles type validation and override priority.
- Pipeline topology: `config/pipeline.yaml` defines model types, scales, stages, DAG dependencies, variants, preprocessing constants, path defaults, and seed defaults.
- Resolver: `ConfigHandler.resolve()` merges YAML layers (model_def → auxiliaries → CLI overrides) → `PipelineConfig.model_validate()`.
- **Config layer is inert**: no mlflow, shutil, or I/O imports. Artifact management lives in `pipeline/artifacts.py`.

> Experiment tracking (MLflow): See experiment-tracking.md.

## Artifact Management

`graphids/pipeline/artifacts.py` — cache-first lookup with filesystem and MLflow fallback.

| Function | Purpose |
|----------|---------|
| `get_artifact(cfg, stage, name, model_type=)` | Locate artifact: cache → experimentruns → MLflow download |
| `put_artifact(cfg, stage, local_path)` | Log to MLflow + populate cache |
| `artifact_exists(cfg, stage, name, model_type=)` | Check without downloading |

Used for cross-stage reads (e.g. loading VGAE checkpoint while training GAT). Same-stage writes use `stage_dir()` directly.

## Orchestration

### Dagster + SLURM (production pipeline)
Per-stage sbatch submission via Dagster assets. Each stage is a separate SLURM job with typed resource profiles from `resources.yaml`. 4 components in `graphids/pipeline/orchestration/`:

| Component | File | Role |
|-----------|------|------|
| **Job Definition** | `job.py` | Pydantic v2 frozen `ResourceSpec` (partition, GPUs, memory, walltime). |
| **DAG Topology** | `dagster_defs.py` | `build_dag_topology() → dict[str, DagNode]` — single source of truth for pipeline DAG. Used by both `build_dagster_assets()` (Dagster entry point) and `fire_and_forget()` (SLURM dependency chains). |
| **SLURM Client** | `pipes_slurm.py` | `PipesSlurmClient`: script gen via `build_cli_cmd()`, sbatch submit, sacct poll, artifact validation via Pydantic contracts. Resource profiles + failure reactions from `resources.yaml`. |
| **Retry State** | `dagster_resources.py` | Per-asset retry metadata (failure reason, node, checkpoint path) persisted to JSON for resource scaling on retry. |

**Fire-and-forget mode**: `fire_and_forget()` submits all jobs with `--dependency=afterok` chains — no polling, SLURM handles ordering. Topological sort via `graphlib.TopologicalSorter`.

**Adaptive retry**: OOM → 2× memory, TIMEOUT → 1.5× time + checkpoint resume, NODE_FAIL → exclude node. Configured in `resources.yaml` `failure_reactions` section.

CLI: `python -m graphids.pipeline.cli orchestrate --dataset hcrl_sa --seeds 42,123,456`

### HPO (Ray Tune, inside SLURM jobs)
- `tune_config.py`: Subprocess-based trainable (each trial spawns `cli.py` for CUDA isolation).
- `sweep_pipeline.py`: Multi-stage HPO sweep DAG (SQLite-backed state).
- Future: Dagster-managed Ray Tune (outer orchestration via Dagster, inner trial scheduling via Ray Tune).

### Shared Principles
- **Subprocess dispatch**: Each stage runs as `subprocess.run()` for CUDA context isolation (~300-500 MB per model). Overhead (~3-5s) is <0.1% of pipeline wall time.
- **Per-stage granularity**: Finer (per-epoch) has massive scheduling overhead; coarser (per-variant) loses restartability.
- **Checkpoint passing**: Filesystem paths, not object store (debuggable, subprocess-compatible).
- Archive restore: `cli.py` archives previous runs before re-running, restores on failure.

## Memory & Batch Sizing

- **DeviceStatsMonitor** (Lightning callback) handles GPU memory logging — no custom memory tracking code.
- **DynamicBatchSampler** (PyG) packs variable-size graphs to a node budget instead of fixed count.
- **Batch sizing**: `safety_factor × configured batch_size` (config-driven). No GPU memory probing.
- **Teacher offloading**: `cfg.training.offload_teacher_to_cpu` moves teacher to CPU between forward passes to save GPU memory. Shared helpers in `modules.py`.

## Inference Serving

`graphids/pipeline/serve.py` — FastAPI endpoints (`/predict`, `/health`) loading VGAE+GAT+DQN from `experimentruns/`. Returns fusion scores via DQN agent's `select_action()` + `_derive_scores()`.

## Dashboard

Consolidated Streamlit app at `dashboard/` (in-repo) reads from HF Datasets (auto-pushed by SLURM epilog via `scripts/data/push_experiments_to_hf.py`). Two data sources: `buckeyeguy/kd-gat-experiments` (MLflow push) + `buckeyeguy/kd-gat-sweeps` (sweep results).

Heavy analysis (UMAP, attention, CKA, etc.) lives in `notebooks/analysis/`.

## General Principles

- Delete unused code completely. No compatibility shims or `# removed` comments.
- Dataset catalog: `graphids/config/datasets.yaml` — single place to register new datasets.
- Leverage library features over custom code: Lightning callbacks, Pydantic validation, PyG batching, sklearn metrics.
