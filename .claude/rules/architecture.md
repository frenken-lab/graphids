# KD-GAT Architecture Decisions

> Import hierarchy: See code-style.md (enforced by tests/test_layer_boundaries.py).

## Config Architecture

5 files, Hydra Compose API:

| File | Role |
|------|------|
| `_hydra_bridge.py` | Schema-merge config composition: `resolve()` (programmatic) + `compose_config()` (CLI). |
| `constants.py` | Project constants, `load_pipeline_yaml()`, topology derivation (`STAGES`, `STAGE_DEPENDENCIES`, etc.). Leaf dependency — no config submodule imports. |
| `paths.py` | Path derivation (`stage_dir`, `checkpoint_path`, lake path primitives). `EnvironmentSettings` for SLURM, MLflow, and run metadata (sweep_id, tags, ckpt_path). |
| `schema.py` | All Pydantic models — pipeline config, architecture sub-configs, dataset catalog entries, artifact validation contracts. `Literal`-validated `model_type`/`scale`. |
| `__init__.py` | Re-exports from all submodules. All external code uses `from graphids.config import X`. |

- Pydantic v2 frozen BaseModels + Hydra config groups + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access, never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: topology data lives in `pipeline.yaml`, loaded by `constants.py`. Preprocessing constants are module-level in `constants.py`.
- Env vars: Path vars (`lake_root`) flow through Hydra `oc.env` → `PipelineConfig`. Infrastructure + run metadata use `EnvironmentSettings` in `paths.py` (`env_prefix="KD_GAT_"`). Run metadata (sweep_id, tags, ckpt_path) lives in EnvironmentSettings, NOT PipelineConfig — doesn't affect config hash.
- Pipeline topology: `config/pipeline.yaml` defines model types, scales, stages, DAG dependencies. Default stages and variants live in `config/conf/config.yaml`.
- **Schema-merge composition**: `_hydra_bridge.py` composes Hydra config groups only → builds full-field schema from PipelineConfig defaults → `OmegaConf.merge(schema, hydra)` → applies nested overrides with `force_add=False` (typo detection) → `PipelineConfig.model_validate()`. Two entry points: `resolve()` (programmatic), `compose_config()` (CLI, returns DictConfig + stage).
- Hydra config groups: `conf/model/` (6 files), `conf/auxiliary/` (2 files), `conf/dataset/` (6 files). Each uses `@package _global_` to merge at root.
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

### HPO (Optuna, inside SLURM jobs)
- `optuna_sweep.py`: Single file — `run_sweep()` (single-stage Optuna study) + `run_sweep_pipeline()` (sequential 3-stage loop). Subprocess-based objective for CUDA isolation. Optuna's built-in SQLite storage provides free resume.

### Shared SLURM module
- `slurm_client.py`: Generic SLURM primitives — `generate_sbatch_script()`, `submit_sbatch()`, `sacct_query()`, `poll_until_done()`, resource profiles, adaptive retry (`scale_resources()`), `SlurmJobFailed`. No Dagster imports.
- `pipes_slurm.py`: Thin Dagster wrapper — `PipesSlurmClient` (script file management, artifact validation, checkpoint discovery). Imports and re-exports from `slurm_client.py`.

### Shared Principles
- **Subprocess dispatch**: Each stage runs as `subprocess.run()` for CUDA context isolation (~300-500 MB per model). Overhead (~3-5s) is <0.1% of pipeline wall time.
- **Per-stage granularity**: Finer (per-epoch) has massive scheduling overhead; coarser (per-variant) loses restartability.
- **Checkpoint passing**: Filesystem paths, not object store (debuggable, subprocess-compatible).
- Archive restore: `cli.py` archives previous runs before re-running, restores on failure. Lifecycle extracted into `_archive_previous()`, `_log_stage_artifacts()`, `_write_lake_manifest()`.

## Evaluation

4 files under `graphids/pipeline/stages/`:

| File | Role |
|------|------|
| `evaluation.py` | Orchestrator (`evaluate()`), per-model evaluators, `compute_metrics`, `probe_embedding_dim` |
| `eval_types.py` | Frozen dataclasses: `GATResult`, `VGAEResult`, `FusionResult` |
| `eval_inference.py` | Typed inference: `run_gat_inference`, `run_vgae_inference`, `run_fusion_inference` |
| `eval_writers.py` | Artifact I/O: `write_embeddings`, `write_attention`, `write_dqn_policy`, `write_cka` |

- **Batched inference**: `run_gat_inference()` and `run_vgae_inference()` use Lightning `trainer.predict()` via `_GATPredictor`/`_VGAEPredictor` wrappers (batch_size=128). GATConv return type workaround: predict_step always uses `return_embedding=True` (consistent type); attention capture stays in separate manual `_capture_attention()` loop (50 samples, uses `return_attention_weights=True` which changes GATConv output type). VGAE component capture falls back to per-sample.
- **Metrics**: `compute_metrics()` uses `torchmetrics.MetricCollection` (GPU-native, no sklearn). Custom: detection-at-FPR, Youden's J via `torchmetrics.functional.binary_roc`.
- **CKA**: Self-contained in `eval_writers.py` (loads models, computes linear CKA, writes JSON).

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
- Leverage library features over custom code: Lightning callbacks, Pydantic validation, PyG batching, torchmetrics.
