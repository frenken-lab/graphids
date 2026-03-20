# KD-GAT Architecture Decisions

> Import hierarchy: See code-style.md.

## Config Architecture

4 files, Hydra Compose API:

| File | Role |
|------|------|
| `_hydra_bridge.py` | Schema-merge config composition: `resolve()` (programmatic). |
| `constants.py` | Project constants, `load_pipeline_yaml()`, topology derivation (`STAGES`, `STAGE_DEPENDENCIES`, etc.). Leaf dependency — no config submodule imports. |
| `paths.py` | Lake path primitives (`lake_cache_dir`, `lake_raw_dir`, `lake_exports_dir`). `EnvironmentSettings` for SLURM and run metadata. |
| `schema.py` | All Pydantic models — pipeline config, architecture sub-configs, dataset catalog entries. `Literal`-validated `model_type`/`scale`. |

- Pydantic v2 frozen BaseModels + Hydra config groups + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access, never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: topology data lives in `pipeline.yaml`, loaded by `constants.py`. Preprocessing constants are module-level in `constants.py`.
- Env vars: Path vars (`lake_root`) flow through Hydra `oc.env` → `PipelineConfig`. Infrastructure + run metadata use `EnvironmentSettings` in `paths.py` (`env_prefix="KD_GAT_"`).
- Pipeline topology: `config/pipeline.yaml` defines model types, scales, stages, DAG dependencies. Default stages and variants live in `config/conf/config.yaml`.
- **Schema-merge composition**: `_hydra_bridge.py` composes Hydra config groups only → builds full-field schema from PipelineConfig defaults → `OmegaConf.merge(schema, hydra)` → applies nested overrides with `force_add=False` (typo detection) → `PipelineConfig.model_validate()`. Entry point: `resolve()` (programmatic/test).
- Hydra config groups: `conf/model/` (6 files), `conf/auxiliary/` (2 files), `conf/dataset/` (6 files). Each uses `@package _global_` to merge at root.
- **Config layer is inert**: no mlflow, shutil, or I/O imports.

## CLI

Single entry point: `python -m graphids` (`__main__.py`).

- **Training/sweep**: `@hydra.main` — `python -m graphids stage=autoencoder model=vgae_large`. `--multirun` enables sweeper plugins.
- **Subcommands**: `orchestrate`, `lake`, `preprocess` use argparse (non-Hydra).
- **Output dirs**: `hydra.run.dir` template + `hydra.job.chdir: true` — Hydra creates the run directory and cd's into it. Lightning writes to cwd.
- **Checkpoint paths**: OmegaConf interpolation in `cfg.checkpoints[model_type]` — no Python path functions.
- **Callbacks**: Instantiated from YAML `_target_` entries via `hydra.utils.instantiate()`.

## Orchestration

submitit + graphlib for SLURM dependency chains. 3 files in `graphids/pipeline/orchestration/`:

| Component | File | Role |
|-----------|------|------|
| **Job Definition** | `job.py` | Pydantic v2 frozen `ResourceSpec` (partition, GPUs, memory, walltime). |
| **DAG Topology** | `dag.py` | `build_dag_topology()` + `run_dag()` — graphlib `TopologicalSorter` for ordering, submitit for SLURM submission with `--dependency=afterok` chains. |
| **SLURM Executor** | `slurm.py` | submitit executor factory with resource profiles from `resources.yaml`. |

CLI: `python -m graphids orchestrate --dataset hcrl_sa --seeds 42,123,456`

## Evaluation

3 files under `graphids/pipeline/stages/`:

| File | Role |
|------|------|
| `evaluation.py` | Orchestrator (`evaluate()`), per-model evaluators, `compute_metrics`, `probe_embedding_dim` |
| `eval_types.py` | Frozen dataclasses: `GATResult`, `VGAEResult`, `FusionResult` |
| `eval_inference.py` | Typed inference: `run_gat_inference`, `run_vgae_inference`, `run_fusion_inference` |

Eval artifacts (embeddings, attention, DQN policy) saved via `EvalArtifactCallback` in `callbacks.py`.

- **Batched inference**: `run_gat_inference()` and `run_vgae_inference()` use Lightning `trainer.predict()` via predictor wrappers.
- **Metrics**: `compute_metrics()` uses `torchmetrics.MetricCollection` (GPU-native, no sklearn).
- **CKA**: Self-contained in `pipeline/stages/cka.py`.

## Memory & Batch Sizing

- **DeviceStatsMonitor** (Lightning callback, instantiated from YAML) handles GPU memory logging.
- **DynamicBatchSampler** (PyG) packs variable-size graphs to a node budget instead of fixed count.
- **Batch sizing**: `safety_factor × configured batch_size` (config-driven, `batch_sizing.py`).
- **Teacher offloading**: `cfg.training.offload_teacher_to_cpu` moves teacher to CPU between forward passes. Shared helpers in `modules.py`.

## Experiment Data

- **Per-run outputs**: `metrics.csv` (CSVLogger), `hparams.yaml` (save_hyperparameters), `best_model.pt` (ModelCheckpoint), `run_metadata.json` (RunMetadataCallback).
- **Dashboard data**: `scripts/data/push_experiments_to_hf.py` reads `metrics.csv` + `hparams.yaml`, writes to HF Dataset.
- **No manifest system**: CSVLogger output is the single source of truth for metrics.

## Logging

structlog with stdlib bridge. One config call at process startup, structured events everywhere.

- All loggers: `import structlog; log = structlog.get_logger()`
- Structured events: `log.info("event_name", key=value)` — no format strings
- Context binding: `structlog.contextvars.bind_contextvars(dataset=..., model=..., stage=...)` at stage entry
- JSON mode: `--json-logs` CLI flag or `KD_GAT_JSON_LOGS=1` env var

## General Principles

- Delete unused code completely. No compatibility shims or `# removed` comments.
- Dataset catalog: `graphids/config/datasets.yaml` — single place to register new datasets.
- Leverage library features over custom code: Lightning callbacks, Pydantic validation, PyG batching, torchmetrics, Hydra instantiate.
