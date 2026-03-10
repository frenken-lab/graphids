# KD-GAT: Project Context

**Updated**: 2026-03-07

## What This Is

CAN bus intrusion detection via knowledge distillation. Large models (VGAE → GAT → DQN fusion) are compressed into small models via KD auxiliaries. Optional temporal stage adds cross-window sequence classification. Runs on OSC HPC via Ray/SLURM with MLflow tracking.

## Architecture

```
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion of both)
                                          ↑                      ↓
                                     EVALUATION ←────── TEMPORAL (optional: GAT + Transformer)
```

**Entry point**: `python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`

## Layered Architecture

Three-layer import hierarchy (enforced by `tests/test_layer_boundaries.py`):

### Layer 1: `graphids/config/` (inert, declarative — no pipeline/ or core/ imports)

- `schema.py` — Pydantic v2 frozen models: `PipelineConfig`, `VGAEArchitecture`, `GATArchitecture`, `DQNArchitecture`, `AuxiliaryConfig`, `TrainingConfig`, `FusionConfig`, `PreprocessingConfig`, `TemporalConfig`. Legacy flat JSON loading via `_from_legacy_flat()`.
- `resolver.py` — YAML composition: `resolve(model_type, scale, auxiliaries="none", **cli_overrides)`. Merge order: defaults → model_def → auxiliaries → CLI. Pydantic v2 `model_validate()` handles schema validation.
- `paths.py` — Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`. String-based variants for flow tasks.
- `constants.py` — Domain/infrastructure constants: feature counts, window sizes, stage→model mappings
- `__init__.py` — Re-exports for clean `from graphids.config import PipelineConfig, resolve, checkpoint_path` usage
- `defaults.yaml` — Global baseline config values
- `datasets.yaml` — Dataset catalog (6 automotive datasets)
- `models/{vgae,gat,dqn}/{large,small}.yaml` — Architecture × Scale definitions
- `auxiliaries/{none,kd_standard}.yaml` — Loss modifier configs (composable)

### Layer 2: `graphids/pipeline/` (orchestration — imports graphids.config freely, lazy imports from graphids.core)

- `cli.py` — Arg parser, MLflow run context (`mlflow.start_run()`), artifact logging, `STAGE_FNS` dispatch, archive restore on failure
- `serve.py` — FastAPI inference server (`/predict`, `/health`) with DQN fusion scoring
- `validate.py` — Config + environment validation utilities
- `stages/` — Training logic split into modules:
  - `training.py` — VGAE (autoencoder) and GAT (curriculum/normal) training
  - `evaluation.py` — Multi-model eval; captures `embeddings.npz`, `dqn_policy.json`
  - `fusion.py` — DQN/MLP/WeightedAvg fusion training
  - `temporal.py` — Temporal graph classification (GAT encoder + Transformer)
  - `data_loading.py` — Dataset loading + graph caching + `training_preamble()`
  - `batch_sizing.py` — Batch size resolution (`safety_factor × batch_size`)
  - `trainer_factory.py` — Lightning Trainer + ModelCheckpoint + EarlyStopping + DeviceStatsMonitor + MLflow autolog
  - `modules.py` — Lightning modules: VGAEModule, GATModule, CurriculumDataModule + teacher offload helpers
  - `loss_landscape.py` — Loss landscape visualization (standalone analysis tool)
  - `utils.py` — Re-exports from submodules
- `orchestration/` — Pipeline orchestration (Ray + scheduler-agnostic):
  - `job.py` — Pydantic v2 frozen models: `JobSpec` (UUID-based DAG), `ResourceSpec`, `JobState`
  - `planner.py` — Domain-aware DAG builder: `build_plan(datasets, seeds, variants) → list[JobSpec]`
  - `store.py` — SQLite state store (WAL mode): run/job/attempt/transition tables
  - `executor.py` — Scheduler backends: `SlurmExecutor`, `FluxExecutor`, `DryRunExecutor`
  - `driver.py` — `PipelineDriver`: submit-and-poll loop, fire-and-forget, retry with resource scaling
  - `ray_pipeline.py` — Config-driven variant pipeline, subprocess dispatch, benchmark mode
  - `ray_slurm.py` — Ray cluster bootstrap on SLURM
  - `sweep_pipeline.py` — Hyperparameter sweep orchestration (SQLite-backed state)
  - `tune_config.py` — Ray Tune search space + OptunaSearch + ASHAScheduler

### Layer 3: `graphids/core/` (domain — imports graphids.config.constants, never imports graphids.pipeline)

- `models/` — vgae.py, gat.py, dqn.py, temporal.py, fusion_features.py, registry.py, _utils.py
- `training/datamodules.py` — Lightning DataModule: dataset loading, splits, DataLoader construction
- `preprocessing/` — Graph construction: dataset.py, engine.py, temporal.py, vocabulary.py, parallel.py, schema.py
- `preprocessing/adapters/` — base.py, can_bus.py (pluggable data source adapters)

## Model Registry

`graphids/core/models/registry.py` centralizes model construction and fusion feature extraction.

| Model | Feature Dim | Extractor | Role |
|-------|-------------|-----------|------|
| `vgae` | 8-D | `VGAEFusionExtractor` | Errors + latent stats + confidence |
| `gat` | 7-D | `GATFusionExtractor` | Class probs + embedding stats + confidence |
| `dqn` | — | None | Consumes features (15-D state) |

## Config System

Config defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers), **dataset**.

```python
from graphids.config import resolve, PipelineConfig
cfg = resolve("vgae", "large", dataset="hcrl_sa")          # No KD
cfg = resolve("gat", "small", auxiliaries="kd_standard")    # With KD
cfg.vgae.latent_dim    # Nested sub-config access
cfg.training.lr        # Training hyperparameters
cfg.has_kd             # Property: any KD auxiliary?
cfg.kd.temperature     # KD auxiliary config (via property)
cfg.active_arch        # Architecture config for active model_type
```

## Data Pipeline

```
graphids/config/datasets.yaml       # Dataset catalog (source of truth)
     ↓
data/automotive/{dataset}/train_*/  →  data/cache/{dataset}/processed_graphs.pt
     (raw CSVs, DVC-tracked)              (PyG Data objects, DVC-tracked)
```

6 datasets: hcrl_ch, hcrl_sa, set_01-04. Cache auto-built on first access.

## Models

| Model | File | Large | Small | Ratio |
|-------|------|-------|-------|-------|
| `GraphAutoencoderNeighborhood` | `vgae.py` | (480,240,48) latent 48 | (80,40,16) latent 16 | ~4x |
| `GATWithJK` | `gat.py` | hidden 48, 3 layers, 8 heads | hidden 24, 2 layers, 4 heads | 5.3x |
| `EnhancedDQNFusionAgent` | `dqn.py` | hidden 576, 3 layers | hidden 160, 2 layers | ~13x |
| `TemporalGraphClassifier` | `temporal.py` | GAT + 2-layer Transformer | — | opt-in |

## Memory Optimization

- `DeviceStatsMonitor` (Lightning callback) — GPU memory/utilization logging
- `DynamicBatchSampler` (PyG) — node-budget batching for variable-size graphs
- Batch sizing: `safety_factor × configured batch_size` (config-driven, no GPU probing)
- Teacher offloading: `cfg.training.offload_teacher_to_cpu` frees GPU between KD forward passes
- `gradient_checkpointing: True` — 30-50% activation memory savings
- `precision: "16-mixed"` — 50% model/activation memory reduction

## Experiment Management

**MLflow** (SQLite backend at `data/mlflow/mlflow.db`):
- `cli.py` wraps each stage in `mlflow.start_run()` with dataset/model/stage/scale tags
- `trainer_factory.py` uses `mlflow.pytorch.autolog()` for per-epoch metrics
- Export to HF Dataset: `scripts/data/push_experiments_to_hf.py` (auto via SLURM epilog)

**Filesystem** (NFS home, permanent):
```
experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]/
├── best_model.pt       # Model checkpoint
├── config.json         # Frozen config (Pydantic JSON)
├── metrics.json        # Training/evaluation metrics
├── embeddings.npz      # VGAE z-mean + GAT hidden layers
├── dqn_policy.json     # DQN alpha values + class breakdown
├── attention_weights.npz # GAT attention head weights
```

## Critical Constraints

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`.
- **Use spawn multiprocessing.** Fork + CUDA = crashes.
- **NFS filesystem.** `.nfs*` ghost files on delete.
- **Never run pytest on login nodes.** Submit via `bash scripts/slurm/run_tests_slurm.sh`.

## Environment

- **Cluster**: OSC Pitzer, RHEL 9, SLURM scheduler
- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Python**: uv venv `.venv/` (`source ~/KD-GAT/.venv/bin/activate`)
- **Key packages**: PyTorch 2.8.0+cu128, PyG 2.7.0, Lightning, Pydantic v2, MLflow, Ray
- **SLURM account**: PAS1266, gpu partition, V100 GPUs
