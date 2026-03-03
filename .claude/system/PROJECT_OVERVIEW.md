# KD-GAT: Project Context

**Updated**: 2026-02-23

## What This Is

CAN bus intrusion detection via knowledge distillation. Large models (VGAE → GAT → DQN fusion) are compressed into small models via KD auxiliaries. Optional temporal stage adds cross-window sequence classification. Runs on OSC HPC via Ray/SLURM with W&B tracking.

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

- `schema.py` — Pydantic v2 frozen models: `PipelineConfig`, `VGAEArchitecture`, `GATArchitecture`, `DQNArchitecture`, `AuxiliaryConfig`, `TrainingConfig`, `FusionConfig`, `PreprocessingConfig`, `TemporalConfig`. Legacy flat JSON loading via `_from_legacy_flat()` (for old config.json files — all dirs now use new naming).
- `resolver.py` — YAML composition: `resolve(model_type, scale, auxiliaries="none", **cli_overrides)`, `list_models()`, `list_auxiliaries()`. Merge order: defaults → model_def → auxiliaries → CLI.
- `paths.py` — Path layout: `{dataset}/{model_type}_{scale}_{stage}[_{aux}]`. String-based variants for flow tasks.
- `constants.py` — Domain/infrastructure constants: feature counts, window sizes, SLURM defaults, memory limits
- `__init__.py` — Re-exports for clean `from graphids.config import PipelineConfig, resolve, checkpoint_path` usage
- `defaults.yaml` — Global baseline config values
- `datasets.yaml` — Dataset catalog (6 automotive datasets)
- `models/{vgae,gat,dqn}/{large,small}.yaml` — Architecture × Scale definitions
- `auxiliaries/{none,kd_standard}.yaml` — Loss modifier configs (composable)

### Layer 2: `graphids/pipeline/` (orchestration — imports graphids.config freely, lazy imports from graphids.core)

- `cli.py` — Arg parser (`--model`/`--scale`/`--auxiliaries`), W&B run lifecycle (`wandb.init`/`wandb.finish`), S3 lakehouse sync, `STAGE_FNS` dispatch
- `stages/` — Training logic split into modules:
  - `training.py` — VGAE (autoencoder) and GAT (curriculum) training
  - `fusion.py` — DQN fusion training (uses `cfg.dqn.*`, `cfg.fusion.*`)
  - `evaluation.py` — Multi-model evaluation and metrics; captures `embeddings.npz`, `dqn_policy.json`, `explanations.npz` (when `run_explainer=True`) as artifacts; optional temporal model evaluation
  - `temporal.py` — Temporal graph classification stage (GAT spatial encoder + Transformer temporal head)
  - `modules.py` — PyTorch Lightning modules (uses `cfg.vgae.*`, `cfg.gat.*`, `cfg.training.*`)
  - `utils.py` — Shared utilities: model loading with cross-model path resolution (`_cross_model_path`, `_STAGE_MODEL_TYPE`), batch size optimization (static/measured/trial), trainer construction, WandbLogger setup
- `orchestration/` — Ray orchestration:
  - `ray_pipeline.py` — Full training DAG via Ray remote tasks
  - `ray_slurm.py` — Ray cluster bootstrap on SLURM
- `validate.py` — Config validation (simplified — Pydantic handles field constraints)
- `tracking.py` — Memory monitoring utilities
- `memory.py` — GPU memory management: static estimation, measured (forward hooks), trial-based (binary search with forward+backward passes)
- `lakehouse.py` — Fire-and-forget Parquet append to data/datalake/
- `export.py` — Filesystem scanning → static JSON/Parquet export for Quarto reports

### Layer 3: `graphids/core/` (domain — imports graphids.config.constants, never imports graphids.pipeline)

## Model Registry

`graphids/core/models/registry.py` centralizes model construction and fusion feature extraction.

**Registered models** (order matters — determines 15-D DQN state layout):

| Model | Feature Dim | Extractor | Role |
|-------|-------------|-----------|------|
| `vgae` | 8-D | `VGAEFusionExtractor` | Errors + latent stats + confidence |
| `gat` | 7-D | `GATFusionExtractor` | Class probs + embedding stats + confidence |
| `dqn` | — | None | Consumes features (15-D state) |

**Usage**:
```python
from graphids.core.models import get, fusion_state_dim, extractors

entry = get("vgae")                  # ModelEntry(model_type, factory, extractor)
model = entry.factory(cfg, num_ids, in_ch)  # Construct model from config
dim = fusion_state_dim()             # 15 (sum of all extractor dims)
pairs = extractors()                 # [("vgae", ext), ("gat", ext)] in registration order
```

**Adding a new model**: Register a `ModelEntry` in `registry.py` with a factory function (lazy import to avoid circular deps) and an optional `FusionFeatureExtractor` implementation.

## Supporting Code: `graphids/core/`

`graphids/pipeline/stages/` imports from these `graphids/core/` modules:
- `graphids.core.models.vgae`, `graphids.core.models.gat`, `graphids.core.models.dqn`, `graphids.core.models.temporal` — model architectures
- `graphids.core.explain` — GNNExplainer feature importance analysis
- `graphids.core.training.datamodules` — `load_dataset()`, `load_test_scenarios()`
- `graphids.core.preprocessing.preprocessing` — `GraphDataset`, graph construction
- `graphids.core.preprocessing.temporal` — `TemporalGrouper`, `GraphSequence` (sliding window over ordered graphs)

`load_dataset()` accepts direct `Path` arguments from `graphids/pipeline/paths.py`. No legacy adapters remain.

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

**Resolution order**: `defaults.yaml` → `models/{type}/{scale}.yaml` → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen.

**Cross-model loading**: `load_vgae(gat_cfg)` resolves to `vgae_*` paths via `_STAGE_MODEL_TYPE` mapping. Each stage has a canonical model owner (autoencoder→vgae, curriculum→gat, fusion→dqn).

## Data Pipeline

```
graphids/config/datasets.yaml       # Dataset catalog (source of truth for dataset metadata)
     ↓
data/automotive/{dataset}/train_*/  →  data/cache/{dataset}/processed_graphs.pt
     (raw CSVs, DVC-tracked)              (PyG Data objects, DVC-tracked)
                                          + id_mapping.pkl
                                          + cache_metadata.json
                                          + test_*.pt (per-scenario test graphs)
```

- 6 datasets: hcrl_ch, hcrl_sa, set_01-04
- Cache auto-built on first access, validated via metadata on subsequent loads
- All data versioned with DVC (remotes: local scratch + AWS S3)

## Models

| Model | File | Large | Small | Ratio |
|-------|------|-------|-------|-------|
| `GraphAutoencoderNeighborhood` | `graphids/core/models/vgae.py` | (480,240,48) latent 48 | (80,40,16) latent 16 | ~4x |
| `GATWithJK` | `graphids/core/models/gat.py` | hidden 48, 3 layers, 8 heads, fc_layers 1 (343k) | hidden 24, 2 layers, 4 heads, fc_layers 2 (65k) | 5.3x |
| `EnhancedDQNFusionAgent` | `graphids/core/models/dqn.py` | hidden 576, 3 layers | hidden 160, 2 layers | ~13x |
| `TemporalGraphClassifier` | `graphids/core/models/temporal.py` | Shared GAT + 2-layer Transformer (hidden 64, 4 heads) | — | opt-in |

DQN state: 15D vector (VGAE 8D: errors + latent stats + confidence; GAT 7D: logits + embedding stats + confidence).

Temporal model wraps a pretrained GAT spatial encoder (optionally frozen) with a `nn.TransformerEncoder` over sequences of graph embeddings. Not part of the DQN fusion state — it's a separate classification path for slow-onset attack detection.

## Memory Optimization

Default config enables memory-efficient training:
- `gradient_checkpointing: True` — 30-50% activation memory savings (~20% compute overhead)
- `precision: "16-mixed"` — 50% model/activation memory reduction
- Both `GATWithJK` and `GraphAutoencoderNeighborhood` support checkpointing via `use_checkpointing` flag

## Critical Constraints

**Do not violate these — they fix real crashes:**

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** `mp_start_method: "spawn"` in config, `mp.set_start_method('spawn', force=True)` in CLI. Fork + CUDA = crashes.
- **DataLoader workers**: `multiprocessing_context='spawn'` on all DataLoader instances.
- **NFS filesystem**: `.nfs*` ghost files appear when processes delete open files. Already in `.gitignore`.
- **No GUI on HPC**: Git auth via SSH key (configured), not HTTPS tokens.

## Experiment Management

**Two-layer architecture**: W&B owns live tracking (params/metrics/UI), filesystem owns artifacts and frozen configs.

**Filesystem** (NFS home, permanent):
```
experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]/
├── best_model.pt       # Model checkpoint
├── config.json         # Frozen config (Pydantic JSON)
├── metrics.json        # Evaluation stage only
├── embeddings.npz      # Evaluation artifact: VGAE z-mean + GAT hidden representations
├── dqn_policy.json     # Evaluation artifact: DQN alpha values + class breakdown
├── explanations.npz    # Evaluation artifact: GNNExplainer feature importance (when run_explainer=True)
├── lightning_logs/     # Training logs (per-epoch metrics CSVs)
```

**W&B** (project `kd-gat`):
- Online runs when network available, offline on SLURM compute nodes
- Sync offline runs: `wandb sync wandb/run-*`

**S3 Lakehouse** (AWS):
- Structured metrics as JSON at `s3://kd-gat/lakehouse/runs/`, queryable via DuckDB
- Fire-and-forget sync from `graphids/pipeline/lakehouse.py`

## Quarto Reports & Dashboard

**Dashboard:** `reports/dashboard.qmd` — single-file, multi-page Quarto dashboard using OJS + Mosaic/vgplot + DuckDB-WASM. Pages: Overview, Performance, Training, GAT & DQN, Knowledge Distillation, Graph Structure, Datasets, Staging. Data loaded from `reports/data/` (Parquet + JSON).

**Paper:** `reports/paper/` — 10-chapter research paper with interactive Mosaic figures embedded. Shared init in `_setup.qmd`, included via `_metadata.yml`. Chapters: Introduction, Background, Related Work, Methodology, Experiments, Results, Ablation, Explainability, Conclusion, Appendix. Figures include: force-directed CAN graphs, training curves, KD transfer scatter, attention heatmaps, UMAP embeddings, CKA similarity, DQN policy distribution, reconstruction error histograms, model size bar charts.

**Slides:** `reports/slides.qmd` — Revealjs presentation.

**Export pipeline**: `graphids/pipeline/export.py` → `export_all()` generates: leaderboard, per-run metrics, training curves, KD transfer, datasets, runs, metric catalog, graph samples, model sizes. Exports go directly to `reports/data/`.

**Deployment:** GitHub Actions renders Quarto on push to main and deploys via `actions/deploy-pages` (Actions-based Pages). CI: lint → test → quarto-build → deploy.

## Environment

- **Cluster**: Ohio Supercomputer Center (OSC), RHEL 9, SLURM scheduler
- **Home**: `/users/PAS2022/rf15/` — NFS v4, 1.7TB — permanent, safe for checkpoints
- **Scratch**: `/fs/scratch/PAS1266/` — GPFS (IBM Spectrum Scale), 90-day purge
- **Git remote**: `git@github.com:RobertFrenken/DQN-Fusion.git` (SSH)
- **Python**: uv venv `.venv/` (`source ~/KD-GAT/.venv/bin/activate`)
- **Key packages**: PyTorch 2.8.0+cu128, PyG 2.7.0, Lightning, Pydantic v2, W&B, Ray
- **Package manager**: uv (lockfile: `uv.lock`, config: `pyproject.toml [tool.uv]`)
- **SLURM account**: PAS3209, gpu partition, V100 GPUs
