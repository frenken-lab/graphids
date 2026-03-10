# Current State

**Date**: 2026-03-07
**Branch**: `simplify-codebase` (uncommitted changes — 29 files changed, net -3,381 lines)

## Ecosystem Status

### Fully Operational (Green)

| Component | Details |
|-----------|---------|
| **Config system** | Pydantic v2 frozen models + YAML composition. `resolve()` → frozen `PipelineConfig`. 6 datasets. Pydantic handles schema validation (no custom key-checking). |
| **Training pipeline** | CLI: `python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`. DeviceStatsMonitor for GPU logging. |
| **Ray orchestration** | `train_pipeline()` and `eval_pipeline()` via Ray remote tasks + SLURM. Config-driven variant pipeline. Benchmark mode via `KD_GAT_BENCHMARK=1`. |
| **SLURM integration** | Pitzer cluster. GPU (2x V100 per node, PAS1266) + CPU partitions. |
| **Graph caching** | Preprocessing v2.0.0 (26-D node features). DynamicBatchSampler for variable-size graphs. |
| **DVC tracking** | Raw data + cache tracked. S3 remote + local scratch remote configured. |
| **MLflow tracking** | SQLite backend at `data/mlflow/mlflow.db`. `cli.py` wraps dispatch in `mlflow.start_run()`. `trainer_factory.py` uses `mlflow.pytorch.autolog()`. |
| **HF Dashboard** | Consolidated Streamlit app at `~/kd-gat-dashboard/`. Data pushed via `scripts/data/push_experiments_to_hf.py`. |
| **Test suite** | All passing on CPU fallback. Needs SLURM verification after simplification work. |
| **CI/CD** | GitHub Actions CI: lint → test. |

### Partially Working (Yellow)

| Component | Issue |
|-----------|-------|
| **Inference server** | `pipeline/serve.py` exists (`/predict`, `/health`). Uses DQN `_derive_scores()`. Untested with current checkpoints. |

## Recent Simplification (Mar 2026)

Codebase reduced from 55 → 48 Python files under `graphids/`:

**Deleted files:**
- `graphids/pipeline/export.py` (1,357 lines) — replaced by MLflow + HF push
- `graphids/pipeline/lakehouse.py` (301 lines) — replaced by MLflow
- `graphids/pipeline/sweep_export.py` (141 lines) — replaced by MLflow
- `graphids/pipeline/tracking.py` (60 lines) — replaced by MLflow autolog
- `graphids/pipeline/errors.py` (25 lines) — custom exceptions unused
- `graphids/pipeline/memory.py` (471→0 lines) — DeviceStatsMonitor replaces custom GPU tracking
- `graphids/pipeline/stages/callbacks.py` (104 lines) — DeviceStatsMonitor replaces custom callbacks
- `graphids/core/explain.py` (170 lines) — rarely-used GNNExplainer gated behind disabled flag
- `graphids/core/preprocessing/adapters/network_flow.py` (291 lines) — dead code, never instantiated

**Simplified files:**
- `batch_sizing.py`: 169→43 lines (config-driven `safety_factor × batch_size` replaces GPU probing)
- `modules.py`: Extracted shared `_teacher_to_device()` / `_teacher_offload()` helpers
- `resolver.py`: Removed `_warn_unused_keys()` (Pydantic handles it)
- `dqn.py`: Fixed hardcoded weights bug in `_derive_scores()`, uses `self._vgae_weights`
- `serve.py`: Proper HTTP 503 instead of dead DQN fallback code
- `ray_pipeline.py`: Extracted `_init_ray()`, data-driven eval variant dispatch

## Data Flow

```
Raw CAN CSVs (6 datasets, DVC)
  → Graph Cache (processed_graphs.pt, DVC)
    → Training Pipeline (VGAE → GAT → DQN, large + small + small-KD)
      → Evaluation (metrics + embeddings + attention + policy)
        → MLflow (sqlite) | experimentruns/ (on disk)
          → push_experiments_to_hf.py → HF Dataset → Streamlit Dashboard
```
