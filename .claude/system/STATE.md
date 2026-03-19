# Current State

**Date**: 2026-03-16
**Branch**: `main`

## Ecosystem Status

### Fully Operational (Green)

| Component | Details |
|-----------|---------|
| **Config system** | Pydantic v2 frozen models + YAML composition. `resolve()` → frozen `PipelineConfig`. 6 datasets. Pydantic defaults are baseline; YAMLs contain only overrides. |
| **Training pipeline** | CLI: `python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`. DeviceStatsMonitor for GPU logging. |
| **Dagster orchestration** | Per-stage SLURM jobs via `dagster_defs.py` + `pipes_slurm.py`. Fire-and-forget mode with `--dependency=afterok` chains. Adaptive retry (OOM→2× mem, TIMEOUT→ckpt resume). 62 tests. |
| **SLURM integration** | Pitzer cluster. GPU (2x V100 per node, PAS1266) + CPU partitions. Resource profiles in `resources.yaml`. |
| **Graph caching** | Preprocessing v3.0.0 (26-D node features). DynamicBatchSampler. Content-addressable hash validation. |
| **DVC tracking** | 6 datasets tracked. S3 + local scratch remotes. |
| **MLflow tracking** | SQLite backend at `data/mlflow/mlflow.db`. Lightning autolog. 181 runs + 37 sweep trials. |
| **HF Dashboard** | Streamlit on HF Spaces. Data pushed via `push_experiments_to_hf.py`. |
| **Test suite** | 11 modules, 125+ tests passing locally. Needs SLURM for GPU tests. |
| **Package gateways** | Enforced API gateways for `graphids/`, `core/`, `pipeline/`, `orchestration/`. `TestGatewayEnforcement` blocks new deep imports. |
| **Data contracts** | Pydantic artifact models (`TrainingArtifact`, `EvaluationArtifact`, `PreprocessingArtifact`) validate stage outputs. |
| **Programmatic API** | `graphids/api.py`: `train()`, `evaluate()`, `orchestrate()` for notebooks/Dagster. |

### Partially Working (Yellow)

| Component | Issue |
|-----------|-------|
| **Inference server** | `serve.py` exists (`/predict`, `/health`). Prototype only, untested with current checkpoints. |
| **Sweep pipeline** | `optuna_sweep.py` — Optuna direct (replaced Ray Tune). SQLite-backed resume, subprocess trial isolation. |

### Not Implemented (Red)

| Component | Status |
|-----------|--------|
| **Statistical significance** | No automated bootstrap CI or paired t-test across seeds. |
| **CI/CD** | No GitHub Actions. Tests run manually via SLURM. |
| **Monitoring** | No drift detection or production monitoring. |
| **Reproducibility metadata** | No git SHA or lockfile in MLflow/artifacts. |
| **SU cost tracking** | Cannot answer "how much did this sweep cost?" |

## Codebase Metrics (Mar 16 2026)

| Subpackage | Files | Role |
|---|---|---|
| config/ | 7 | Schema, resolution, paths, catalog, constants, contracts + YAML files |
| core/ | 1 | data.py (dataset loading, NFS-safe caching) |
| core/models/ | 9 | VGAE, GAT, DQN, temporal, fusion, registry, protocols |
| core/preprocessing/ | 10 | Graph construction, adapters, vocabulary |
| pipeline/ | 5 | CLI, serve, validate, subprocess_utils, api.py |
| pipeline/stages/ | 10 | Training, evaluation, fusion, temporal |
| pipeline/orchestration/ | 6 | Dagster defs, pipes_slurm, sweep, tune, job |
| **Total** | **54** | **10,483 lines** |

GitNexus: 4,112 nodes, 6,235 edges, 168 execution flows.
Import hierarchy verified clean. Gateway enforcement test passing.

## Data Flow

```
Raw CAN CSVs (6 datasets, DVC)
  → Graph Cache (processed_graphs.pt, content-hash validated)
    → Training Pipeline (VGAE → GAT → DQN, large + small + small-KD)
      → Evaluation (metrics + embeddings + attention + policy)
        → Artifact validation (Pydantic contracts)
          → MLflow (sqlite) | experimentruns/ (on disk)
            → push_experiments_to_hf.py → HF Dataset → Streamlit Dashboard
```
