# Current State

**Date**: 2026-03-11
**Branch**: `main`

## Ecosystem Status

### Fully Operational (Green)

| Component | Details |
|-----------|---------|
| **Config system** | Pydantic v2 frozen models + YAML composition. `resolve()` → frozen `PipelineConfig`. 6 datasets. |
| **Training pipeline** | CLI: `python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name>`. DeviceStatsMonitor for GPU logging. |
| **Ray orchestration** | `train_pipeline()` via Ray remote tasks + subprocess dispatch. Config-driven variants. Single SLURM allocation model. |
| **SLURM integration** | Pitzer cluster. GPU (2x V100 per node, PAS1266) + CPU partitions. |
| **Graph caching** | Preprocessing v2.0.0 (26-D node features). DynamicBatchSampler. Version-gated rebuild. |
| **DVC tracking** | 6 datasets tracked. S3 + local scratch remotes. |
| **MLflow tracking** | SQLite backend at `data/mlflow/mlflow.db`. Lightning autolog. 181 runs + 37 sweep trials. |
| **HF Dashboard** | Streamlit on HF Spaces. Data pushed via `push_experiments_to_hf.py`. |
| **Test suite** | 10 modules, 2,417 lines. All passing on CPU. Needs SLURM verification. |

### Partially Working (Yellow)

| Component | Issue |
|-----------|-------|
| **Inference server** | `serve.py` exists (`/predict`, `/health`). Prototype only, untested with current checkpoints. |
| **Sweep pipeline** | `sweep_pipeline.py` + `tune_config.py` functional but tightly coupled to Ray Tune. |
| **State management** | `store.py` + `job.py` exist but only used by sweep_pipeline. Ray pipeline has NO persistent state. |

### Not Implemented (Red)

| Component | Status |
|-----------|--------|
| **Per-stage SLURM jobs** | Entire pipeline runs in single allocation. Deleted coordinator/executor had this. |
| **Dry-run / plan mode** | Cannot preview pipeline before execution. Deleted planner.py had this. |
| **Statistical significance** | No automated bootstrap CI or paired t-test across seeds. |
| **CI/CD** | No GitHub Actions. Tests run manually via SLURM. |
| **Monitoring** | No drift detection or production monitoring. |

## Orchestration Status

The orchestration layer has gone through three iterations, all within Mar 9-10:

1. **coordinator.py** (862→915 lines) — stateful SLURM coordinator with per-stage sbatch, checkpoint-aware resume, adaptive resource scaling. Built and deleted.
2. **5-component system** (planner, executor, driver, store, job — 1,751 lines) — scheduler-agnostic with SLURM/Flux/DryRun backends. Built and deleted.
3. **Current: Ray wrapper** (ray_pipeline.py 360 lines + ray_slurm.py 65 lines) — subprocess dispatch within single SLURM allocation. Functionally a sequential shell script.

All deleted code is in git history. Key commits: `188f13c`, `3ba9111`, `123845f`, `971dd6e`, `c697b63`.

Decision pending: Path A (Keep Ray + add Submitit), Path B (Parsl), or Path C (custom coordinator + Submitit). See `~/plans/orchestration-tool-evaluation.md`.

## Codebase Metrics (Mar 11 2026)

| Subpackage | Files | Lines | Role |
|---|---|---|---|
| config/ | 6 | 903 | Schema, resolution, paths, catalog, constants |
| core/models/ | 8 | 1,652 | VGAE, GAT, DQN, temporal, fusion, registry |
| core/preprocessing/ | 10 | 1,746 | Graph construction, adapters, vocabulary |
| core/training/ | 2 | 454 | Lightning DataModule + dataset loading |
| pipeline/cli.py | 1 | 631 | Entry point, MLflow context, archive logic |
| pipeline/stages/ | 10 | 2,225 | Training, evaluation, fusion, temporal |
| pipeline/orchestration/ | 7 | 1,526 | Ray, sweep, tune, store, job, SLURM bridge |
| pipeline/ (other) | 3 | 396 | serve, validate, subprocess_utils |
| **Total** | **50** | **9,540** | |

Import hierarchy verified clean: pipeline→config: 40, pipeline→core: 20, core→config: 5. No violations.

## Critical Gaps (from ecosystem registry)

1. **State loss on SLURM timeout** — Ray pipeline has no persistent state
2. **Single-allocation bottleneck** — no per-stage SLURM job submission
3. **No reproducibility metadata** — no git SHA or lockfile in MLflow/artifacts
4. **No SU cost tracking** — cannot answer "how much did this sweep cost?"

## Data Flow

```
Raw CAN CSVs (6 datasets, DVC)
  → Graph Cache (processed_graphs.pt, DVC)
    → Training Pipeline (VGAE → GAT → DQN, large + small + small-KD)
      → Evaluation (metrics + embeddings + attention + policy)
        → MLflow (sqlite) | experimentruns/ (on disk)
          → push_experiments_to_hf.py → HF Dataset → Streamlit Dashboard
```
