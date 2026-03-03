# Current State

**Date**: 2026-03-01
**Branch**: `main`

## Ecosystem Status

### Fully Operational (Green)

| Component | Details |
|-----------|---------|
| **Config system** | Pydantic v2 frozen models + YAML composition. `resolve(model_type, scale, auxiliaries, **overrides)` → frozen `PipelineConfig`. 6 datasets in `graphids/config/datasets.yaml`. |
| **Training pipeline** | 72 legacy validation runs archived to `data/datalake_archive/`. CLI: `python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name>` |
| **Ray orchestration** | `train_pipeline()` and `eval_pipeline()` via Ray remote tasks + SLURM. `--local` flag for Ray local mode. Subprocess-per-stage dispatch (intentional — CUDA context isolation). `small_nokd` runs concurrently with `large`. Benchmark mode via `KD_GAT_BENCHMARK=1`. |
| **SLURM integration** | Pitzer cluster. GPU (2x V100 per node, 362GB RAM, PAS1266) + CPU partitions. Account set in `.env` (`KD_GAT_SLURM_ACCOUNT`). |
| **Graph caching** | All 6 datasets cached with test scenarios (`processed_graphs.pt` + `test_*.pt`). DynamicBatchSampler for variable-size graphs. |
| **DVC tracking** | Raw data + cache tracked. S3 remote + local scratch remote configured. |
| **Export pipeline** | 8 lightweight exporters (~2s, login node safe) → `reports/data/`. Heavy analysis in notebooks. |
| **Datalake** | Parquet-based structured storage in `data/datalake/` (runs, metrics, configs, artifacts, training curves). DuckDB analytics views. S3 backup via SLURM epilog. |
| **Quarto site** | Dashboard + paper + slides rendered via Quarto. Auto-deployed to GitHub Pages via GitHub Actions on push to main. |
| **Test suite** | 108 tests (88 passed, 20 skipped). All passing on CPU fallback. |
| **CI/CD** | GitHub Actions CI: lint → test → quarto-build → deploy (Actions-based Pages). All 4 jobs green. |

### Partially Working (Yellow)

| Component | Issue |
|-----------|-------|
| **W&B tracking** | 77 online runs; 3 offline runs moved to `data/datalake_archive/wandb/`. |
| **Paper figures** | Mosaic figures deployed with vgplot@0.21.1 CDN. Paper chapters use `{{< include _setup.qmd >}}` for OJS init. Pending runtime verification in browser. |
| **Inference server** | `pipeline/serve.py` exists (`/predict`, `/health`). Untested with current checkpoints. |

### Not Integrated

| Component | Status |
|-----------|--------|
| **RAPIDS GPU acceleration** | Removed. pip wheels conflict with PyTorch cu128 + PyG cu126. Single uv env is the answer. |

## Next Phase: Research Platform

The codebase is transitioning from validation (72 runs, binary classification) to a research platform with expanded scope:

- **Data-driven Ray DAG** — Config-driven stage chain replaces hardcoded 3-variant functions (Phase 2)
- **Attack-type metadata** — Graphs carry attack-type labels as metadata for future node-level classification (Phase 3)
- **Pluggable fusion** — DQN, MLP, and weighted average fusion methods selectable via config (Phase 4)
- **Resource tracking** — Every run self-documents wall-clock time, GPU peak memory, SLURM job ID (Phase 5)
- **Profiling jobs** — Orchestration benchmark + conv_type profiling to close cuGraph decision gate (Phase 6)

## Archived Legacy Runs

72 validation runs (6 datasets × 12 configs) archived to `data/datalake_archive/`:
- `runs.parquet`, `metrics.parquet`, `configs.parquet`, `artifacts.parquet`
- `training_curves/` (36 Parquet files)
- `artifacts/` (5 subdirs: attention_weights, cka_similarity, dqn_policy, embeddings, recon_errors)
- `wandb/` (3 offline runs)

Run directories remain in `experimentruns/` (2.7 GB). New runs will write to `data/datalake/` (fresh).

**Per dataset (12 runs each):**
- `vgae_{large,small,small_kd}_autoencoder` (3 VGAE)
- `gat_{large,small,small_kd}_curriculum` (3 GAT)
- `dqn_{large,small,small_kd}_fusion` (3 DQN)
- `eval_{large,small,small_kd}_evaluation` (3 Eval)

## Data Flow

```
Raw CAN CSVs (6 datasets, 10.8 GB, DVC)
  → Graph Cache (processed_graphs.pt + test_*.pt, DVC)
    → Training Pipeline (VGAE → GAT → DQN, large + small + small-KD)
      → Evaluation (metrics + embeddings + attention + policy)
        → W&B (77 online) | Datalake (Parquet) | experimentruns/ (on disk)
          → Export Pipeline (8 lightweight exporters → reports/data/)
            → Quarto Site (dashboard + paper + slides)
              → GitHub Pages (auto-deploy via GitHub Actions on push to main)
```

## OSC Environment

- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)
- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)
- **Ray temp**: `/fs/scratch/PAS1266/.ray/`
- **W&B**: Project `kd-gat` (offline on compute nodes, sync later)
- **Reports**: `reports/` (Quarto site — auto-deployed to GitHub Pages)
