# KD-GAT Architecture Decisions

> Import hierarchy: See code-style.md (enforced by tests/test_layer_boundaries.py).

## Config Architecture

- Pydantic v2 frozen BaseModels + YAML composition + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access, never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: domain/infrastructure constants live in `config/constants.py` (not in PipelineConfig). Hyperparameters live in PipelineConfig.
- Resolver: Pydantic v2's `model_validate()` handles schema validation — no custom key-checking needed.

> Experiment tracking (MLflow): See experiment-tracking.md.

## Orchestration

- Ray (`graphids/pipeline/orchestration/`) with `@ray.remote` tasks. `train_pipeline()` fans out per-dataset work concurrently.
- `--local` flag uses Ray local mode. HPO via Ray Tune with OptunaSearch + ASHAScheduler.
- **Subprocess dispatch**: Each stage runs as `subprocess.run()` for clean CUDA context. Overhead (~3-5s) is negligible vs training time (minutes-hours). CUDA contexts (~300-500 MB) are not reclaimable without process restart.
- **Per-stage granularity**: Finer (per-epoch) has massive scheduling overhead; coarser (per-variant) loses ability to re-run single stages.
- **Checkpoint passing**: Filesystem paths, not Ray object store (subprocesses can't access object store; checkpoints are small; path-based is debuggable).
- **Concurrent variants**: `small_nokd_pipeline` launches concurrently with `large_pipeline` (no teacher checkpoint dependency). On single-GPU, Ray serializes GPU tasks automatically; on multi-GPU, enables true parallelism.
- **No Ray Data**: Datasets fit in memory and PyG's heterogeneous graph `Data` objects are incompatible with Ray Data's Arrow-based tabular format.
- Archive restore: `graphids/pipeline/cli.py` archives previous runs before re-running, restores on failure.
- **Benchmark mode**: Set `KD_GAT_BENCHMARK=1` to log per-stage spawn overhead, execution time, inter-stage gaps, and GPU utilization to JSONL. See `scripts/profiling/benchmark_orchestration.sh`.

### Orchestration Design Rationale
Subprocess-per-stage kept for CUDA context isolation (~300-500 MB per model), fault isolation, and stage-level restartability. Overhead (~3-5s) is <0.1% of pipeline wall time.
Full analysis: `~/plans/orchestration-redesign-decision.md`

## Memory & Batch Sizing

- **DeviceStatsMonitor** (Lightning callback) handles GPU memory logging — no custom memory tracking code.
- **DynamicBatchSampler** (PyG) packs variable-size graphs to a node budget instead of fixed count.
- **Batch sizing**: `safety_factor × configured batch_size` (config-driven). No GPU memory probing.
- **Teacher offloading**: `cfg.training.offload_teacher_to_cpu` moves teacher to CPU between forward passes to save GPU memory. Shared helpers in `modules.py`.

## Inference Serving

`graphids/pipeline/serve.py` — FastAPI endpoints (`/predict`, `/health`) loading VGAE+GAT+DQN from `experimentruns/`. Returns fusion scores via DQN agent's `select_action()` + `_derive_scores()`.

## Dashboard

Consolidated Streamlit app at `~/kd-gat-dashboard/` reads from HF Datasets (auto-pushed by SLURM epilog via `scripts/data/push_experiments_to_hf.py`). Two data sources: `buckeyeguy/kd-gat-experiments` (MLflow push) + `buckeyeguy/kd-gat-sweeps` (sweep results).

Heavy analysis (UMAP, attention, CKA, etc.) lives in `notebooks/analysis/`.

## General Principles

- Delete unused code completely. No compatibility shims or `# removed` comments.
- Dataset catalog: `graphids/config/datasets.yaml` — single place to register new datasets.
- Leverage library features over custom code: Lightning callbacks, Pydantic validation, PyG batching, sklearn metrics.
