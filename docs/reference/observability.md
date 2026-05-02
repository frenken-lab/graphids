# Observability & Profiling

> Updated: 2026-04-16 | Environment: OSC Pitzer, V100 (16 GB), CUDA 12.6, PyTorch 2.8, PyG 2.7

## Architecture

Two stores: **MLflow** for run-level metadata + scalar metrics timeseries + device telemetry, **OTel** for spans + structured-log events. They share a `Resource` populated by `SlurmResourceDetector` and an identity-derived `run_name` that links rows across both.

**Phase A** (process startup — `graphids/_otel.py:init_providers`, called from the Typer `@app.callback()` in `graphids/cli/app.py`):
- `TracerProvider` + optional Wandb Weave OTLP exporter (gated on `WANDB_API_KEY`)
- `LoggerProvider` -> `ConsoleLogRecordExporter(out=stderr)` + `LoggingHandler` bridges stdlib logging -> OTel
- `SlurmResourceDetector` merges SLURM env vars into the shared `Resource`

**Phase B** (after `run_dir` is known — `graphids/_otel.py:wire_file_exporters`, called from `cli/training.py::_prepare`):
- `BatchSpanProcessor` -> `ConsoleSpanExporter(out=run_dir/traces.jsonl)` — the `training.fit` span + structured-log events

**Phase C** (at fit-start — `graphids/_mlflow.py::start_training_run`, called from `orchestrate/stage.py::train`):
- Opens MLflow run in per-axis experiment `graphids/{dataset}/{group}` (SQLite backend at `{LAKE_ROOT}/mlflow.db` — shared, distinct from per-user `RUN_ROOT`). **MLflow is a hard dep** (since 2026-04-24); failures propagate. Only soft-failure paths are documented in `_mlflow.py` module docstring. **Idempotent**: search_runs by `run_name` + phase=fit; FAILED/KILLED → resume same `run_id`; TERMINATED → new (reaper owns); RUNNING/FINISHED → refuse unless `GRAPHIDS_FORCE_RESUME=1`; git-SHA change → new run (option b: don't mix commits in one row).
- Tags identity: `graphids.{phase, run_dir, dataset, group, variant, seed, model_type, scale}` + SLURM (`slurm.job_id`, `slurm.cluster_name`) when set. Upstream-teacher lineage is **not** a run-level tag — it lives on the `LoggedModel` entity (see fit-end below).
- Enables MLflow system-metrics sampler (background thread, 5s interval) — GPU util, VRAM, CPU, memory, disk, network.
- `MLflowTrainingCallback` forwards every key in `trainer.callback_metrics` (whatever the model layer logged via `self.log(...)`) to MLflow via one `log_batch` per epoch at `step=epoch`. At epoch 0 it stamps `params.graphids.{budget_target_bytes, num_workers, prefetch_factor}` + `tags.graphids.num_workers_source` (one-shot).
- At fit-end: `metrics.graphids.peak_vram_mb` (single point at `step=current_epoch`), `tags.graphids.budget_binding`, and `MlflowClient.create_logged_model` — metadata-only `LoggedModel` (no artifact bytes) with `name=run_name`, `model_type={fq_python_class}`, `source_run_id`, `tags.graphids.{ckpt_path, ckpt_sha256, dataset, group, variant, seed}`, `params.graphids.run_dir`. Idempotent on resume (query-first by `source_run_id`, update tags or create). Run closes FINISHED (or FAILED on exception).

## Wired tooling

| Layer | Tool | Where |
|-------|------|-------|
| Run metadata + scalar metrics | MLflow SQLite backend | `graphids/_mlflow.py` |
| Per-epoch metrics | `MLflowTrainingCallback` | `configs/_lib/defaults.libsonnet` callbacks.mlflow |
| Device telemetry (GPU/CPU/mem) | MLflow system-metrics sampler (psutil + nvidia-ml-py) | `_mlflow.start_training_run` |
| Structured logging | `_StructuredAdapter` -> `LoggingHandler` | `graphids/_otel.py` |
| Traces + log events (per-run) | `traces.jsonl` via `ConsoleSpanExporter` | `{run_dir}/traces.jsonl` |
| Wandb Weave (optional) | OTLP HTTP exporter to `trace.wandb.ai` | `graphids/_otel.py`, gated on `WANDB_API_KEY` |
| Op-level profiling | PyTorchProfiler (chrome traces) | `python -m graphids submit --mode gpu --length short --command "python -m graphids profile"` |
| SLURM job accounting | sacct summary + log rotation | `_epilog.sh` |
| CUDA alloc config | `expandable_segments:True,garbage_collection_threshold:0.8` | `_preamble.sh` |
| Mixed precision | `precision: 16-mixed` (default); supervised stage overrides to `32-true` | `configs/_lib/defaults.libsonnet`; `configs/stages/supervised.jsonnet` |
| Gradient checkpointing | `use_reentrant=False` | `_conv.py` |

## MLflowTrainingCallback (`graphids/core/mlflow_callback.py`)

Installed via `defaults.libsonnet callbacks.mlflow`. Run lifecycle is owned by `_mlflow.start_training_run` (called from `stage.train` before `trainer.fit`); this callback only writes into the active run.

- `on_train_epoch_end`: `mlflow.log_metrics({**trainer.callback_metrics, lr, early_stop.wait, early_stop.best_score}, step=current_epoch)` — passthrough, not a whitelist. The current model surface logs `train_loss`, `val_loss`; VGAE additionally logs per-component telemetry (`train_recon`, `train_canid`, `train_nbr`, `train_kl`) and per-class val splits (`val_loss_benign`, `val_loss_attack`); GAT/MLP/WAvg add `train_acc`/`val_acc`. Adding a `self.log(...)` call in any module flows to MLflow without callback changes.
- `on_fit_end`: `log_final_fit(peak_vram_mb, epochs_run, best_ckpt_path, run_dir)` + `end_training_run("FINISHED")`
- `on_exception`: `end_training_run("FAILED")`

Device telemetry is captured by MLflow's background system-metrics thread while the run is active — no per-batch NVML hooks needed. Span lifecycle for `training.fit` is a single span created implicitly via `trainer.fit` wrapping; cross-stage KD lineage (VGAE→GAT→fusion) is recoverable via `graphids.ckpt_sha256` tags + upstream ckpt paths stored in downstream `resolved.json`.

## Storage layers

- **MLflow run store** (`{LAKE_ROOT}/mlflow.db` + `mlartifacts/`) — authoritative
  for run metadata, params, per-epoch scalar metrics, and device telemetry.
  One fit-phase row (opened at fit-start, closed at fit-end) + one test-phase
  row (post-hoc sink in `stage.evaluate`), both sharing `run_name =
  {group}_{variant}_{dataset}_seed{N}[_{cluster}]` and distinguished by the
  `graphids.phase` tag. Query via `mlflow.search_runs` or
  `client.get_metric_history(run_id, key)`.
- **Per-run traces** (`{run_dir}/traces.jsonl`) — OTel spans + structured-log
  events (`budget_probed`, `vram_drift_detected`, `early_stopping`, etc.).
  Parsed by `graphids/core/run_io.py::load_traces` (polars NDJSON). Useful
  for debugging single runs; not a query surface for cross-run analysis.

## GPU profiling tools

| Tool | Best for | On OSC? |
|------|----------|---------|
| `torch.cuda.max_memory_allocated()` | Peak memory -> batch sizing | Yes |
| `torch.cuda.memory._record_memory_history()` | Memory leak debugging (pickle -> pytorch.org/memory_viz) | Yes |
| `torch.profiler.profile` | Per-operator cost, CPU<->GPU gaps (JSON/CSV) | Yes |
| nsys (`module load nvhpc/25.1`) | System-wide CPU<->GPU bottlenecks | Yes |
| ncu (`module load nvhpc/25.1`) | Per-kernel roofline (10-100x slower) | Yes |

### nsys invocation (OSC)

```bash
module load nvhpc/25.1  # nsys 2024.7.1.84
nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx,osrt,cudnn,cublas \
  -o /fs/scratch/PAS1266/profiles/my_run \
  python -m graphids fit --config configs/stages/autoencoder.jsonnet

nsys stats my_run.nsys-rep                                          # summary
nsys stats --report cuda_gpu_kern_sum --format csv my_run.nsys-rep  # kernel CSV
nsys export --type=sqlite my_run.nsys-rep                           # for queries
```

Focused profiling: `torch.cuda.cudart().cudaProfilerStart()` / `Stop()` in code, then `nsys profile --capture-range=cudaProfilerApi ...`

### ncu invocation (use after nsys finds slow kernels)

```bash
ncu --kernel-name "scatter_mean" --launch-count 5 \
  -o /fs/scratch/PAS1266/profiles/scatter_report \
  python -m graphids fit ...
```

**WARNING:** ncu replays each kernel 10-100x. GraphIDS bottleneck is CPU-side (data loading), so ncu priority is LOW.

### PyG-specific notes

- Variable-size graph batches cause kernel dimension variance per step. Use `--pytorch=autograd-shapes-nvtx` to see batch size effects.
- `max_memory_allocated` (tensors used) is the correct metric for batch sizing, not `max_memory_reserved` (allocator blocks).

## Tool decisions (don't re-investigate)

**Adopt**: OpenTelemetry (traces + metrics + logs), MLflow run store (SQLite on GPFS, file artifacts), PyTorchProfiler, nsys (one-off), torch.cuda memory APIs, sacct profiler

**Skip** (with reasons):
- **nvprof**: deprecated. **ncu**: 10-100x slower, only after nsys finds bad kernel. **DCGM**: needs admin (error -37 conflicts with SLURM GPU accounting).
- **cuGraph/cugraph-pyg**: graph classification, not sampling. **kvikIO/GDS**: no OSC infra.
- **cudnn.benchmark**: CNN-only. **channels_last**: image tensors. **TF32**: Ampere+ only. **CUDA Graphs**: variable-size batches.
- **Aim**: RocksDB NFS issues. **Neptune**: dead. **DVC**: duplicates staging. **pytorch_memlab**: abandoned. **MLflow file-store backend**: deprecated Feb 2026; we use the SQLite backend.
- **torch.compile `reduce-overhead`**: increases memory. Use default mode only.
- **wandb (direct dep)**: removed — OTel + optional Weave OTLP. Wandb Weave receives traces when `WANDB_API_KEY` is set.

## V100 deprecation warning

cuDNN 9.11+ drops V100 (Volta, compute 7.0). PyTorch 2.8 ships cuDNN 9.10.2 (last Volta version). **Pin `torch<2.9` when it ships.** Sources: [PyTorch #162574](https://github.com/pytorch/pytorch/issues/162574), [cuDNN 9.11.0 notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html)

## Sources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [PyTorch CUDA memory docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch GPU memory blog](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
