# Observability & Profiling

> Updated: 2026-04-09 | Environment: OSC Pitzer, V100 (16 GB), CUDA 12.6, PyTorch 2.8, PyG 2.7

## Architecture

OpenTelemetry is the single observability layer. Three signal types share one `Resource` and use the same exporter pattern.

**Phase A** (process startup — `graphids/_otel.py:init_providers`, called from the Typer `@app.callback()` in `graphids/cli/app.py`):
- `TracerProvider` + optional Wandb Weave OTLP exporter (gated on `WANDB_API_KEY`)
- `MeterProvider` placeholder (replaced by Phase B once `run_dir` is known)
- `LoggerProvider` -> `ConsoleLogRecordExporter(out=stderr)` + `LoggingHandler` bridges stdlib logging -> OTel
- `SlurmResourceDetector` merges SLURM env vars (`SLURM_JOB_ID`, partition, nodelist, etc.) into the shared `Resource`

**Phase B** (after `run_dir` is known — `graphids/_otel.py:wire_file_exporters`, called from `cli/_training.py::_prepare` and `orchestrate/stage.py::train`):
- `SimpleSpanProcessor` -> `ConsoleSpanExporter(out=run_dir/traces.jsonl)`
- `PeriodicExportingMetricReader` (10s) -> `ConsoleMetricExporter(out=run_dir/metrics.jsonl)`

## Wired tooling

| Layer | Tool | Where |
|-------|------|-------|
| Training metrics | `OTelTrainingLogger` (custom logger protocol) | `configs/_lib/defaults.libsonnet` trainer.logger |
| Span lifecycle + VRAM + GPU stats | `OTelTrainingCallback` (custom callback protocol) | `configs/_lib/defaults.libsonnet` callbacks.otel |
| Structured logging | `_StructuredAdapter` -> `LoggingHandler` | `graphids/_otel.py` |
| Traces (per-run) | `traces.jsonl` via `ConsoleSpanExporter` | `{run_dir}/traces.jsonl` |
| Metrics (per-run) | `metrics.jsonl` via `ConsoleMetricExporter` | `{run_dir}/metrics.jsonl` |
| Wandb Weave (optional) | OTLP HTTP exporter to `trace.wandb.ai` | `graphids/_otel.py`, gated on `WANDB_API_KEY` |
| Op-level profiling | PyTorchProfiler (chrome traces) | `scripts/slurm/submit.sh profile` |
| DuckDB catalog | (removed 2026-04-10 pending redesign) | — |
| SLURM job accounting | sacct summary + log rotation | `_epilog.sh` |
| CUDA alloc config | `expandable_segments:True,garbage_collection_threshold:0.8` | `_preamble.sh` |
| Mixed precision | `precision: 16-mixed` | `configs/_lib/defaults.libsonnet` |
| Gradient checkpointing | `use_reentrant=False` | `_conv.py` |

## OTelTrainingCallback (`graphids/core/monitoring.py`)

Installed via `defaults.libsonnet callbacks.otel`. Lifecycle:

- `on_fit_start`: opens `training.fit` span; sets `ml.run_dir`, `ml.model_class`, `ml.max_epochs`, identity attrs (stage, dataset, scale, seed, model_type); initializes NVML; discovers upstream `traces.jsonl` files via `vgae_ckpt_path`/`gat_ckpt_path` on the datamodule and records them as OTel span links for cross-stage KD lineage
- `on_train_batch_start/end`: batch duration histogram, loss histogram, VRAM gauges (allocated + reserved MiB), NVML hardware gauges (GPU utilization %, temperature, power W)
- `on_train_epoch_end`: span event `epoch.end` with train_loss, val_loss, LR, early_stopping wait count + best score
- `on_fit_end`: final `callback_metrics` as span attributes, `ml.epochs_run`, `ml.checkpoint.best_path`, status OK
- `on_exception`: records exception, status ERROR

## OTelTrainingLogger (`graphids/core/monitoring.py`)

Installed via `defaults.libsonnet trainer.logger`. Implements the custom `LoggerBase` protocol used by `core.trainer.Trainer`:

- `log_metrics`: each unique metric name -> cached OTel histogram (instruments created on first use)
- `log_hyperparams`: flattened params -> span attributes via `hparam.*` prefix

## Storage layers

OTel data in `traces.jsonl`/`metrics.jsonl` is **Layer 1** of a
three-layer architecture. **Layer 2** is a proposed workflow SQLite
(`{lake_root}/workflow.db`) for orchestration state (retries, skips,
mid-flight rows). **Layer 3** is the DuckDB analytics catalog
(`{lake_root}/catalog/graphids.duckdb`) rebuilt on demand from Layer 1 —
the old `orchestrate/ops/catalog.py` builder was removed 2026-04-10
pending redesign. Full schemas, write models, query patterns, and
implementation plan in
[`observability-data-layers.md`](observability-data-layers.md).

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

**Adopt**: OpenTelemetry (traces + metrics + logs), DuckDB catalog from traces.jsonl, PyTorchProfiler, nsys (one-off), torch.cuda memory APIs, sacct profiler

**Skip** (with reasons):
- **nvprof**: deprecated. **ncu**: 10-100x slower, only after nsys finds bad kernel. **DCGM**: needs admin (error -37 conflicts with SLURM GPU accounting).
- **cuGraph/cugraph-pyg**: graph classification, not sampling. **kvikIO/GDS**: no OSC infra.
- **cudnn.benchmark**: CNN-only. **channels_last**: image tensors. **TF32**: Ampere+ only. **CUDA Graphs**: variable-size batches.
- **MLflow**: NFS locking. **Aim**: RocksDB NFS issues. **Neptune**: dead. **DVC**: duplicates staging. **pytorch_memlab**: abandoned.
- **torch.compile `reduce-overhead`**: increases memory. Use default mode only.
- **wandb (direct dep)**: removed — OTel + optional Weave OTLP. Wandb Weave receives traces when `WANDB_API_KEY` is set.

## V100 deprecation warning

cuDNN 9.11+ drops V100 (Volta, compute 7.0). PyTorch 2.8 ships cuDNN 9.10.2 (last Volta version). **Pin `torch<2.9` when it ships.** Sources: [PyTorch #162574](https://github.com/pytorch/pytorch/issues/162574), [cuDNN 9.11.0 notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html)

## Sources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [PyTorch CUDA memory docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch GPU memory blog](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
