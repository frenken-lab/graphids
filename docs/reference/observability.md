# Observability & Profiling

> Updated: 2026-04-02 | Environment: OSC Pitzer, V100 (16 GB), CUDA 12.6, PyTorch 2.8, PyG 2.7
> Consolidates: observability.md, gpu-profiling-tools.md

## Wired tooling

| Layer | Tool | Where |
|-------|------|-------|
| Training metrics | WandbLogger + CSVLogger | `trainer.yaml` loggers |
| Full config logging | WandbSaveConfigCallback | `cli.py` |
| CSVLogger save_dir | `link_arguments` → `default_root_dir` | `cli.py` |
| GPU memory telemetry | DeviceStatsMonitor | `trainer.yaml` callbacks |
| GPU system metrics | wandb pynvml (util%, temp, power) | Automatic, 15s interval |
| Op-level profiling | PyTorchProfiler (chrome traces) | `scripts/submit.sh profile` |
| SLURM resource profiler | sacct: RSS, CPU%, wall time | `python -m graphids profile` |
| VRAM probe | `_probe_bytes_per_node()`, KD-aware | `datamodule.py` (prefers `_step()`, falls back to `forward()`) |
| Orchestration UI | dagster webserver + daemon | `scripts/dev/dagster-ui.sh` |
| Checkpoint handoff | CheckpointPathIOManager | JSON sidecars at `{lake_root}/.dagster/io/` |
| SLURM job accounting | sacct summary + log rotation | `_epilog.sh` |
| CUDA alloc config | `expandable_segments:True,garbage_collection_threshold:0.8` | `_preamble.sh` |
| Mixed precision | `precision: 16-mixed` | `trainer.yaml` |
| Gradient checkpointing | `use_reentrant=False` | `_conv.py` |

## GPU profiling tools

| Tool | Best for | Programmatic? | On OSC? |
|------|----------|---------------|---------|
| `torch.cuda.max_memory_allocated()` | Peak memory → batch sizing | Yes | Yes |
| `torch.cuda.memory._record_memory_history()` | Memory leak debugging | Yes (pickle → pytorch.org/memory_viz) | Yes |
| `torch.profiler.profile` | Per-operator cost, CPU↔GPU gaps | Yes (JSON/CSV) | Yes |
| nsys (`module load nvhpc/25.1`) | System-wide CPU↔GPU bottlenecks | Yes (SQLite/CSV/Arrow) | Yes |
| ncu (`module load nvhpc/25.1`) | Per-kernel roofline (10-100x slower) | Yes (CSV) | Yes |

### nsys invocation (OSC)

```bash
module load nvhpc/25.1  # nsys 2024.7.1.84
nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx,osrt,cudnn,cublas \
  -o /fs/scratch/PAS1266/profiles/my_run \
  python -m graphids fit --config graphids/config/stages/autoencoder.yaml

# Results (no GUI needed):
nsys stats my_run.nsys-rep                                          # summary
nsys stats --report cuda_gpu_kern_sum --format csv my_run.nsys-rep  # kernel CSV
nsys export --type=sqlite my_run.nsys-rep                           # for queries
```

Focused profiling for long runs: `torch.cuda.cudart().cudaProfilerStart()` / `Stop()` in code, then `nsys profile --capture-range=cudaProfilerApi ...`

### ncu invocation (use after nsys finds slow kernels)

```bash
ncu --kernel-name "scatter_mean" --launch-count 5 \
  -o /fs/scratch/PAS1266/profiles/scatter_report \
  python -m graphids fit ...
```

**WARNING:** ncu replays each kernel 10-100x. Profile a few kernels only. KD-GAT bottleneck is CPU-side (data loading), so ncu priority is LOW.

### PyG-specific notes

- Variable-size graph batches cause kernel dimension variance per step. Use `--pytorch=autograd-shapes-nvtx` to see batch size effects.
- `max_memory_allocated` (tensors used) is the correct metric for batch sizing, not `max_memory_reserved` (allocator blocks).
- `memory_stats()["num_alloc_retries"]` > 0 means near-OOM (logged by DeviceStatsMonitor).

## Remaining work

| Pri | Action | Effort |
|-----|--------|--------|
| P1 | Run 1 nsys profiling job | 1 SLURM job |
| P1 | Run 1 memory snapshot job | 1 SLURM job |
| P2 | ThroughputMonitor callback | ~20 LOC |
| P2 | Benchmark `torch.compile` | 1 SLURM job |
| P3 | Feed sacct output into DuckDB | Script |

## Gaps

- DuckDB catalog (`kd_gat.duckdb`) — no code writes to it; wandb partially replaces.
- sacct → DuckDB ingest — `python -m graphids profile --json` produces structured data, no auto-ingest.

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| wandb network failure | Medium | CSVLogger backup for `self.log()` metrics. GPU system metrics wandb-only. |
| Unsupervised models missing accuracy | Low | Expected — VGAE/DGI have no accuracy. Filter by model type. |
| RL fusion dynamic keys | Low | wandb handles sparse columns. |

## Tool decisions (don't re-investigate)

**Adopt**: wandb, DeviceStatsMonitor, CSVLogger, dagster UI, PyTorchProfiler, nsys (one-off), torch.cuda memory APIs, sacct profiler

**Skip** (with reasons — don't revisit):
- **nvprof**: deprecated. **ncu**: 10-100x slower, only after nsys finds bad kernel. **DCGM**: needs admin (error -37 conflicts with SLURM GPU accounting).
- **cuGraph/cugraph-pyg**: graph classification, not sampling. **kvikIO/GDS**: no OSC infra.
- **cudnn.benchmark**: CNN-only. **channels_last**: image tensors. **TF32**: Ampere+ only. **CUDA Graphs**: variable-size batches.
- **MLflow**: NFS locking. **Aim**: RocksDB NFS issues. **Neptune**: dead. **DVC**: duplicates staging. **pytorch_memlab**: abandoned.
- **torch.compile `reduce-overhead`**: increases memory. Use default mode only.

## V100 deprecation warning

cuDNN 9.11+ drops V100 (Volta, compute 7.0). PyTorch 2.8 ships cuDNN 9.10.2 (last Volta version). **Pin `torch<2.9` when it ships.** Sources: [PyTorch #162574](https://github.com/pytorch/pytorch/issues/162574), [cuDNN 9.11.0 notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html)

## Sources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [PyTorch CUDA memory docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch GPU memory blog](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [Kempner GPU profiling handbook](https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/scalability/gpu_profiling.html)
