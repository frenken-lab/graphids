# Research: NVIDIA GPU Profiling Tools for PyTorch on HPC
> Last modified: 2026-03-30

**Status:** Complete
**Environment:** OSC Pitzer, SLURM, V100 (cc 7.0), CUDA 12.x, PyTorch 2.8.0+cu128

## Tool Overview

| Tool | Granularity | Best For | Programmatic? | On OSC? |
|------|-------------|----------|---------------|---------|
| `torch.cuda.memory_stats()` | Per-step | Peak memory → batch sizing | Yes (dict) | Yes (torch 2.8) |
| `torch.cuda.memory._record_memory_history()` | Per-allocation | Memory leak debugging | Yes (pickle) | Yes (torch 2.8) |
| NVTX + `torch.profiler` | Per-operator | CPU↔GPU gaps, operator cost | Yes (JSON/CSV) | Yes |
| Nsight Systems (nsys) | System-wide | CPU↔GPU bottlenecks, kernel gaps | Yes (SQLite/CSV) | Yes (`module load nvhpc/25.1`) |
| Nsight Compute (ncu) | Per-kernel | Kernel efficiency, roofline | Yes (CSV/ncu-rep) | Yes (`module load nvhpc/25.1`) |

### Rejected / Not Practical

| Tool | Verdict | Reason |
|------|---------|--------|
| nvprof | **Rejected** | Deprecated since CUDA 11.0, removed in CUDA 13.0. Still works on V100 (`/apps/.../cuda/.../12.6.2-.../bin/nvprof`) but nsys+ncu supersede it entirely. |
| DCGM | **Limited** | `nv-hostengine` runs on compute nodes (systemd, since 2026-01-26). `dcgmi dmon` works for temp/power but **profiling module fails** (error -37: conflict with SLURM GPU accounting). GPU util/SM/tensor fields unavailable. `sacct` has no GPU TRES fields. `nvidia-smi dmon` is the practical fallback for SM util %. |

## KD-GAT Use Case Recommendations

1. **Peak GPU memory → batch sizes:** `torch.cuda.max_memory_allocated()` + `reset_peak_memory_stats()`. Already in `_probe_bytes_per_node()`. For deeper investigation: `_record_memory_history()` + pytorch.org/memory_viz.
2. **Training bottlenecks:** `torch.profiler.profile` first (per-operator), then `nsys profile --pytorch=autograd-shapes-nvtx` if CPU↔GPU gap suspected.
3. **SLURM observability:** wandb system metrics (15s polling) + `sacct` + `_epilog.sh` GPU utilization report (already in use).

---

## Nsight Systems (nsys) — System-Wide Timeline

Captures all CPU/GPU activity: CUDA API calls, kernel launches, memory copies, NVTX ranges, cuDNN/cuBLAS ops, OS runtime, Python sampling. Modern replacement for nvprof.

### OSC Invocation

```bash
module load nvhpc/25.1  # nsys 2024.7.1.84

# Basic profiling (no code changes needed)
nsys profile \
  --pytorch=autograd-shapes-nvtx \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -o /fs/scratch/PAS1266/profiles/my_run \
  python -m graphids fit --config graphids/config/stages/autoencoder.yaml

# With Python stack sampling (CPU bottleneck identification)
nsys profile \
  --python-sampling=true --python-sampling-frequency=1000 \
  -t cuda,nvtx,osrt \
  -o /fs/scratch/PAS1266/profiles/my_run \
  python -m graphids fit ...
```

The `--pytorch` flag (nsys >= 2024.x) auto-annotates autograd ops with NVTX ranges + tensor shapes. Options: `autograd-nvtx`, `autograd-shapes-nvtx`, `functions-trace`.

### Focused Profiling (Long Runs)

Profile only specific steps via `cudaProfilerApi`:

```python
# In training code:
if step == 5: torch.cuda.cudart().cudaProfilerStart()
if step == 8: torch.cuda.cudart().cudaProfilerStop()
```

```bash
nsys profile --capture-range=cudaProfilerApi --stop-on-range-end=true ...
```

### Programmatic Result Access

**SQLite export (richest):**
```bash
nsys export --type=sqlite my_run.nsys-rep
# Key tables: CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_RUNTIME,
#   NVTX_EVENTS, CUDA_CALLCHAINS
```

**Built-in stats (CSV/text, no GUI needed):**
```bash
nsys stats my_run.nsys-rep                                          # full summary
nsys stats --report cuda_gpu_kern_sum --format csv my_run.nsys-rep  # kernel summary as CSV
# Reports: cuda_gpu_kern_sum, cuda_gpu_trace, cuda_api_sum, nvtx_sum, osrt_sum, etc.
```

**Polars-friendly exports:** `nsys export --type=arrow` or `--type=parquetdir`.

### SLURM Job Script Pattern

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load nvhpc/25.1
source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh

nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --stop-on-range-end=true \
  -o "${TMPDIR}/profile_${SLURM_JOBID}" \
  python -m graphids fit ...

cp "${TMPDIR}/profile_${SLURM_JOBID}.nsys-rep" /fs/scratch/PAS1266/profiles/
nsys stats "${TMPDIR}/profile_${SLURM_JOBID}.nsys-rep"
```

### PyG Note

Variable-size graph batches cause kernel dimension variance per step. Profile multiple steps and use `--pytorch=autograd-shapes-nvtx` to see how batch size variation affects scatter/gather kernel selection.

---

## Nsight Compute (ncu) — Kernel-Level Analysis

Per-kernel roofline analysis: compute-bound vs memory-bound, SM occupancy, L1/L2 cache hits, instruction mix, register spilling, warp efficiency.

**Typical workflow:** nsys first (find slow/frequent kernels) → ncu on those specific kernels.

### When ncu vs nsys

| Question | nsys | ncu |
|----------|------|-----|
| GPU idle between steps? | Yes | No |
| Why is this kernel slow? | No | Yes |
| Data loading bottleneck? | Yes | No |
| Is attention kernel memory-bound? | No | Yes |

### OSC Invocation

```bash
module load nvhpc/25.1  # ncu 2024.3.2.0

# Profile specific kernel (fast — avoid profiling ALL kernels)
ncu --kernel-name "scatter_mean" --launch-count 5 \
  -o /fs/scratch/PAS1266/profiles/scatter_report \
  python -m graphids fit ...

# Profile within NVTX range
ncu --nvtx --nvtx-include "training_step" -o report python -m graphids fit ...

# Export to CSV
ncu --import report.ncu-rep --csv > kernel_metrics.csv
```

**WARNING:** ncu replays each kernel multiple times (10-100x slowdown). Never run on a full training job — profile a few kernels or steps only.

**KD-GAT priority: LOW.** Bottleneck is likely CPU-side (data loading, batch construction) rather than kernel efficiency, given V100 hardware and model sizes.

---

## torch.cuda Memory APIs

Most directly useful for the batch-sizing problem.

### Peak Memory Tracking (Already in Use)

```python
torch.cuda.reset_peak_memory_stats(device)
# ... forward + backward + optimizer step ...
peak = torch.cuda.max_memory_allocated(device)
```

| Function | Use Case |
|----------|----------|
| `memory_allocated()` | Snapshot of live tensor memory |
| `max_memory_allocated()` | Per-step peak for batch sizing |
| `memory_reserved()` | Total allocator footprint (includes free blocks) |
| `max_memory_reserved()` | True GPU memory consumption |
| `reset_peak_memory_stats()` | Reset peak counters before measuring a region |

**allocated vs reserved:** `allocated` = memory holding tensors. `reserved` = memory held by caching allocator (includes reusable free blocks). For batch sizing, `max_memory_allocated` is correct.

### memory_stats() — Detailed Allocator Statistics

Returns ~80-key dict, structured as `{category}.{pool}.{metric}`. Key entries:

```python
stats = torch.cuda.memory_stats(device)
stats['allocated_bytes.all.peak']  # peak tensor memory
stats['reserved_bytes.all.peak']   # peak allocator footprint
stats['num_alloc_retries']         # non-zero = near OOM (leading indicator)
stats['num_ooms']                  # actual OOM count
```

`num_alloc_retries` counts forced cache flushes — non-zero means you're close to OOM. This is what Lightning's `DeviceStatsMonitor` logs (calls `memory_stats()` and sends all keys to logger).

### _record_memory_history() — Allocation Tracing

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run a few training steps ...
torch.cuda.memory._dump_snapshot("/fs/scratch/PAS1266/profiles/mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

Visualize by dragging pickle into https://pytorch.org/memory_viz, or load programmatically (dict with `segments` + `device_traces` keys with stack traces).

**WARNING:** Expensive — each allocation logs a Python stack trace (~8+ MB/step). Limit to a few steps.

**PyG relevance:** Variable-size batches cause irregular allocation patterns. Memory viz reveals whether peak memory comes from forward (activations), backward (gradients), or optimizer state — validates whether `_probe_bytes_per_node()`'s `_GRAD_MULTIPLIER = 2` is accurate.

### memory_snapshot() — Fragmentation Diagnosis

`torch.cuda.memory_snapshot()` returns segment dicts with block-level detail. Useful when `reserved >> allocated` (fragmentation: allocator holds large blocks with unfillable gaps).

### Practical Recipe: Per-Step Peak Memory Callback

```python
class PeakMemoryLogger(pl.Callback):
    """Log per-step peak GPU memory for batch size optimization."""
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        peak_mb = torch.cuda.max_memory_allocated(pl_module.device) / 1e6
        pl_module.log("memory/peak_allocated_mb", peak_mb)
        if hasattr(batch, "num_nodes"):
            pl_module.log("memory/peak_mb_per_node", peak_mb / batch.num_nodes)
```

Gives direct signal for the VRAM budget system — if `peak_mb_per_node` varies wildly, the probe's single-sample estimate may be unreliable.

---

## NVTX — Custom Annotations

Lightweight annotation API (~0.5-2 us overhead per push/pop). Ranges appear in nsys timelines.

### API

```python
import torch.cuda.nvtx
torch.cuda.nvtx.range_push("forward_pass")  # push/pop (nestable)
output = model(batch)
torch.cuda.nvtx.range_pop()

with torch.cuda.nvtx.range("optimizer_step"):  # context manager
    optimizer.step()

torch.cuda.nvtx.mark("epoch_start")  # point event
```

### Automatic Annotations (No Code Changes)

1. **nsys `--pytorch` flag (recommended):** `nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx ...`
2. **`torch.autograd.profiler.emit_nvtx()`:** Links backward NVTX ranges to forward ops.

### PyG Annotations Worth Adding

```python
with torch.cuda.nvtx.range("message_passing"):
    out = self.conv(x, edge_index)  # scatter/gather heavy
with torch.cuda.nvtx.range("batch_construction"):
    batch = next(dataloader_iter)   # CPU-bound
```

Reveals whether training time is dominated by message passing (GPU) or batch construction (CPU).

---

## torch.profiler.profile — Built-In Profiler

Combines CPU profiling, CUDA kernel tracing, and memory tracking. Outputs Kineto traces for Chrome/Perfetto/TensorBoard.

### Invocation

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    record_shapes=True, profile_memory=True, with_stack=True, with_flops=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

Schedule: `wait=1` (skip step 0), `warmup=1` (discard step 1), `active=3` (record steps 2-4).

### Result Access

```python
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
prof.export_memory_timeline("memory_timeline.html")
prof.export_stacks("stacks.txt", "self_cuda_time_total")  # for flamegraph
```

Meta's [HTA library](https://github.com/facebookresearch/HolisticTraceAnalysis) provides programmatic analysis: `TraceAnalysis(trace_dir=...).get_gpu_kernel_breakdown()`.

### torch.profiler vs nsys

| Aspect | torch.profiler | nsys |
|--------|----------------|------|
| Setup | Python API, no external tool | CLI wrapper, `module load nvhpc/25.1` |
| Scope | PyTorch ops only | Entire system (data loading, I/O, threads) |
| Memory tracking | Per-operator allocation deltas | No (use torch APIs) |
| FLOPS | Estimated per-operator | No |
| Output size | Small (JSON) | Large (GBs possible) |
| GUI | Chrome/Perfetto/TensorBoard | Nsight Systems GUI |

**Use torch.profiler** for quick per-operator analysis. **Use nsys** when you need the full system picture (GPU starvation from slow DataLoader, etc.).

---

## OSC Module Quick Reference

```bash
module load nvhpc/25.1    # nsys 2024.7.1.84, ncu 2024.3.2.0
module load cuda/12.6.2    # nvprof (deprecated, V100 only)
dcgmi --version            # 3.2.6 (hostengine not running on login nodes)
# torch 2.8.0+cu128: memory_stats, memory_snapshot, _record_memory_history all available
```

---

## Sources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [nvprof deprecation announcement](https://forums.developer.nvidia.com/t/announcement-cuda-nvprof-and-visual-profiler-are-deprecated/358159)
- [DCGM + SLURM integration blog](https://developer.nvidia.com/blog/job-statistics-nvidia-data-center-gpu-manager-slurm/)
- [PyTorch CUDA memory docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch GPU memory visualization blog](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [PyTorch profiler docs](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch profiler TensorBoard tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [Kempner GPU profiling handbook](https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/scalability/gpu_profiling.html)
- [mcarilli nsys PyTorch commands gist](https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223)
- [nsys2json converter](https://github.com/chenyu-jiang/nsys2json)
- [Purdue RCAC nsys guide](https://www.rcac.purdue.edu/knowledge/profilers/nvidia_nsight_systems)
- [Nsight Systems SQLite export examples](https://archive.docs.nvidia.com/nsight-systems/2021.5/nsys-exporter/examples.html)
- [NERSC profiling tools guide](https://docs.nersc.gov/tools/performance/nvidiaproftools/)
- [Speed Up PyTorch 3x with Nsight](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [nsys vs ncu NVIDIA forum](https://forums.developer.nvidia.com/t/which-tool-can-accurately-obtain-kernel-performance-ncu-or-nsys/360150)
- [PyTorch NVTX source](https://github.com/pytorch/pytorch/blob/main/torch/cuda/nvtx.py)
- [nvtx Python docs](https://nvtx.readthedocs.io/en/latest/index.html)
- OSC cluster: `module spider nvhpc`, `dcgmi --version`, torch 2.8.0 API checks (tested 2026-03-30)
