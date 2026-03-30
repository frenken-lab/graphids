# Research: NVIDIA GPU Profiling Tools for PyTorch on HPC

> Last modified: 2026-03-30
> Environment: OSC Pitzer, V100 (cc 7.0), CUDA 12.x, PyTorch 2.8.0+cu128

## Tool Overview

| Tool | Best For | Programmatic? | On OSC? |
|------|----------|---------------|---------|
| `torch.cuda.max_memory_allocated()` | Peak memory → batch sizing | Yes | Yes |
| `torch.cuda.memory._record_memory_history()` | Memory leak debugging | Yes (pickle → pytorch.org/memory_viz) |  Yes |
| `torch.profiler.profile` | Per-operator cost, CPU↔GPU gaps | Yes (JSON/CSV) | Yes |
| nsys (`module load nvhpc/25.1`) | System-wide CPU↔GPU bottlenecks | Yes (SQLite/CSV/Arrow) | Yes |
| ncu (`module load nvhpc/25.1`) | Per-kernel roofline (10-100x slower) | Yes (CSV) | Yes |

**Rejected:** nvprof (deprecated since CUDA 11), DCGM profiling module (error -37 conflicts with SLURM GPU accounting; `nvidia-smi dmon` is the fallback for SM util%).

## KD-GAT Priorities

1. **Batch sizing:** `_probe_bytes_per_node()` in `preprocessing/datamodule.py:36-77` (already wired). For deeper investigation: `_record_memory_history()` + pytorch.org/memory_viz.
2. **Training bottlenecks:** `torch.profiler` first (per-operator), then nsys if CPU↔GPU gap suspected.
3. **SLURM observability:** wandb system metrics (15s) + sacct + `_epilog.sh` (already wired).

## OSC Invocations

### nsys — System-Wide Timeline

```bash
module load nvhpc/25.1  # nsys 2024.7.1.84

nsys profile \
  --pytorch=autograd-shapes-nvtx \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -o /fs/scratch/PAS1266/profiles/my_run \
  python -m graphids fit --config graphids/config/stages/autoencoder.yaml

# Results (no GUI needed):
nsys stats my_run.nsys-rep                                          # full summary
nsys stats --report cuda_gpu_kern_sum --format csv my_run.nsys-rep  # kernel CSV
nsys export --type=sqlite my_run.nsys-rep                           # SQLite for queries
```

**Focused profiling** for long runs: `torch.cuda.cudart().cudaProfilerStart()` / `Stop()` in code, then `nsys profile --capture-range=cudaProfilerApi --stop-on-range-end=true ...`

### SLURM Job Script Pattern (nsys)

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load nvhpc/25.1
source /users/PAS2022/rf15/KD-GAT/scripts/slurm/_preamble.sh

nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx,osrt \
  -o "${TMPDIR}/profile_${SLURM_JOBID}" \
  python -m graphids fit ...

cp "${TMPDIR}/profile_${SLURM_JOBID}.nsys-rep" /fs/scratch/PAS1266/profiles/
nsys stats "${TMPDIR}/profile_${SLURM_JOBID}.nsys-rep"
```

### ncu — Kernel-Level (use after nsys finds slow kernels)

```bash
module load nvhpc/25.1
ncu --kernel-name "scatter_mean" --launch-count 5 \
  -o /fs/scratch/PAS1266/profiles/scatter_report \
  python -m graphids fit ...
```

**WARNING:** ncu replays each kernel multiple times (10-100x slowdown). Profile a few kernels only. KD-GAT priority: LOW — bottleneck is likely CPU-side (data loading, batch construction).

## PyG-Specific Notes

- Variable-size graph batches cause kernel dimension variance per step. Profile multiple steps and use `--pytorch=autograd-shapes-nvtx` to see how batch size variation affects scatter/gather kernel selection.
- `allocated` = memory holding tensors. `reserved` = allocator footprint (includes reusable free blocks). For batch sizing, `max_memory_allocated` is correct.
- `memory_stats()["num_alloc_retries"]` > 0 means near-OOM (leading indicator). Logged by DeviceStatsMonitor.

## Sources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [PyTorch CUDA memory docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | [PyTorch profiler docs](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch GPU memory visualization blog](https://pytorch.org/blog/understanding-gpu-memory-1/)
- [Kempner GPU profiling handbook](https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/scalability/gpu_profiling.html)
- OSC cluster: `module spider nvhpc`, torch 2.8.0 API checks (tested 2026-03-30)
