# Pre-batch Timing Rationale — hcrl_sa (probe 2026-04-07)

Justifies the choice of prebatched training + workers=0 over per-step collation
with workers. Numbers are from a single probe run; see `data-flow.md` for current
architecture.

## Dataset and hardware

- hcrl_sa: 19,085 graphs, mean 38.3 nodes, mean 98 edges, ~10.8 KB/graph
- V100 16GB, PCIe 3.0 x16 (~12 GB/s), pin_memory ~20 GB/s

## Probe numbers (optimizer+compile warmup included)

| Model      | Budget (nodes) | Graphs/batch | Batch MB | T_collation (old) | T_gpu  | H2D (est) | Pin (est) |
|------------|---------------|-------------|----------|-------------------|--------|-----------|-----------|
| VGAE small | 404,718       | 10,557      | 113.9    | 380.7 ms          | 154.9 ms | 9.5 ms  | 5.7 ms  |
| VGAE large | 343,522       | 8,959       | 96.7     | 327.1 ms          | 286.8 ms | 8.1 ms  | 4.8 ms  |
| GAT small  | 233,339       | 6,086       | 65.7     | 218.5 ms          | 261.5 ms | 5.5 ms  | 3.3 ms  |
| GAT large  | 62,439        | 1,629       | 17.6     | 58.9 ms           | 121.6 ms | 1.5 ms  | 0.9 ms  |

T_collation = `Batch.from_data_list()` cost **on the old per-step path only**.
H2D = batch_MB / 12 GB/s.  Pin = batch_MB / 20 GB/s.

## Why prebatch wins

**Old path (per-step collation, 3 workers):** `T_step = max(T_collation/workers, T_gpu) + T_H2D`.
For VGAE small: max(381/3, 155) + 10 = max(127, 155) + 10 = 165 ms/step, GPU util ~60-83%.
Workers needed because T_collation (381 ms) >> T_gpu (155 ms).

**New path (prebatch at setup, workers=0):** `_prebatched_train_dataloader` calls
`NodeBudgetBatchSampler` once at setup to plan batches, then `Batch.from_data_list()`
once per batch (not per step, not per epoch). The training loop iterates over a
`list[Batch]` with `num_workers=0` wrapped by `PrefetchLoader` for async H2D.

Per-step CPU work: pin(5.7 ms) + H2D(9.5 ms) = 15.2 ms. GPU step = 154.6 ms.
CPU finishes preparing next batch in 15.2 ms, waits 139 ms. GPU never idles.

Worst case across all model/scale combos: VGAE small at 15.2/154.6 = 9.8% overhead.

|                | OLD (3 workers) | NEW (prebatched) |
|----------------|-----------------|------------------|
| T_step         | ~165 ms         | ~155 ms          |
| GPU util       | ~60-83%         | ~100%            |
| CPUs           | 5               | 1-2              |

Workers add IPC serialization overhead (~2-5 ms) for zero benefit when each
`__getitem__` is O(1) list lookup. See `graphids/core/data/datamodule/graph.py:328`.

## Generalization and limits — the probe is dataset- and scale-conditional

The "~100% GPU util" line above is true for the regime it was measured in
(hcrl_sa, ~400k-node budgets, V100). It does **not** generalize to all
combinations. A counter-measurement from set_01 makes the pattern
explicit.

### set_01, VGAE small, V100 (job 47045030, 2026-04-22)

| Metric | Value | Source |
|---|---|---|
| GPU 0 utilization — mean | 27.4% | MLflow `system/gpu_0_utilization_percentage`, 5s sampling, 643 points |
| GPU 0 utilization — median | 16.0% | same |
| Fraction of samples < 80% | 86.0% | same |
| GPU VRAM usage — median | 93.3% | `system/gpu_0_memory_usage_percentage` |
| Wall-clock, 197 epochs | 54:31 | sacct |

VRAM is packed correctly — the two-point probe is sized right. But GPU
compute idles ~86% of the time.

### Why the probe doesn't predict this

Per-step util obeys `GPU_util ≈ T_gpu / (T_gpu + T_cpu_step)`. Both
terms scale with the workload:

| Term | hcrl_sa VGAE-small (Apr 7) | set_01 VGAE-small (Apr 22) | Scaling |
|---|---|---|---|
| T_gpu | ~155 ms | **~5 ms (est.)** | ∝ batch_nodes × model_FLOPs |
| T_cpu_step | ~15 ms (pin+H2D) | ~20-30 ms (clone+dispatch) | relatively flat |

Under the April probe, T_gpu ≫ T_cpu — prebatch fully hides the CPU
step. Under a smaller model or tighter node budget, T_gpu shrinks
faster than T_cpu does, and util drops mechanically even with the
same pipeline. **This is a workload characteristic, not a regression.**

### What's load-bearing in the pipeline (don't "fix" these)

- **`pin_memory=device is None`** at `_spawn_loader` (graph.py:56) and
  missing from `_prebatched_loader` (graph.py:71-81) is deliberate.
  PyG's `PrefetchLoader` (called from `_prefetch`, graph.py:23-27)
  owns pinning + async H2D on its own CUDA stream. Enabling
  `pin_memory=True` at the DataLoader level would double-pin.
- **`_clone_collate`** at graph.py:38-40 runs every step because
  PyG `Data.to(device)` is in-place (see `critical-constraints.md`)
  and the pre-built batches are shared state. This is the last
  main-process CPU cost in the prebatch path.

### Configuration guidance

- **Smoke runs (`--smoke` → gpudebug 1h wall).** Expect low GPU util
  on small models / small datasets. Measure correctness (no NaN, no
  IndexError, loss curve shape), not throughput.
- **Production fits.** Use `--length long` (gpu partition, 4h+) and
  `scale="medium"` or `"large"` so T_gpu dominates. For set_01, move
  to `--cluster cardinal` (H100) — the compute headroom lets you
  push node budgets high enough to saturate. The April hcrl_sa probe
  numbers are representative of this regime.
- **Don't chase GPU util on smoke runs.** A compute-tiny model on a
  V100 will show 20-40% util no matter how well the data pipeline is
  tuned — that's hardware + model, not pipeline.

### Diagnostic hierarchy

1. `MLflow system/gpu_*` metrics (5s sampler, already on) — aggregate
   util, memory, power. Queryable across runs.
2. OTel `ml.batch.duration_s` in `traces.jsonl` — per-step wall.
3. Explicit `nvidia-smi dmon -s u` during a live run — real-time
   nvml counters.

Never reason about throughput from wall-clock alone — matches the
`feedback_device_metrics_first` lesson.
