# DataLoader Performance Analysis

> Consolidated from 6 investigation files. Last updated: 2026-04-02.
> Resource profiles: `configs/resources/job_profiles.json` (source of truth).

## Current State

| Setting | Value | Where configured |
|---------|-------|-----------------|
| Start method | `spawn` | `__main__.py` (process-global) |
| Sharing strategy | `file_system` | `__main__.py` (process-global) |
| num_workers | 2 | Stage YAML `data.init_args.num_workers` |
| persistent_workers | True | `make_graph_loader` default |
| Collation | PyG standard `Batch.from_data_list()` | PyG DataLoader default |
| Batch sampler | `DynamicBatchSampler` | `make_graph_loader` |
| Precision | 16-mixed | `trainer.yaml` |

**Steady-state:** T_c ≈ 52ms (warm cache, epoch 2+). T_gpu ≈ 10ms (VGAE) / 25ms (GAT).

---

## Key Findings

### 1. Collation: warm cache beats FastCollate

| Path | Time/batch | When |
|------|-----------|------|
| FastCollate (vectorized) | 85ms | Every batch |
| `from_data_list` cold | 166ms | Epoch 1 only |
| `from_data_list` warm | 52ms | Epoch 2-300 |

With `persistent_workers=True`, workers survive across epochs. FastCollate was correctly deleted. Standard collation with warm cache is faster.

### 2. Memory: RSS is inflated, PSS is the real metric

Workers mmap the same shared memory file — tensor data is shared, not copied.

| Metric | 2 workers, set_02 | Notes |
|--------|-------------------|-------|
| PSS (real) | ~22.5G | Physical unique pages |
| RSS (reported) | ~37.7G | Double/triple counts shared pages |

`--mem` requests of 36-48G were ~2x over-provisioned. **Needs GPU-node PSS verification** (`cat /proc/self/smaps_rollup | grep Pss` in `worker_init_fn`).

### 3. Multiprocessing: current setup is correct

- **spawn** required (CUDA initialized before DataLoader)
- **file_system** required on OSC (`vm.max_map_count=65530` breaks file_descriptor)
- **forkserver** not worth it (23% cold start improvement on epoch 1 of 300)
- **persistent_workers** essential for warm-cache performance

### 4. V100 is the sweet spot for this workload

Faster GPUs are harder to feed because T_c is CPU-bound. V100's T_gpu/T_c ratio is most favorable. A100/H100 only make sense for larger models. See `throughput-model.md` §2 for the analysis.

---

## Open Items

### CurriculumDataModule rebuilds DataLoader every epoch

Rebuilds DataLoader → kills persistent workers → 3-5s spawn per worker per epoch.
300 epochs × 2 workers × 4s = **40 min of pure spawn overhead**.
Fix: Create DataLoader once, update `CurriculumSampler.set_epoch()` only.

### PSS verification on GPU node

Submit short job with PSS logging in `worker_init_fn` to confirm under real SLURM accounting. If confirmed, reduce `--mem` in resource profiles.

### Per-worker `_data_list` cache bloat

Each worker builds ~1.5-2G `_data_list` cache on set_02. With `persistent_workers=True`, cache persists. No fix identified without sacrificing warm-cache performance.

---

## Cross-references

- `data-flow.md` — full data flow diagrams
- `observability.md` — tool evaluations and GPU profiling invocations
- `throughput-model.md` — cost model, regime analysis, budget system
- `../decisions/0004-keep-custom-vram-probe.md` — why profilers can't replace probe
