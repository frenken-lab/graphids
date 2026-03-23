# GPU Utilization & Training Efficiency — First Principles Analysis

**Date**: 2026-03-23
**Context**: V100 GPU (32GB) on OSC Pitzer. GPU utilization ~50% during training. `safety_factor` hack (0.45–0.90) halves batch sizes to avoid OOM.

---

## The Problem

Three symptoms, one root cause:
1. GPU utilization stuck at ~50%
2. OOM crashes with full batch sizes → `safety_factor` hack
3. Preprocessing cache jobs OOM at 48GB, need 96GB

## What's on the GPU During Training

### Memory breakdown (VGAE large, batch_size=4096, safety_factor=0.5)

| Component | Footprint | Notes |
|---|---|---|
| Model parameters | ~3-5 MB | Small GNN: [480→240→48] hidden dims |
| Optimizer state (Adam m+v) | ~10-15 MB | 2× model size, always float32 |
| Embedding table | ~64 KB | 500 IDs × 32 dim × 4 bytes. **Not the issue.** |
| Batch data (x, edge_index, edge_attr, ...) | ~50-100 MB | Scales with node_budget |
| **Forward activations (6 layers)** | **~750 MB - 1.5 GB** | **The dominant cost.** Scales with total_nodes × hidden_dim × num_layers |
| Backward activations + gradients | ~1-2.4 GB | Retained for backprop unless checkpointed |
| Teacher model (KD mode) | ~3-5 MB | On GPU when `offload_teacher_to_cpu=False` |
| PyTorch allocator overhead | ~200 MB | Fragmentation from variable-size batches |
| **Total** | **~2.5-4.8 GB** | Out of 32 GB. Headroom exists. |

### Why only 2.5-4.8 GB on a 32 GB card?

Because `safety_factor=0.5` halves the node budget:
- Nominal: `batch_size=4096 × p95_nodes≈100 = 409,600 nodes → ~4.8 GB`
- With safety_factor=0.5: `2048 × 100 = 204,800 nodes → ~2.5 GB`

**The GPU has 27+ GB of unused VRAM.** The model is too small and batches too conservative to fill it.

## Why OOM Happens Despite Small Total Memory

**Variable-size graph batching creates spikes.** DynamicBatchSampler packs graphs to a node budget, but:

1. **Outlier graphs**: A window with 200 nodes (vs p95=100) consumes 2× expected memory
2. **Activation scaling**: GATConv attention creates `[num_edges × num_heads]` intermediates. Dense CAN bus windows can have 100+ edges per node → quadratic spike
3. **Fragmentation**: CUDA allocator can't reuse freed blocks when batch sizes vary wildly → `torch.cuda.memory_reserved()` >> `memory_allocated()`
4. **Gradient checkpointing gaps**: Current checkpointing covers conv layers only, not batch norm/activation/dropout intermediates

**The safety_factor is a blunt instrument** — it caps ALL batches to avoid rare spikes, wasting 27 GB of VRAM on the 99% of normal batches.

## The Tools Available

### 1. SLURM (resource allocation)

| Lever | Current | Notes |
|---|---|---|
| `--cpus-per-task` | 4 | 1 main + 1 GPU kernel + 2 DataLoader workers |
| `--mem` | 48G | CPU-side RAM for data staging + preprocessing |
| `--gres=gpu:1` | 1 V100 | 32 GB VRAM |
| TMPDIR | Local SSD | Data staged here by `_preamble.sh` for fast I/O |

### 2. PyG DataLoader

| Lever | Current | Purpose |
|---|---|---|
| `DynamicBatchSampler` | Enabled | Packs graphs to node budget instead of fixed count |
| `num_workers=2` | Correct for 4 CPUs | Workers prefetch batches while GPU computes |
| `pin_memory=True` | Enabled | Enables async CPU→GPU transfer with `non_blocking` |
| `persistent_workers=True` | Enabled | Avoids Python interpreter restart per epoch |
| `exclude_keys` | **Not used** | Could skip transferring unused Data attributes |

### 3. CUDA / PyTorch

| Lever | Current | Purpose |
|---|---|---|
| `precision="16-mixed"` | Enabled | Float16 forward/backward, float32 optimizer |
| `gradient_checkpointing` | Partial (conv only) | Trades compute for memory on intermediates |
| `torch.compile` | Disabled | Kernel fusion for small ops → better utilization |
| `non_blocking=True` | **Just added** | Overlaps CPU→GPU copy with GPU compute |
| `set_sharing_strategy("file_system")` | **Just added** | Avoids /dev/shm mmap OOM in multiprocessing |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Already set in `_preamble.sh`, reduces fragmentation |

### 4. CPU→GPU I/O

| Stage | Current | Bottleneck? |
|---|---|---|
| Disk → CPU | mmap=True on torch.load | Fast if data on TMPDIR (local SSD). Slow if NFS. |
| CPU → GPU | pin_memory + non_blocking | Now overlapped with compute |
| Within GPU | Mixed precision | Forward in fp16, backward in fp16, optimizer in fp32 |

## Changes Made in This Session

### Preprocessing (cache rebuild)
1. **Sequential `collect()`** — reduces peak memory ~20-30 GB
2. **Polars vectorized local IDs + bulk `to_torch()`** — eliminates per-window Polars overhead, ~7x loop speedup
3. **Parallel graph assembly** — `ProcessPoolExecutor` with spawn, adaptive chunking, numpy IPC
4. **`file_system` sharing strategy** — in both parent and worker processes for HPC compatibility
5. **NetworkX clustering** — kept (no reinvention), distributed across 8 workers

### Training pipeline
6. **`non_blocking=True`** on all data-transfer `.to(device)` calls:
   - `data_loading.py:99` — `cache_predictions` batch loop
   - `eval_inference.py:144,158,190` — GAT/VGAE evaluation loops
   - `training.py:146` — VGAE state caching
   - `temporal.py:100,103,106,121,124` — temporal stage (was worst offender: per-graph sync transfer in training_step)
   - `cka.py:59` — CKA analysis

7. **`set_sharing_strategy("file_system")`** in `__main__.py` — fixes DataLoader mmap OOM for training

## What Would Actually Improve GPU Utilization

### Tier 1: Configuration changes (no code)

| Change | Effect | Risk |
|---|---|---|
| Raise `safety_factor` to 0.9+ | Larger batches → GPU busy longer per step | OOM on outlier batches |
| Enable `torch.compile` (`compile_model: true`) | 10-30% kernel throughput via fusion | One-time warmup; may not support all PyG ops |
| Increase `batch_size` in model presets | Same as raising safety_factor | Same OOM risk |

### Tier 2: Targeted code changes

| Change | Effect | Effort |
|---|---|---|
| **Replace safety_factor with per-batch OOM retry** | Full VRAM utilization on normal batches, graceful fallback on spikes | Medium — catch CUDA OOM in training_step, halve batch, retry |
| **Extend gradient checkpointing** to batch norm + activation | ~30% activation memory reduction → larger safe batches | Medium — wrap full conv_forward block |
| **`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`** | Reduce fragmentation from variable-size batches | Config only — add to _preamble.sh |
| **Profile actual peak memory** per dataset | Replace empirical safety_factors with data-driven values | Add `torch.cuda.max_memory_allocated()` logging |

### Tier 3: Architectural (requires design)

| Change | Effect | Effort |
|---|---|---|
| **Batch-level memory estimator** | Predict batch memory from node/edge counts, skip too-large batches | High — needs profiling data |
| **Gradient accumulation** | Simulate large batches with small physical batches | Medium — Lightning supports `accumulate_grad_batches` |
| **Teacher CPU offload** (`offload_teacher_to_cpu: True`) | Free teacher model VRAM during non-KD steps | Config change, already implemented |

## The Clean Fix for safety_factor

The safety_factor exists because DynamicBatchSampler's node budget can't predict activation memory. The clean replacement:

```python
# In training_step or a custom callback:
try:
    loss = self._forward_and_loss(batch)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Split batch in half and retry
    half = batch.num_graphs // 2
    batch1 = Batch.from_data_list(batch.to_data_list()[:half])
    batch2 = Batch.from_data_list(batch.to_data_list()[half:])
    loss = (self._forward_and_loss(batch1) + self._forward_and_loss(batch2)) / 2
```

This uses full VRAM capacity on 99% of batches and gracefully handles the 1% outliers. No more guessing safety_factors.

## Training Code: Per-Step Resource Waste

Audit of the training loop found 6 additional issues beyond the `.to(device)` fix.

### 1. `create_neighborhood_targets` — dense O(N × num_ids) allocation every step (CRITICAL)

`vgae.py:196`: `torch.zeros(num_nodes, self.num_ids, device=node_id.device)`

Every training AND validation step allocates a dense `[total_batch_nodes, num_ids]` float32 target matrix on GPU. For a batch with 10K nodes and 500 CAN IDs = 20 MB. This is then fed to `F.binary_cross_entropy_with_logits` which allocates another 20 MB for the loss. Both are freed immediately, causing constant alloc/free churn.

**Fix**: Pre-allocate a reusable buffer, or use sparse targets with `F.multilabel_soft_margin_loss`.

### 2. DynamicBatchSampler 500-graph scan every DataLoader creation (HIGH)

`data_loading.py:67-70` iterates 500 random graphs to estimate `mean_nodes`. This runs:
- Once per `trainer.fit()` for `CANBusDataModule`
- **Every epoch** for `CurriculumDataModule` (rebuilds DataLoader at `modules.py:344`)

**Fix**: Cache `mean_nodes` in `cache_metadata.json` during preprocessing.

### 3. CurriculumDataModule rebuilds DataLoader every epoch (HIGH)

`modules.py:344`: `train_dataloader()` calls `make_dataloader()` every epoch to adjust the curriculum sample. This destroys persistent workers and triggers the 500-graph scan again.

**Fix**: Use `Sampler` replacement instead of rebuilding the entire DataLoader.

### 4. `_curriculum_sample` full Python sort every epoch (LOW-MEDIUM)

`modules.py:374`: `sorted(scores)[int(len(scores) * percentile / 100)]` — full Python sort of all normal-graph scores every epoch.

**Fix**: Use `np.partition` (O(n)) instead of `sorted` (O(n log n)).

### 5. Teacher GPU residency + `empty_cache()` tradeoff (MEDIUM)

`offload_teacher_to_cpu=False` (default) → teacher lives on GPU permanently (~100-400 MB VRAM wasted).
If set to True, `_teacher_on_device` context manager calls `torch.cuda.empty_cache()` on every step exit — a full CUDA context flush.

**Fix**: Offload teacher to CPU but remove the `empty_cache()` call (let PyTorch's caching allocator handle it).

### 6. `cache_predictions` unpacks batches with `to_data_list()` (LOW, DQN only)

`data_loading.py:100`: `for g in batch.to_data_list()` — converts batched GPU tensor back to individual Python Data objects inside a loop. Defeats batched inference.

**Fix**: Use scatter reduction on the batched tensors directly (already done elsewhere in the codebase).

## Preprocessing: Parallel Assembly Status

The parallel graph assembly (Opt 4) has a blocking issue: `spawn` worker startup overhead exceeds the compute savings. Each worker imports torch + PyG + scipy (~3-5s), and 8 workers competing for 8 CPUs causes a ~25 minute hang.

**Status**: Sequential mode (`KD_GAT_GRAPH_WORKERS=1`) is being benchmarked. The vectorization gains from Opts 1-3 (Polars join + bulk to_torch + torch slicing) are the real win — projected ~7x improvement even without multiprocessing.

**Options for parallelism**:
- `fork` context would avoid reimporting, but violates CUDA safety constraints
- Reduce to 2-4 workers (less CPU contention)
- Use `threading` instead (GIL-limited but avoids spawn overhead for I/O-bound work)
- Pre-warm a persistent worker pool across epochs (not per-call)
- Accept sequential with the ~7x vectorization win

## Verification Checklist

- [ ] Benchmark hcrl_ch sequential cache rebuild (job 45969879, target: <5 min vs 15 min baseline)
- [ ] Run full test suite via SLURM
- [ ] Profile GPU memory during VGAE training: `torch.cuda.max_memory_allocated()`
- [ ] Verify `KD_GAT_LAKE_ROOT` points to TMPDIR in SLURM jobs (not NFS)
- [ ] Measure `create_neighborhood_targets` allocation size per dataset
- [ ] Test `safety_factor=0.9` after fixing neighborhood target allocation
- [ ] Test `compile_model=true` compatibility with PyG GATConv
