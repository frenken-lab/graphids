# CAN Bus GNN Pipeline — Session Ground Truth
> Paste this at the start of every Claude session. Last updated: 2026-04-04.
> Source of truth for pipeline state, what has been tried, what was rejected, and open problems.
>
> **Scope:** data pipeline + throughput model + probe only. Does **not** cover
> orchestration (see §0 below for pointers).

---

## 0. What This Doc Does NOT Cover

This document is scoped to the training data pipeline, throughput model, and
VRAM/worker sizing. Architecture layers above are authoritative elsewhere — do
not infer them from this doc:

| Layer | Entry point | Code |
|---|---|---|
| Config resolution (merge + audit trail) | `ConfigResolver.resolve()` | `graphids/orchestrate/resolve.py` |
| Dagster assets / SLURM submission | `dg launch`, `scripts/submit.sh` | `graphids/orchestrate/{assets,component,slurm}.py` |
| Run record sidecars (status, metrics, phases) | `RunRecordCallback` | `graphids/core/contracts/run_record.py`, `graphids/core/models/_lightning.py` |
| DuckDB catalog rebuild | `python -m graphids rebuild-catalog` | `graphids/commands/rebuild_catalog.py` |
| Multi-point budget calibration CSV | `python -m graphids probe-budget` | `graphids/commands/profile_budget.py` (commit `6e3424a`) |
| Cost-model plots (Altair + Polars) | `python -m graphids.plots.budget` | `graphids/plots/{budget,transforms}.py` (commit `6e3424a`) |

See `CLAUDE.md` and `docs/decisions/` for these. Every training run writes a
`run_record.json` sidecar consumed by the catalog; the pipeline here is
downstream of that.

---

## 1. Project Context

PhD research at Ohio State (CAR Mobility Systems Lab). Building GNNs for CAN bus intrusion
detection targeting ICML 2026. Models: VGAE, GAT, DGI, DQN fusion ensemble.
Compute: OSC (Ohio Supercomputer Center) Pitzer nodes, V100 16GB, SLURM scheduler.
Stack: PyTorch Geometric (PyG), PyTorch Lightning, Polars, W&B, Ray.

---

## 2. Pipeline Architecture (Ground Truth)

### Phase 1 — Preprocessing (runs once, CPU only)

```
Raw CSVs (NFS)
  → pl.scan_csv() [lazy]
  → column normalization, hex payload → 8×Float32, Shannon entropy, attack tagging
  → sort by timestamp → .collect() [FIRST MATERIALIZATION, ~5M rows × 15 cols]
  → vocabulary: unique arb_ids → dense int IDs (~30-50 unique)
  → sliding_window_graphs() [window=100, stride=100 → ~50K windows]
      → Three parallel lazy frames:
          stats_lf:    group_by(_wid, node_id) → 35 node features
          edges_base:  shift(-1).over(_wid)    → 11 edge features
          labels_lf:   group_by(_wid)          → y, attack_type
      → Sequential .collect() [saves ~20-30GB peak vs parallel]
      → local ID remapping (bulk Polars join)
      → Polars → torch bulk handoff (.to_torch Float32)
      → Zero-copy collation: bulk tensors ARE the collated format.
        RLE boundaries from group_by become slices dict directly.
        No per-window Data objects. No list[Data]. No collate() call.
        Peak memory: ~1x final tensor size (was ~3x before 2026-03-31 fix).
  → Returns (Data, slices_dict, num_graphs) from bulk tensors
  → Presort kept graphs by node count before materialization (v8.0.0)
      → similar-size graphs are adjacent on disk
      → NodeBudgetBatchSampler + bucket shuffle → sequential page faults
      → stabilizes VRAM allocator block reuse (less fragmentation)
  → atomic_save() → torch.save + fsync + rename
  → cache/v8.0.0/{dataset}/processed/data_train.pt  [set_02 ≈ 5.9 GB]
```

### Phase 2 — Training Data Loading (every epoch, every batch)

```
Storage: NFS (~50ms) → Scratch/GPFS (~5ms) → TMPDIR/local SSD (~0.1ms)
         staged by _preamble.sh

Setup (once):
  torch.load(data_train.pt, mmap=True)    ← memory-mapped, pages fault on access
  train/val split: torch.randperm(seed=42), 80/20, both share mmap'd data

Per-batch loop:
  NodeBudgetBatchSampler [main process] (v8.0.0)
    → reads num_nodes_per_graph from dataset.slices (zero mmap reconstructions)
    → bucket shuffle: sort by size → N buckets → shuffle bucket + within-bucket order
    → walks sizes in bucketed order, accumulates until budget
    → sends index lists to worker queues
    → Replaces PyG's DynamicBatchSampler (which walked dataset[i].num_nodes
      per graph per epoch = 50K mmap Data reconstructions per epoch on set_02).

  Worker processes [spawn, num_workers=2, persistent_workers=True]
    → dataset[i] → InMemoryDataset.__getitem__ → tensor slicing
    → Batch.from_data_list(data_list)           ← COLLATION (hot loop)
        Cold (epoch 1, _data_list=None): ~166ms/batch
        Warm (epoch 2+, _data_list cached):  ~52ms/batch
    → return Batch via IPC (file_system sharing strategy)

  pin_memory=True → page-locked RAM
  batch.to(device, non_blocking=True) → async DMA (~1-2ms)

  GPU: VGAE ~10ms fwd+bwd | GAT ~25ms fwd+bwd
```

### Steady-State Timing (measured, Run 003)

```
Time (ms):  0    10    20    30    40    50    60
Worker 0:   │▓▓▓▓▓▓▓▓▓▓ warm collate (52ms) ▓▓▓▓│▓▓▓▓...
Worker 1:   │  ▓▓▓▓▓▓▓▓▓▓ warm collate ▓▓▓▓▓▓▓│▓▓▓...
GPU:        │█fwd█│█bwd█│    gap ~30ms    │█fwd█│

Predicted util: 38%  (2 workers / 52ms collate = 38 batches/s vs GPU 100/s)
Measured util:  83-90%  (prefetch buffer smoothing + batch size variance)
```

### Measured Probe Values (job 46273452, Pitzer V100, set_01)

| model | scale | bpn | bwd_mult | α (ms) | β (μs/node) | γ (μs/graph) |
|-------|-------|-----|----------|--------|-------------|--------------|
| vgae | small | 34,601 | 1.39 | 7.1 | 0.00 | 65 |
| vgae | large | 50,112 | 1.26 | 6.9 | 0.16 | 65 |
| gat | small | 59,838 | 1.29 | 2.7 | 0.85 | 65 |
| gat | large | 223,738 | 1.52 | 4.6 | 0.73 | 65 |
| dgi | small | 13,974 | 2.0* | 7.1 | 0.03 | 65 |
| dgi | large | 80,142 | 2.0* | 6.1 | 0.06 | 65 |

*DGI backward probe fails, falls back to _GRAD_MULTIPLIER=2.0.

### Derived Worker Requirements (from throughput model)

| model/scale | mem_budget (nodes) | T_gpu (ms) | T_collate (ms) | Workers needed |
|-------------|-------------------|------------|----------------|----------------|
| vgae/small | ~400K | ~7 | ~920 | 132 (impractical) |
| vgae/large | ~230K | ~44 | ~530 | **13** |
| gat/small | ~190K | ~164 | ~437 | **3** |
| gat/large | ~52K | ~42 | ~120 | **3** |
| dgi/small | ~600K | ~25 | ~1,384 | 56 (impractical) |
| dgi/large | ~145K | ~15 | ~334 | 23 |

VGAE/small and DGI/small: β ≈ 0 means GPU finishes in α ≈ 7ms regardless of batch size.
No number of workers can keep up. Candidates for CUDA Graphs or CPU-only training.

---

## 3. Confirmed Diagnosis: Overhead-Bound Regime

This pipeline is **overhead-bound**, not memory-bound or compute-bound.
The GPU idles waiting for CPU to collate. Adding kernel fusion, mixed precision,
or operator optimization does nothing when the GPU has no work.

**The correct fix chain:**
```
VRAM capacity → max batch size → T_gpu per step
                               → T_collation per step (= γ × B)
                                   → workers = ceil(T_collation / T_gpu)
                                       → CPUs = workers + 2
                                           → SLURM request
```

**Rule:** Maximize batch size first (fills VRAM). Then scale workers to match.
Never shrink batch to fit workers — workers are cheap, GPU time is not.

**Literature backing:**
- SALIENT (2021): 3x speedup from pipeline optimization alone. ~28% of time is GPU.
- BGL/NSDI 2023: ~10% GPU utilization typical in DGL training.
- PyG #4891: DataLoader is 59-83% of runtime.
- NVIDIA DL Perf Guide: increase batch size when overhead-bound.
- Horace He "Making DL Go Brrr": increase data size to escape overhead-bound regime.

---

## 4. Known Bugs and Constraints

### ~~Bug: torch.compile VRAM pool inflation~~ — **FIXED (commit `ad23162`)**
`torch.compile` used to inflate the reserved pool by 4-8GB before the probe,
causing 2-3x batch underestimates. `expandable_segments:True` lets the allocator
return unused segments, so the probe sees real free VRAM.

Evidence: `scripts/slurm/_preamble.sh:38-40` —
```bash
if [[ "${SKIP_CUDA_CONF:-0}" != "1" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
fi
```
Every SLURM job inherits this unless `SKIP_CUDA_CONF=1` (CPU-only jobs).

### ~~Bug: γ measurement contaminated by GPU state~~ — **FIXED (commit `dba35a3`)**
`calibrate_at_budget()` now synchronizes before every CPU timing block.

Evidence: `graphids/core/preprocessing/budget.py:116-124` —
```python
torch.cuda.synchronize()
gc.collect()
collation_samples = []
for _ in range(3):
    t0 = time.perf_counter()
    batch = Batch.from_data_list(graphs)
    collation_samples.append(time.perf_counter() - t0)
t_collation = statistics.median(collation_samples)
```
3-sample median, measured at the real operating batch size. `_probe_vram()` also
synchronizes at `budget.py:252, 260, 278` around all peak-memory reads.

### Constraint: vm_map_count = 65,530 (Linux default, no sudo on OSC)
Every mmap region consumes a vm_area_struct entry. With mmap=True on a 5.9GB file,
multiple workers, IPC file_system sharing, and CUDA context mappings, the budget
fills fast. Adding workers without checking this causes silent slowdowns.

**Check during training:**
```bash
cat /proc/$(pgrep -f "train.py")/maps | wc -l
```

**Fix without sudo:** Load fully into RAM (no mmap=True) if TMPDIR has space:
```python
data = torch.load(f"{os.environ['TMPDIR']}/data_train.pt")  # no mmap=True
```
This eliminates mmap vm regions for the data file. Workers share via shared_memory IPC.

### Constraint: IPC strategy is file_system (suboptimal)
Current: workers serialize Batch → write temp file → main reads → deserializes.
Better: shared_memory (direct pointer, no serialize/deserialize, no file I/O).

**Fix:**
```python
import torch.multiprocessing as mp
mp.set_sharing_strategy('shared_memory')
```
Risk: shared_memory consumes file descriptors (1 per tensor). With num_workers=2
and current batch sizes this is safe. Check: `ulimit -n` (typically 4096 on OSC).

### Constraint: VRAM 4GB used / 12GB reserved gap
Not a utilization problem — it's allocator fragmentation from variable batch sizes.
DynamicBatchSampler produces high-variance batches; allocator reserves new chunks
for unexpected large batches instead of reusing cached blocks.

**Fix:** Bucket shuffle in DynamicBatchSampler (see Section 6).

---

## 5. What Has Been Tried — Do Not Re-suggest

### ✅ Already implemented and working
- **Zero-copy bulk tensor collation** (2026-03-31) — `core/preprocessing/features.py:415-481`. Eliminated 3x memory peak; `Data`+`slices_dict` built directly from RLE boundaries, no per-window `Data` objects, no `collate()` call.
- **pin_memory=True** — `core/preprocessing/datamodule.py:36` default.
- **persistent_workers=True** — `datamodule.py:52` (when `num_workers > 0`).
- **multiprocessing_context='spawn'** — `datamodule.py:53` (mandatory with CUDA; see `.claude/rules/critical-constraints.md`).
- **non_blocking=True transfers** — `datamodule.py:323` (FusionDataModule); `PrefetchLoader` for the graph path.
- **mmap=True on torch.load** — `core/preprocessing/datasets/can_bus.py:90-92`.
- **DynamicBatchSampler (node budget)** — `datamodule.py:249-251`, `max_num=result.budget, mode="node"`.
- **Polars lazy preprocessing** — `core/preprocessing/features.py:244` (`df.lazy()`), sequential `.collect()` at line 277-282 to cap peak memory.
- **AMP (mixed precision)** — enabled via trainer precision in `config/defaults/trainer.yaml`.
- **`expandable_segments:True`** — `scripts/slurm/_preamble.sh:38-40` (commit `ad23162`). Fixes torch.compile VRAM pool inflation.
- **γ timer synchronization** — `core/preprocessing/budget.py:116` (commit `dba35a3`). `torch.cuda.synchronize()` before every CPU timing block, 3-sample median, measured at the operating batch size (not extrapolated).
- **`ResourceProfile` dataclass + GPU-first sizing chain** — `budget.py:70-84` and `compute_resource_profile()` at `budget.py:161-210` (commit `dba35a3`). Derives `workers = ceil(t_collation_us / t_gpu_us)` capped to `SLURM_CPUS_PER_TASK - 2`, `prefetch_factor = 4 if workers ≥ 8 else 2`, memory = `workers × rss + base + headroom`. Wired into `datamodule.py:203, 223-245`.
- **Multi-point probe-budget CLI** — `graphids/commands/profile_budget.py` (commit `6e3424a`). Measures at 4 VRAM fractions (0.25/0.50/0.75/1.0) and writes `{lake_root}/reference/budget_calibration.csv` for downstream model fitting (γ, α, β via least-squares) in `graphids/plots/`.
- **Budget is purely VRAM-ceiling** — `budget.py:390` (`mem_budget = int(effective_free * _SAFETY_MARGIN / effective_bpn)`). No throughput floor clamps the batch. (Old "shrink the batch to match throughput" framing is gone; see commit `75a6f39`. `plots/transforms.py:51-58` still has a `throughput_floor()` method, but it is an *analytic* helper that computes the minimum batch to amortize α for visualization — it does **not** clamp the training budget.)
- **Presorted v8.0.0 cache** — `core/preprocessing/features.py` sorts kept graphs by node count before building `(Data, slices)`. Adjacent graphs on disk have similar size; bucket shuffle produces sequential mmap page faults instead of random ones. Also reduces VRAM allocator fragmentation (the 4GB/12GB reserved gap in §4) by shrinking batch-to-batch size variance. Requires cache rebuild.
- **`NodeBudgetBatchSampler`** — `core/preprocessing/datamodule.py`. Custom sampler that reads `num_nodes_per_graph` from `CANBusDataset` (derived from `slices["x"]` at zero I/O, exposed as a property) and does bucket shuffle internally. Replaces PyG's `DynamicBatchSampler`, which was walking `dataset[i].num_nodes` per graph per epoch (50K mmap `Data` reconstructions per epoch on set_02, confirmed by inspecting `DynamicBatchSampler.__iter__`). The `dataset._data_list = None` hack is gone.

### ❌ Tried and rejected — do not re-suggest
- **Custom BulkTensorCollater bypassing from_data_list():** Implemented and tested.
  Was SLOWER after cold start than PyG's warm _data_list cache path.
  Reason: fine-grained slicing on mmap'd tensors causes page faults per slice,
  and may create additional vm regions that push toward the 65k map limit.
  PyG's _data_list cache (epoch 2+) works on already-materialized objects in
  worker RAM with no mmap access — that's why it wins warm.
  **Do not suggest custom collaters. The 52ms warm path is the correct baseline.**

### 🔲 Suggested but not yet implemented (open work items)
See Section 6.

---

## 6. Open Work Items (Prioritized)

> Completed items moved to §5. See commit refs there before re-suggesting:
> `ad23162` (expandable_segments), `dba35a3` (γ sync + ResourceProfile +
> calibrate_at_budget + workers formula), `75a6f39` (throughput floor removal),
> `6e3424a` (multi-point probe-budget + Altair plots).

> **Tracked elsewhere:** End-to-end SLURM wiring of `ResourceProfile` (probe →
> sbatch header) moved to frenken-lab/graphids#31. The sizing chain is
> implemented and wired to the DataLoader, but the sbatch allocation still
> comes from static `config/resources/profiles/{model}.yaml`. That issue also
> tracks two probe gaps surfaced during the audit: hardcoded worker RSS
> (`budget.py:167`) and unmeasured `vm_area_struct` count.

### Priority 1: Fix vm_map_count before scaling workers
**What:** Check current map count. If >40k, switch from mmap=True to full load.
**Steps:**
1. During a training run: `cat /proc/$(pgrep -f "train.py")/maps | wc -l`
2. If >40k: switch to `torch.load(path)` without mmap in TMPDIR
3. Switch IPC: `mp.set_sharing_strategy('shared_memory')`
4. Then safely increase num_workers

### ~~Priority 2: Bucket shuffle~~ — **SHIPPED in `NodeBudgetBatchSampler`**
Bucket shuffle is implemented in `core/preprocessing/datamodule.py`
(`_bucket_shuffled_indices`). Combined with the presorted v8.0.0 cache, this
addresses the 4GB/12GB VRAM reserved gap in §4 by keeping batch-to-batch
size variance low.

<details>
<summary>Historical reference design (kept for context — no longer needed)</summary>
**What:** Sort graphs by size before batching so adjacent batches have similar
node counts. Allocator reuses cached blocks instead of reserving new chunks.
Reduces 12GB reserved → closer to actual 4GB peak.

```python
class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, node_budget, shuffle=True):
        self.sizes = [data.num_nodes for data in dataset]
        self.node_budget = node_budget
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.sizes)))
        if self.shuffle:
            # Bucket shuffle: sort by size, shuffle within buckets
            indices.sort(key=lambda i: self.sizes[i])
            bucket_size = max(1, len(indices) // 20)
            buckets = [indices[i:i+bucket_size]
                       for i in range(0, len(indices), bucket_size)]
            random.shuffle(buckets)
            for b in buckets:
                random.shuffle(b)
            indices = [i for b in buckets for i in b]
        batch, current = [], 0
        for idx in indices:
            if current + self.sizes[idx] > self.node_budget and batch:
                yield batch
                batch, current = [], 0
            batch.append(idx)
            current += self.sizes[idx]
        if batch:
            yield batch
```
**Note:** PyG's own `DynamicBatchSampler` does not support bucket shuffle out
of the box. The sampler has been replaced with `NodeBudgetBatchSampler` which
does support it — see above.

</details>

### Priority 3: CUDA Graphs for VGAE/small and DGI/small (long-term)
**What:** These models have β ≈ 0, meaning T_gpu ≈ α ≈ 7ms (kernel launch overhead only).
No amount of workers keeps up because the GPU finishes before any pipeline depth helps.
CUDA Graphs capture the entire training step as one kernel, eliminating per-launch overhead.
**Status:** Not yet investigated. Requires static input shapes (problematic with
variable batch sizes from DynamicBatchSampler — needs padding or bucket-fixed sizes).
**See Section 12 for full design and implementation.**

---

## 7. Throughput Model (First Principles)

Throughput: T = N_V_batch / Δt_step

Step time with pipeline overlap (W workers, sufficient prefetch):
  Δt_step ≈ max(Δt_collate / W,  Δt_forward + Δt_backward)

Collation cost (PyG from_data_list, O(B)):
  Δt_collate ≈ γ × B     where γ = 65 μs/graph (measured)

GPU cost (affine model):
  T_gpu(N) = α + β × N   where α = kernel launch overhead, β = per-node cost

Pipeline saturation condition:
  workers_needed = ceil(Δt_collate / T_gpu)

**The key insight:** γ is fixed at 65μs/graph regardless of model.
T_gpu varies by model (β term). Larger models need fewer workers.
Smaller models (β ≈ 0) need impractical worker counts → use CUDA Graphs.

---

## 8. Budget.py Current State (commit `dba35a3`)

Two-stage design, both in `graphids/core/preprocessing/budget.py`:

1. **`node_budget()`** (line 309): VRAM ceiling only. Reads dataset mean/p95 from
   `cache_metadata.json`, reads free VRAM via `torch.cuda.mem_get_info()`, calls
   `_probe_vram()` (line 228) to measure `bytes_per_node` and
   `backward_multiplier` on a ~2000-node batch, applies edge-aware margin, and
   returns `BudgetResult.budget = mem_budget` (line 390). No throughput floor.

2. **`calibrate_at_budget()`** (line 91) + **`compute_resource_profile()`** (line 161):
   Measures `T_collation` (3-sample median) and `T_gpu` (BenchmarkTimer) at the
   actual operating batch size, then derives workers/prefetch/cpus/memory.

**`ResourceProfile` dataclass** (`budget.py:70-84`):
```python
@dataclass
class ResourceProfile:
    node_budget: int
    graphs_per_batch: int
    t_collation_us: float      # measured at operating batch size
    t_gpu_us: float            # forward × backward_multiplier
    workers: int               # ceil(t_collation / t_gpu), capped to max_cpus - 2
    prefetch_factor: int       # 2 for ≤4 workers, 4 for ≥8
    cpus: int                  # workers + 2
    memory_gb: int             # workers × rss + base + headroom
```

**Derivation** (`budget.py:183-191`):
```python
workers = max(1, math.ceil(t_collation_us / t_gpu_us))
if max_cpus is not None:
    workers = min(workers, max(1, max_cpus - 2))
prefetch_factor = 4 if workers >= 8 else 2
cpus = workers + 2
memory_gb = max(16, math.ceil(workers * worker_rss_gb + base_rss_gb + 4))
```

**Call site** (`datamodule.py:203, 223-245`): `_build_loader` reads
`SLURM_CPUS_PER_TASK`, calls `calibrate_at_budget` → `compute_resource_profile`,
and hands the resulting `workers/prefetch_factor` to PyG's `DataLoader`.
`nw = 2` at `datamodule.py:245` is the fallback only when the probe fails.

**Multi-point variant** (`commands/profile_budget.py`, commit `6e3424a`):
measures at 4 VRAM fractions (0.25/0.50/0.75/1.0), writes
`{lake_root}/reference/budget_calibration.csv`, fed into `graphids/plots/` for
least-squares fitting of (γ, α, β) across models.

---

## 9. Secondary Optimizations (apply after pipeline is fixed)

These only help once the GPU is actually busy. Current priority: low.

| Optimization | Regime | Status |
|---|---|---|
| expandable_segments | Overhead (VRAM pool) | ✅ Shipped — `_preamble.sh:38-40` (commit `ad23162`) |
| torch.compile | Memory/compute-bound | Enabled via `try_compile()` at `core/models/_training.py:13-39` (GPS skipped). Consider disabling for VGAE/small (wastes 4-8GB, β≈0 means no benefit) |
| Tensor core alignment (hidden dims ÷ 8) | Compute-bound | Not audited |
| CUDA Graphs | Overhead (kernel launch) | Future — needed for VGAE/small, DGI/small. See Section 12. |
| Gradient accumulation | Memory-bound | Not needed — VRAM headroom exists |
| prefetch_factor=3-4 | Overhead | ✅ Auto-selected — `budget.py:189` (`4 if workers ≥ 8 else 2`) |

---

## 10. Quick Diagnostic Commands

```bash
# Check vm_map_count during training
cat /proc/$(pgrep -f "train.py")/maps | wc -l

# Check TMPDIR available space (for full load vs mmap decision)
df -h $TMPDIR

# Check fd limit (for shared_memory IPC safety)
ulimit -n

# VRAM fragmentation check (add to training loop)
python -c "
import torch
print('allocated:', torch.cuda.memory_allocated()/1e9, 'GB')
print('reserved: ', torch.cuda.memory_reserved()/1e9, 'GB')
print('peak:     ', torch.cuda.max_memory_allocated()/1e9, 'GB')
"

# Profile collation vs transfer vs forward vs backward (run isolated)
# See throughput-model.md Section 1.7 for full diagnostic protocol
```


---

## 12. CUDA Graphs vs Triton — Decision and Implementation

### Why Not Triton (Yet)

Triton and CUDA custom kernels reduce **β** — the per-node execution cost inside a kernel.
They do not reduce **α** — the per-step kernel launch overhead.

From probe data:
```
vgae/large:  T_gpu = α + β×N = 6.9ms + 0.16μs×230K = 6.9ms + 37ms = ~44ms
             β term is meaningful. Workers fix this (see frenken-lab/graphids#31).

vgae/small:  T_gpu = α + β×N = 7.1ms + 0.00μs×400K = 7.1ms + 0ms  = ~7ms
             β ≈ 0. A perfect Triton kernel saves 0ms. α dominates.

gat/large:   T_gpu = α + β×N = 4.6ms + 0.73μs×52K  = 4.6ms + 38ms = ~42ms
             β term is real. After pipeline is fixed, Triton is justified here.
             GAT scatter/gather is sparse and fuse-able. But this is step 5.
```

**Rule:** Only consider Triton when β × N_budget is large relative to α AND the
pipeline (workers, expandable_segments) is already fixed. Current state: pipeline
is not fixed. Triton is premature.

**Triton becomes justified for GAT after Priorities 1-5 are complete:**
GAT's attention computation runs four separate kernels (feature projection,
attention coefficients, softmax, aggregation). A fused Triton kernel combines
these into one pass — reduces memory bandwidth and kernel count simultaneously.
Expected gain: 1.5-2x on GAT specifically. Weeks of implementation work.

### Why CUDA Graphs (For VGAE/small, DGI/small)

CUDA Graphs capture a sequence of GPU kernels as a single replayable graph.
The CPU dispatches once; the GPU replays with zero per-kernel launch overhead.
This directly attacks α, which is the only cost for β≈0 models.

```
Without CUDA Graphs:  CPU dispatches kernel 1, kernel 2, ..., kernel N per step
                      Each dispatch: ~3-5μs CPU overhead + PCIe round-trip
                      N kernels × 3μs = significant fraction of 7ms T_gpu

With CUDA Graphs:     CPU says "replay graph" once
                      GPU runs all kernels with no CPU involvement
                      α approaches zero; T_gpu approaches β×N (≈0 for VGAE/small)
```

**Constraint: CUDA Graphs require static input shapes.**
Your DynamicBatchSampler produces variable node counts per batch. Two solutions:

**Solution A — Bucket graphs (recommended):**
Pre-define discrete node count buckets. Capture one CUDA Graph per bucket.
Route each batch to nearest bucket, padding with dummy nodes if needed.

```python
# In datamodule.py or a new cuda_graph_manager.py

import torch
import math
from torch_geometric.data import Batch

BUCKET_SIZES = [50_000, 100_000, 150_000, 200_000, 250_000]  # tune to your distribution

class CUDAGraphManager:
    """
    Manages a pool of captured CUDA Graphs, one per node-count bucket.
    Handles padding, static buffer management, and graph replay.

    Usage:
        manager = CUDAGraphManager(model, optimizer, loss_fn, buckets=BUCKET_SIZES)
        manager.warmup(sample_batch)        # call once before training
        loss = manager.step(batch)          # replaces manual fwd/bwd/step
    """

    def __init__(self, model, optimizer, loss_fn, buckets, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.buckets = sorted(buckets)
        self.device = device

        # One static buffer set per bucket
        # Populated during warmup, reused every step
        self.static_inputs = {}   # bucket_size → dict of static tensors
        self.static_loss = {}     # bucket_size → scalar tensor
        self.graphs = {}          # bucket_size → CUDAGraph

    def _get_bucket(self, n_nodes):
        """Round up to nearest bucket. Raises if exceeds max."""
        for b in self.buckets:
            if n_nodes <= b:
                return b
        raise ValueError(
            f"n_nodes={n_nodes} exceeds largest bucket={self.buckets[-1]}. "
            f"Add a larger bucket or increase node_budget cap."
        )

    def _pad_batch(self, batch, target_nodes):
        """
        Pad batch to exactly target_nodes by appending dummy nodes.
        Dummy nodes have zero features and are masked in the loss.
        Edge index is unchanged — dummy nodes have no edges.
        """
        n_real = batch.num_nodes
        n_pad = target_nodes - n_real
        if n_pad < 0:
            raise ValueError(f"batch ({n_real} nodes) exceeds bucket ({target_nodes})")
        if n_pad == 0:
            return batch, n_real

        # Zero-pad node features
        pad_x = torch.zeros(n_pad, batch.x.size(1),
                            dtype=batch.x.dtype, device=batch.x.device)
        batch.x = torch.cat([batch.x, pad_x], dim=0)

        # Extend batch vector (all padding assigned to last graph, doesn't matter)
        pad_batch_vec = torch.full((n_pad,), batch.batch[-1].item(),
                                   dtype=torch.long, device=batch.x.device)
        batch.batch = torch.cat([batch.batch, pad_batch_vec])

        # y is graph-level — no padding needed
        return batch, n_real

    def warmup(self, sample_batches):
        """
        Capture CUDA Graphs for each bucket using sample batches.
        Call ONCE after model is compiled and moved to device.
        sample_batches: dict {bucket_size: Batch} or list of representative batches.

        Warmup sequence per bucket:
          1. Run 3 eager steps to initialize cuDNN state, allocator, etc.
          2. Capture the graph on step 4.
        """
        print("Warming up CUDA Graphs...")
        for bucket in self.buckets:
            if bucket not in sample_batches:
                print(f"  Skipping bucket {bucket} (no sample provided)")
                continue

            batch = sample_batches[bucket].to(self.device)
            batch, n_real = self._pad_batch(batch, bucket)

            # Allocate static input buffers (these are reused every step)
            self.static_inputs[bucket] = {
                'x': batch.x.clone(),
                'edge_index': batch.edge_index.clone(),
                'batch_vec': batch.batch.clone(),
                'y': batch.y.clone(),
                'n_real': n_real,
            }
            self.static_loss[bucket] = torch.zeros(1, device=self.device)

            # Eager warmup passes (3x to stabilize cuDNN + allocator)
            s = self.static_inputs[bucket]
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(s['x'], s['edge_index'], s['batch_vec'])
                loss = self.loss_fn(out[:s['n_real']], s['y'])
                loss.backward()
                self.optimizer.step()

            # Capture
            self.optimizer.zero_grad(set_to_none=True)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.model(s['x'], s['edge_index'], s['batch_vec'])
                # Mask loss to real nodes only — padding must not contribute
                captured_loss = self.loss_fn(out[:s['n_real']], s['y'])
                captured_loss.backward()
            self.graphs[bucket] = g
            self.static_loss[bucket] = captured_loss
            print(f"  Captured graph for bucket {bucket:,} nodes")

        print(f"CUDA Graphs ready for buckets: {list(self.graphs.keys())}")

    def step(self, batch):
        """
        Run one training step using CUDA Graph replay.
        Copies real batch data into static buffers, replays graph, steps optimizer.
        Returns scalar loss value.
        """
        n_real = batch.num_nodes
        bucket = self._get_bucket(n_real)
        batch = batch.to(self.device)
        batch, _ = self._pad_batch(batch, bucket)

        s = self.static_inputs[bucket]

        # Copy real data into static buffers (in-place, shape-preserving)
        # This is the only CPU→GPU transfer per step
        s['x'][:n_real].copy_(batch.x[:n_real])
        s['x'][n_real:].zero_()                     # zero out padding
        s['edge_index'].copy_(batch.edge_index)
        s['batch_vec'].copy_(batch.batch)
        s['y'].copy_(batch.y)
        s['n_real'] = n_real                         # update mask boundary

        # Replay — entire fwd+bwd runs on GPU with one CPU call
        self.graphs[bucket].replay()

        # Optimizer step runs eagerly (stateful, can't be captured easily)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return self.static_loss[bucket].item()
```

**Integration with PyTorch Lightning:**

CUDA Graphs conflict with Lightning's automatic `training_step` because Lightning
wraps forward/backward in hooks that break graph capture. Options:

Option A — Disable Lightning's automation for the training step:
```python
class GNNModule(pl.LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.automatic_optimization = False   # take manual control
        self.graph_manager = None             # set in on_fit_start

    def on_fit_start(self):
        # Build sample batches for warmup — one per bucket
        sample_batches = self._build_warmup_samples()
        self.graph_manager = CUDAGraphManager(
            self.model, self.optimizers(), self.loss_fn,
            buckets=BUCKET_SIZES, device=self.device
        )
        self.graph_manager.warmup(sample_batches)

    def training_step(self, batch, batch_idx):
        loss_val = self.graph_manager.step(batch)
        self.log('train/loss', loss_val, prog_bar=True)
        return torch.tensor(loss_val)   # Lightning expects a tensor

    def _build_warmup_samples(self):
        """Create one synthetic batch per bucket for graph capture."""
        samples = {}
        for bucket in BUCKET_SIZES:
            # Use real graphs up to bucket size from dataset
            # or synthesize: important that feature dims match exactly
            data_list = []
            total = 0
            for i in range(len(self.trainer.datamodule.train_dataset)):
                d = self.trainer.datamodule.train_dataset[i]
                if total + d.num_nodes <= bucket:
                    data_list.append(d)
                    total += d.num_nodes
                if total >= bucket * 0.9:   # 90% fill is enough
                    break
            if data_list:
                samples[bucket] = Batch.from_data_list(data_list)
        return samples
```

Option B — Use CUDA Graphs only outside Lightning (standalone training loop).
Simpler but loses Lightning callbacks, W&B logging, checkpointing. Not recommended.

**Known incompatibilities to check before implementing:**
- `torch.compile` + CUDA Graphs: can interact, but PyTorch 2.x supports this
  via `torch.compile(model, backend='cudagraphs')` — avoids manual capture entirely.
  Try this first before manual CUDAGraphManager.
- AMP (mixed precision): supported, but `GradScaler` must run outside the graph.
  Move `scaler.step(optimizer)` and `scaler.update()` after `graph.replay()`.
- DGI's `_GRAD_MULTIPLIER=2.0` fallback: if DGI backward is captured in the graph,
  the gradient multiplier must be baked in as a static scalar — cannot change per-step.

### Simpler Alternative: torch.compile(backend='cudagraphs')

Before building CUDAGraphManager manually, try:

```python
model = torch.compile(model, backend='cudagraphs')
```

This tells the compiler to automatically capture and replay CUDA Graphs for static
sub-graphs within the model, without requiring fixed input shapes at the Python level.
It handles shape changes by recompiling — acceptable if DynamicBatchSampler produces
a bounded number of distinct shapes (the presorted v8.0.0 cache + `NodeBudgetBatchSampler` bucket shuffle naturally produce this — see §5).

**Try this first.** If it works, it replaces the entire CUDAGraphManager.
If it causes recompilation storms (check with `TORCH_LOGS=recompiles`), fall back
to manual bucket-based capture.

### Decision Tree for Low-β Models

```
Is β ≈ 0 and T_gpu ≈ α?  (VGAE/small, DGI/small)
  └─ Yes →  Try torch.compile(backend='cudagraphs') first
              └─ Recompilation storms? → Manual CUDAGraphManager with bucket sizes
              └─ Works? → Done. α approaches 0.
  └─ No  →  Workers + expandable_segments fix it (Priorities 1-2)
              └─ Still slow after workers scaled? → Profile β term specifically
                  └─ β large for GAT? → Triton fused attention kernel (step 5)
```

---

## 13. Triton — When and What

**Prerequisite:** Priorities 1-5 complete AND GPU is measured as still bottlenecked
after worker scaling. Do not implement before confirming with profiler.

**The justified case — GAT fused attention:**
PyG's GAT runs four separate kernels per message-passing layer:
1. Linear projection of node features: `x' = xW`  (dense matmul)
2. Attention coefficient computation: `e_ij = LeakyReLU(a^T [x'_i || x'_j])` (sparse gather)
3. Softmax over neighbors: `α_ij = softmax_j(e_ij)` (sparse reduce)
4. Weighted aggregation: `h_i = Σ_j α_ij x'_j` (sparse scatter)

Each kernel launch costs ~3-5μs. With L=3 layers and K=4 heads, that's ~48-80μs of
launch overhead per step, plus 3 separate passes over edge_index data (memory bandwidth).

A fused Triton kernel does all four operations in one pass:
- One kernel launch (not four)
- edge_index read once (not three times)
- Intermediate results stay in SRAM (not written to HBM between kernels)

Expected gain: 1.5-2x for GAT specifically. DGI and VGAE have simpler message
passing and benefit less.

**Do not implement yet.** This is documented here so future sessions don't
re-suggest it prematurely. Revisit after CUDA Graphs are validated and
GAT is confirmed as the remaining bottleneck via nsys/ncu profiling.

---

## 14. Files Referenced

**Pipeline (scope of this doc):**

| File | Purpose |
|---|---|
| `docs/reference/data-flow.md` | Full pipeline architecture ground truth |
| `docs/reference/throughput-model.md` | Cost model, probe values, sizing chain, literature |
| `graphids/core/preprocessing/budget.py` | VRAM probe, γ/α/β calibration, `ResourceProfile`, `compute_resource_profile()` |
| `graphids/core/preprocessing/features.py` | Polars lazy preprocessing, `sliding_window_graphs`, zero-copy bulk-tensor collation |
| `graphids/core/preprocessing/datamodule.py` | `_build_loader` (line 198), `CANBusDataModule`, `make_graph_loader`, `PrefetchLoader`, DynamicBatchSampler wiring |
| `graphids/core/preprocessing/datasets/can_bus.py` | `CANBusDataset` + `torch.load(..., mmap=True)` (line 90-92) |
| `graphids/core/preprocessing/stages/curriculum.py` | `CurriculumSampler` + `CurriculumDataModule`; now uses `NodeBudgetBatchSampler` with the `indices=` mapping (curriculum-style subset-to-full translation). **Fixes latent bug** where PyG's `DynamicBatchSampler` wrapped in `Subset` yielded subset-local positions that got misinterpreted as full-dataset positions by `DataLoader` — the curriculum filter was not actually reaching the model. Verified via `DynamicBatchSampler(Subset(dataset, [0,3,5,7,9]))` yielding `[0,1,2,3,4]`. |
| `graphids/core/models/_training.py` | `try_compile()` wrapper, `KDAuxiliary` TypedDict, `eval_mode()` |
| `scripts/slurm/_preamble.sh` | SLURM job setup, data staging, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `graphids/commands/profile_budget.py` | `python -m graphids probe-budget` (multi-point calibration CSV) |
| `graphids/plots/{budget,transforms}.py` | Altair/Polars budget cost-model plots |

**Orchestration (out of scope — see §0):**

| File | Purpose |
|---|---|
| `graphids/orchestrate/resolve.py` | `ConfigResolver` (exclusive merge path, cross-field validation, audit trail) |
| `graphids/orchestrate/{assets,component,slurm}.py` | Dagster asset defs + SLURM submission |
| `graphids/core/contracts/run_record.py` | `RunRecord` Pydantic schema |
| `graphids/core/models/_lightning.py` | `RunRecordCallback` (writes sidecar on fit_start/end/exception) |
| `graphids/commands/rebuild_catalog.py` | DuckDB catalog rebuild from sidecars |
