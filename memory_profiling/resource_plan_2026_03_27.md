# Resource Plan — Data-Driven Profiles (2026-03-27)

> Derived from: sacct history (March 2026), investigation.md, investigation_v2.md,
> scenario.md, walkthrough.md, PLAN.md GPU profile, project memory notes.

## Job failure summary (March 2026)

| State | Count | % of non-cancelled |
|---|---|---|
| COMPLETED | 1,116 | 49% |
| FAILED | 449 | 20% |
| OUT_OF_MEMORY | 103 | 5% |
| TIMEOUT | 12 | <1% |
| CANCELLED | 421 | (manual, excluded) |

### Root causes (three categories cover 90% of failures)

**1. Submitit orchestrator failures: 340 jobs, 7% success rate**

All from the short-lived submitit orchestration attempt (job range 45943xxx–45973xxx).
Undersized resources (4G/12G/16G when training needs 28-48G), short time limits (5-30 min
when training takes 1-4 hrs), and code bugs during the migration. This entire category
is gone — we're not using submitit.

**2. OOM: 103 jobs, two patterns**

- **CPU OOM on cache/preprocessing** (18 jobs): requested 48G, actual RSS 47-56G.
  Needs 64G+ with headroom.
- **GPU-side worker memory bloat** (8 jobs): `kd-gat-smoke` at 20-48G. Spawn workers
  pickle full dataset tensors (5.9G/worker on set_02). OOM'd at 24G with 2 workers
  (job 45984063).

**3. Code iteration bugs: ~100 jobs**

`kd-gat-landscape` (11 fail → 6 success), `kd-gat-tune` (14 fail → 2 success).
Normal R&D — fix code, resubmit. Not a resource problem.

## The performance model

From investigation_v2.md (validated against measured data):

```
GPU_util = min(1.0, num_workers × T_gpu / T_collate)
RAM      = M_base + num_workers × D + overhead

where:
  T_collate ≈ 25ms  (PyG DynamicBatchSampler + standard collation, post-FastCollate deletion)
  T_gpu     ≈ 10ms  (VGAE) / 25ms (GAT)
  M_base    ≈ 15G   (PyTorch + CUDA context + mmap'd dataset)
  D         ≈ dataset on-disk size (5.9G for set_02)
  overhead  ≈ 4G    (DynamicBatchSampler cache, Python/OS)
```

### Validation against measured data

```
Old collate (T_c=70ms): predicted 29% @ 2 workers → measured 30%  ✓
New collate (T_c=25ms): predicted 80% @ 2 workers → measured 82%  ✓
```

### Current state of collation (2026-03-27)

The custom `_FastCollate` / `_SlicesBatchSampler` / `_IndexDataset` were deleted in the
2026-03-25 cleanup. Current stack:
- PyG `DynamicBatchSampler` + standard PyG `DataLoader` via `make_graph_loader`
- `PrefetchLoader` wrapping train/val loaders for async GPU transfer
- `dataset._data_list = None` after sampler init to prevent DBS bloat

**CONFIRMED: T_c has regressed to ~70ms.** `_FastCollate` was deleted in commit `7ece283`
("Replace custom DataLoader/collation/assembly with PyG APIs"). The current stack uses
`DynamicBatchSampler` for batch composition + PyG's standard `Batch.from_data_list()` for
collation. `Batch.from_data_list()` was measured at T_c≈70ms in the original investigation.

Additionally, `PrefetchLoader` was also removed (commit `b73ae3d` — conflicted with
Lightning device management). So we lost both speedups: fast collation AND async GPU
transfer.

**Impact on resource profiles:** The profiles below assume T_c=70ms (the current state).
VGAE needs 7 workers to saturate (infeasible) — best practical is 4 workers at 57% util.
GAT needs 3 workers at 100%. See investigation section for paths forward.

## Resource profiles

### GPU training — worker/memory tradeoff (T_c=70ms, current state)

| Model | T_gpu | Workers to saturate V100 | RAM at saturation (set_02) | Practical workers | Practical GPU util |
|---|---|---|---|---|---|
| VGAE | 10ms | 7 (infeasible) | 57G | 4 | 57% |
| GAT | 25ms | 3 | 33G | 3 | 100% |

7 workers for VGAE is infeasible on Pitzer (needs 8 CPUs, 57G RAM). Best practical
is 4 workers at 57% util. **Restoring fast collation (T_c→25ms) would let 3 workers
saturate VGAE — see investigation section.**

### Concrete profiles (for resources.yaml)

```yaml
resource_profiles:
  # GPU training
  # T_c=70ms (current: PyG Batch.from_data_list standard collation)
  # RAM formula: 15G + num_workers × dataset_size + 4G overhead
  # Workers: VGAE=4 (57% util), GAT=3 (100% util)

  vgae:
    medium:
      autoencoder:
        partition: gpu
        gres: "gpu:1"
        time: "03:00:00"        # 57% util → ~2× wall time vs saturated
        mem: "43G"              # 15 + 4×5.9 + 4 ≈ 43G
        cpus_per_task: 5        # 4 workers + 1 main
        num_workers: 4
      curriculum:
        partition: gpu
        gres: "gpu:1"
        time: "03:00:00"
        mem: "43G"
        cpus_per_task: 5
        num_workers: 4
    large:
      autoencoder:
        partition: gpu
        gres: "gpu:1"
        time: "05:00:00"
        mem: "54G"              # 15 + 4×8 + 4 (larger dataset footprint)
        cpus_per_task: 5
        num_workers: 4

  gat:
    medium:
      normal:
        partition: gpu
        gres: "gpu:1"
        time: "03:00:00"        # measured 2:11
        mem: "37G"              # 15 + 3×5.9 + 4 ≈ 37G
        cpus_per_task: 4        # 3 workers + 1 main
        num_workers: 3          # saturates at T_c=70ms (3×25/70 = 107% → 100%)
      curriculum:
        partition: gpu
        gres: "gpu:1"
        time: "03:00:00"
        mem: "37G"
        cpus_per_task: 4
        num_workers: 3

  dqn:
    medium:
      fusion:
        partition: gpu
        gres: "gpu:1"
        time: "01:00:00"
        mem: "16G"              # flat TensorDataset, no PyG DataLoader
        cpus_per_task: 2
        num_workers: 0

  bandit:
    medium:
      fusion:
        partition: gpu
        gres: "gpu:1"
        time: "01:00:00"
        mem: "16G"
        cpus_per_task: 2
        num_workers: 0

  # CPU preprocessing
  preprocess:
    any:
      preprocess:
        partition: cpu
        time: "02:00:00"        # set_02 took 1:07
        mem: "72G"              # measured 56G peak, 1.3× headroom
        cpus_per_task: 8

  # CPU tests
  test:
    any:
      test:
        partition: cpu
        time: "00:30:00"
        mem: "16G"
        cpus_per_task: 8

failure_reactions:
  OUT_OF_MEMORY:
    scale_mem: 1.4              # +40%: adds ~12G (2 workers' worth)
    max_retries: 2
  TIMEOUT:
    scale_time: 1.5
    max_retries: 1
  NODE_FAIL:
    max_retries: 2
```

### If fast collation is restored (T_c→25ms)

```yaml
  # Updated profiles — VGAE saturates with fewer workers, less RAM
  vgae:
    medium:
      autoencoder:
        time: "02:00:00"        # 100% util → 1.5× wall time headroom
        mem: "37G"              # 15 + 3×5.9 + 4
        cpus_per_task: 4        # 3 workers + main
        num_workers: 3          # saturates (3×10/25 = 120% → 100%)

  gat:
    medium:
      normal:
        mem: "28G"              # 15 + 2×5.9 + 4; 2 workers saturate
        cpus_per_task: 3
        num_workers: 2          # (2×25/25 = 200% → 100%)
```

### Dataset-specific memory scaling

Not all datasets are equal. Config generator should read `cache_metadata.json` for
actual dataset size and apply the RAM formula:

| Dataset | Size on disk | RAM (2 workers) | RAM (3 workers) |
|---|---|---|---|
| hcrl_ch | ~0.3G | 20G | 20G |
| hcrl_sa | ~0.5G | 21G | 21G |
| set_01 | ~3.0G | 25G | 28G |
| set_02 | ~5.9G | 31G | 37G |
| set_03 | ~5.0G | 29G | 34G |
| set_04 | ~5.0G | 29G | 34G |

For small datasets (hcrl_*), fewer workers are fine and 20G mem suffices.
For large datasets (set_01–04), the profile above applies.

## num_workers wiring

The resource profile knows the right `num_workers`. The config generator writes it into
the stage YAML so that both sbatch args AND DataModule config come from the same source:

```yaml
# Stage YAML — num_workers comes from resource profile
data:
  init_args:
    num_workers: 3   # ← from resource_profiles[vgae][medium][autoencoder].num_workers
```

Single source of truth: resource profile drives sbatch mem/cpus AND DataLoader workers.

## CurriculumDataModule worker restart overhead

From project memory (research_spawn_mmap_hpc.md):

> `CurriculumDataModule` rebuilds DataLoader every epoch → kills persistent workers,
> triggers re-import (3-5s per worker spawn). 300 epochs × 3 workers × 4s = **60 min
> of pure spawn overhead**.

**Fix:** Replace DataLoader rebuild with sampler replacement. `CurriculumSampler.set_epoch()`
already updates indices — the DataLoader should be created once in `setup()` with
`persistent_workers=True`. Only the sampler's internal state changes.

This is a ~10-line fix in `curriculum.py` and is already tracked in the preprocessing
consolidation plan.

## Collation performance: confirmed regression

### Timeline

| Date | Commit | T_collate | GPU util (2w) | Collation method |
|---|---|---|---|---|
| pre-03-25 | `527857b` | ~70ms | 30% | `Batch.from_data_list()` |
| 03-25 spike | `527857b` | ~25ms | 82% | `_FastCollate` (tensor slicing) |
| 03-25 cleanup | `7ece283` | ~70ms | ~30% | `Batch.from_data_list()` — **FastCollate deleted** |
| 03-25 | `b73ae3d` | — | — | `PrefetchLoader` also removed (Lightning conflict) |

**Two speedups were added and then both removed in the same day.**

### What `_FastCollate` did (190 lines, deleted in `7ece283`)

Instead of `Batch.from_data_list()` (walks every graph, concatenates, recomputes
offsets, builds batch vector), `_FastCollate` sliced pre-concatenated tensors
from PyG's `InMemoryDataset._data` store:

```python
# Standard PyG (T_c≈70ms for 9800 graphs):
data_list = [dataset[i] for i in indices]  # separate() per graph
batch = Batch.from_data_list(data_list)     # re-concatenate everything

# _FastCollate (T_c≈25ms):
# Skip separate→re-collate. Slice directly from the already-concatenated store.
slices = dataset.slices
x = dataset._data.x[node_start:node_end]   # single tensor slice
edge_index = dataset._data.edge_index[..., edge_start:edge_end]  # etc.
batch = _compute_batch_vector(slices, indices)
```

2.8× speedup because it avoids:
- 9800 `separate()` calls (decompose concatenated store → individual Data objects)
- 9800 Python Data object allocations
- `Batch.from_data_list()` re-concatenation (undo then redo)

### Why it was deleted

Commit `7ece283` message: "Replace custom DataLoader/collation/assembly with PyG APIs".
The intent was correct (reduce custom code), but the performance impact was severe.
The profiling data shows this took GPU util from 82% back to ~30% on the same hardware.

### What `PrefetchLoader` did (also deleted)

Wrapped train/val DataLoaders to overlap CPU→GPU transfer with compute. Removed in
`b73ae3d` because it moved data to GPU before Lightning moved the model, causing
"index on cuda, weights on cpu" crashes.

Lightning handles `pin_memory=True` + `non_blocking=True` internally, which provides
some overlap but is not equivalent to a dedicated prefetch wrapper.

## Three problems, three solutions

### Problem 1: Collation speed (T_c=70ms → should be 25ms)

**Solution: restore fast collation as a `collate_fn` for `make_graph_loader`.**

The original `_FastCollate` was a class with `__call__` — it can be passed as
`collate_fn` to PyG's DataLoader. This is cleaner than the original approach
(which was a custom batch sampler + dataset wrapper). The key insight from the
spike (`tests/spikes/spike_fast_collate.py`):

```python
def fast_collate_fn(dataset):
    """Return a collate_fn that slices from InMemoryDataset._data directly."""
    data = dataset._data
    slices = dataset.slices

    def collate(indices):
        # Slice pre-concatenated tensors instead of separate→re-collate
        ...
        return batch

    return collate
```

Pass to `make_graph_loader`:
```python
if isinstance(dataset, InMemoryDataset) and hasattr(dataset, 'slices'):
    loader = PyGDataLoader(dataset, batch_sampler=sampler,
                           collate_fn=fast_collate_fn(dataset), **common)
```

**Impact:** T_c drops from 70ms→25ms. VGAE saturates at 3 workers instead of 7.
RAM drops from 43G→37G. Wall time halved.

**Lines:** ~60 (the collation logic is simple — the original had 190 lines because
it also included `_SlicesBatchSampler` and `_IndexDataset` which are no longer needed
since `DynamicBatchSampler` handles batch composition).

### Problem 2: Worker memory bloat (5.9G per worker)

Each spawn worker receives the full dataset via pickle. On set_02 with 4 workers:
`15G + 4×5.9G + 4G = 43G`. This is the dominant RAM cost.

**Solution options (ranked by complexity):**

**(a) Main-process collation + prefetch thread (0 extra workers, ~20G total)**

If fast collation runs at 25ms and we only need 80% GPU util, a single thread in
the main process can produce batches fast enough for GAT. For VGAE (T_g=10ms),
we'd need the thread to stay ahead — GIL is released during tensor ops so this
might work.

```python
class PrefetchThread:
    """Background thread that runs fast_collate and pins results."""
    def __init__(self, dataset, sampler, prefetch_count=4):
        self._queue = queue.Queue(maxsize=prefetch_count)
        self._collate = fast_collate_fn(dataset)
        # Thread produces batches, training loop consumes
```

No spawn workers → no pickle copies → no memory bloat. But needs spike to
verify GIL doesn't kill throughput.

**(b) Workers mmap .pt file directly (same workers, ~20G total)**

Instead of pickling the full dataset to workers, have workers mmap the same .pt
file independently. Each worker creates its own view into the memory-mapped data.
OS page cache handles dedup — no extra copies.

Requires changing how `InMemoryDataset` initializes in workers. More complex
than (a) but parallelizes collation across CPUs.

**(c) Pre-batched dataset (offline, 0 collation at train time)**

Run `DynamicBatchSampler` offline, save pre-collated batches as individual .pt
files. Workers just `torch.load(batch_N.pt)`. Zero collation cost.

Downside: batches are fixed per epoch (no shuffle within batch). Curriculum
learning needs re-batching per epoch. Only viable for non-curriculum stages.

### Problem 3: CurriculumDataModule rebuilds DataLoader every epoch

From project memory: rebuilds DataLoader → kills persistent workers → 3-5s
spawn per worker per epoch. 300 epochs × 4 workers × 4s = **80 min overhead**.

**Solution:** Create DataLoader once with `persistent_workers=True`. Update
`CurriculumSampler` internal state via `set_epoch()` — the sampler already
supports this. The DataLoader doesn't need rebuilding, only the sampler's
active index set changes.

~10-line fix in `curriculum.py`. Already tracked in preprocessing plan.

## Recommended execution order

1. **Re-profile current state** — submit 5-epoch VGAE job on set_02 with 2 workers,
   confirm GPU util is ~30% (validating the regression)
2. **Restore fast collation** as a `collate_fn` (~60 lines) — confirm GPU util returns
   to ~80%
3. **Fix CurriculumDataModule** worker restart (~10 lines)
4. **Spike: main-process prefetch thread** — if it works, eliminates memory bloat entirely
5. **Update resource profiles** based on re-profiling results
6. **If prefetch thread doesn't work**, fall back to the worker-based profiles above

### Profiling command (step 1)

```bash
sbatch --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=43G \
  --cpus-per-task=5 --account=PAS1266 --signal=B:USR1@60 \
  --output=slurm_logs/profile_collate_%j.out \
  --wrap="source scripts/slurm/_preamble.sh && \
    srun python -m graphids fit \
      --model graphids.core.models.vgae.VGAEModule \
      --data graphids.core.preprocessing.datamodule.CANBusDataModule \
      --data.dataset=set_02 --data.num_workers=2 \
      --trainer.max_epochs=5 --trainer.accelerator=gpu --trainer.devices=1 \
      --trainer.callbacks+=pytorch_lightning.callbacks.DeviceStatsMonitor"
```

Check GPU util from DeviceStatsMonitor CSV log or `nvidia-smi` sampling in the output.
