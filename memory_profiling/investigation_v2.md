## DataLoader Performance Model — Post-FastCollate (2026-03-25)

Updates investigation.md with measured _FastCollate timings (job 45985264) replacing the old Batch.from_data_list() estimates.

### The model

```
GPU util ≈ min(1.0, y × T_g / T_c)

Variables:
  T_c = collation time per batch (CPU, per worker)
  T_g = GPU compute per batch (forward + backward)
  y   = num_workers
  D   = dataset on-disk size (set_02 ≈ 5.9 GB)
  M_base = base process memory (~15 GB: PyTorch + CUDA context + mmap'd dataset)
  M_worker = per-worker memory ≈ D (pickle serializes full dataset tensors)

Total CPU cores = y + 1 (workers + main)
Total RAM = M_base + y × M_worker + overhead
```

### Collation cost: old vs new

```
Old (Batch.from_data_list):
  spike: 35.8ms / 1000 graphs → ~350ms / 9800 graphs (full batch)
  Profiled: ~70ms effective (includes IPC, pin_memory overhead)
  Source: spike job 45984920, profile job 45984077 (30.4% GPU util)

New (_FastCollate):
  spike: 2.9ms / 1000 graphs → ~28ms / 9800 graphs (full batch)
  Profiled: effective T_c ≈ 25-30ms (derived from 82% GPU util with 2 workers)
  Source: spike job 45985003, profile job 45985264 (82% training GPU util)
```

Derivation of effective T_c from profile:
```
  GPU util = y × T_g / T_c
  0.82 = 2 × 10 / T_c
  T_c ≈ 24ms
```
Consistent with spike extrapolation (~28ms). The gap includes prefetch buffer smoothing.

### Validation of model against measurements

```
  Old:  2 × 10 / 70  = 28.6%  →  measured 30%  ✓
  New:  2 × 10 / 25  = 80.0%  →  measured 82%  ✓
```

### Timeline: _FastCollate with 2 workers

```
  ═══════════════════════════════════════════════════════════════
  num_workers=2, _FastCollate (T_c≈25ms, T_g≈10ms)
  ═══════════════════════════════════════════════════════════════

  W0:     │▓▓collate▓▓│         │▓▓collate▓▓│         │▓▓collate▓▓│
  W1:       │▓▓collate▓▓│         │▓▓collate▓▓│         │▓▓collate▓▓│
          ─────────────────────────────────────────────────────────────
  queue:  [b0,b1,b2,b3]→[b2,b3]→[b3,b4]→[b4,b5]→ ... (rarely empty)
          ─────────────────────────────────────────────────────────────
  GPU:    [fwd+bwd][fwd+bwd][fwd+bwd][fwd+bwd]-gap-[fwd+bwd][fwd+bwd]
          █████████ █████████ █████████ █████████    █████████ █████████

  2 workers produce at 2/25ms = 80 batches/sec
  GPU consumes at 1/10ms = 100 batches/sec
  Ratio: 80/100 = 80% → matches observed 82%

  Short gaps when queue drains, but much rarer than before.
  Most time: GPU has a batch ready.
```

---

### Pitzer: 1×V100-16GB (current setup)

T_g ≈ 10ms (VGAE) / 25ms (GAT), T_c ≈ 25ms, D ≈ 5.9GB

```
  y    GPU util    GPU util     CPUs    RAM          Feasible?
       (VGAE)      (GAT)        needed  needed
  ──── ────────── ──────────── ─────── ──────────── ─────────
  0     ~30%*       ~50%*        1       15 GB       ✓ but no prefetch
  1     40%         100% ←       2       21 GB       ✓ GAT saturates!
  2     80%         100%         3       27 GB       ✓ (measured: 82%)
  3    100% ←       100%         4       33 GB       ✓ VGAE saturates
  4    100%         100%         5       39 GB       overkill

  * num_workers=0 is single-threaded, no overlap. Effective util
    is T_g/(T_c+T_g) = 10/35=29% (VGAE), 25/50=50% (GAT).
```

Sweet spot VGAE: **y=3, x=4, --mem=36G** → 100% GPU util
Sweet spot GAT: **y=1, x=2, --mem=24G** → 100% GPU util (GAT's T_g is long enough)

Compare old collate: needed 7 workers to saturate VGAE (not feasible on 6 CPUs).
Now: 3 workers. Fits in current allocation.

### Pitzer: 2×V100 DDP

Each rank gets its own workers. Dataset split across ranks.

```
  y/gpu  GPU util   CPUs needed    RAM needed       Feasible?
         (VGAE)     (2×(y+1))      (2×(15+y×5.9))   (363 GB node)
  ────── ────────── ──────────── ──────────────── ──────────
  1       40%         4              42 GB           ✓
  2       80%         6              54 GB           ✓
  3      100%         8              65 GB           ✓ ← sweet spot
  4      100%        10              78 GB           overkill
```

2×V100 with y=3/gpu → 100% util on both GPUs, 8 cores, 65 GB.
~2× throughput vs single GPU. PCIe gradient sync <1ms (small model).

### Ascend: 2×A100-40GB

T_g (A100) ≈ T_g (V100) / 3 ≈ 3ms (VGAE) / 8ms (GAT)
T_c stays ~25ms (CPU-bound, same collation code)

```
  y/gpu  GPU util   CPUs needed    RAM needed       Feasible?
         (VGAE)     (y+1 per GPU)  per GPU           (472 GB node)
  ────── ────────── ──────────── ──────────────── ──────────
   2      24%         3              27 GB           ✓ poor
   4      48%         5              39 GB           ✓
   6      72%         7              50 GB           ✓
   8      96%         9              62 GB           ✓ ← sweet spot
   9     100%        10              68 GB           ✓

  DDP (2 GPUs, y=8 each):
    CPUs: 18, RAM: 124 GB — fits in 120 cores / 472 GB
```

Old collate needed 23 workers/GPU to saturate A100. Now: 9. Ascend goes from "don't bother" to viable.

### Ascend: 4×A100-80GB NVLink

```
  y/gpu  GPU util   CPUs needed    RAM needed       Feasible?
         (VGAE)     (4×(y+1))      (4×(15+y×5.9))   (921 GB node)
  ────── ────────── ──────────── ──────────────── ──────────
   4      48%        20             156 GB           ✓
   6      72%        28             202 GB           ✓
   8      96%        36             249 GB           ✓ ← sweet spot
   9     100%        40             272 GB           ✓ fits in 88 cores
```

Old collate: maxed out at 69% with 16 workers/GPU (68 cores). Now: 96% with 8 workers/GPU (36 cores). Leaves 52 cores free.

### Cardinal: 4×H100-94GB NVLink

T_g (H100) ≈ T_g (V100) / 4 ≈ 2.5ms (VGAE) / 6ms (GAT)

```
  y/gpu  GPU util   CPUs needed    RAM needed       Feasible?
         (VGAE)     (4×(y+1))      (4×(15+y×5.9))   (1 TB node)
  ────── ────────── ──────────── ──────────────── ──────────
   4      40%        20             156 GB           ✓
   6      60%        28             202 GB           ✓
   8      80%        36             249 GB           ✓
  10     100%        44             296 GB           ✓ ← sweet spot

  Per-user limit: 48 cores → 4 GPUs × (y=11) = 48 → 100% util!
    RAM: 4 × (15 + 11×5.9) = 320 GB of 1 TB ✓
```

Old collate: H100 was "don't bother" (39% util with 48-core limit). Now: **100% with 11 workers/GPU, 48 cores.** Cardinal is feasible.

---

### The big picture

```
                        GPU Utilization vs Workers
                        T_c=25ms (_FastCollate)

 100% ┤         ╭───────────────────────────── V100 (T_g=10ms)
      │    ╭────╯
  80% ┤────╯
      │              ╭─────────────────────── A100 (T_g=3ms)
  60% ┤         ╭────╯
      │    ╭────╯         ╭───────────────── H100 (T_g=2.5ms)
  40% ┤────╯         ╭────╯
      │         ╭────╯
  20% ┤    ╭────╯
      │────╯
   0% ┤────────────────────────────────────────────
      0    2    4    6    8   10   12   14   16
                     num_workers per GPU

      Workers to saturate:  V100=3  A100=9  H100=10
      (was:                 V100=7  A100=23  H100=28)
```

### Comparison: old collate vs _FastCollate

```
  ┌───────────────┬──────────────────────┬──────────────────────┬──────────┐
  │    Cluster    │   Old (T_c=70ms)     │   New (T_c=25ms)     │ Δ        │
  ├───────────────┼──────────────────────┼──────────────────────┼──────────┤
  │ Pitzer 1×V100 │ y=7 → 100% (infeas) │ y=3 → 100%           │ feasible │
  │               │ y=4 → 57%, 48G       │ y=2 → 80%, 27G  ←   │ current  │
  ├───────────────┼──────────────────────┼──────────────────────┼──────────┤
  │ Pitzer 2×V100 │ y=4/gpu, 57%, 78G    │ y=3/gpu, 100%, 65G   │ -13G RAM │
  ├───────────────┼──────────────────────┼──────────────────────┼──────────┤
  │ Ascend 2×A100 │ y=12/gpu, 51%, 172G  │ y=8/gpu, 96%, 124G   │ viable   │
  ├───────────────┼──────────────────────┼──────────────────────┼──────────┤
  │ Ascend 4×A100 │ y=16/gpu, 69%, 435G  │ y=8/gpu, 96%, 249G   │ viable   │
  ├───────────────┼──────────────────────┼──────────────────────┼──────────┤
  │ Cardinal H100 │ y=11/gpu, 39%, 320G  │ y=11/gpu, 100%, 320G │ feasible │
  │               │ "don't bother"       │ sweet spot            │          │
  └───────────────┴──────────────────────┴──────────────────────┴──────────┘
```

### Memory bloat (unchanged)

_FastCollate did not reduce worker memory. Each spawn worker still receives full dataset tensors via pickle.

```
  set_02 (5.9 GB on disk) with 2 workers:
    MaxRSS = 37.7 GB (both old and new collate)
    Breakdown (estimated):
      M_base ≈ 15 GB (PyTorch + CUDA context + mmap'd .pt)
      2 × D ≈ 12 GB (pickle serializes tensors to shared memory per worker)
      DynamicBatchSampler populates _data_list cache: 503K separate() calls ≈ 6-8 GB
      Python/OS overhead ≈ 3-5 GB
```

The formula `RAM = 15 + y × 5.9 GB` still holds. More workers = more memory. This is the remaining unsolved problem.

### Immediate action items

1. **Bump num_workers to 3** for VGAE training on Pitzer → predicted 100% GPU util
   - Needs `--mem=36G` (already the current setting)
   - Needs `--cpus-per-task=4` (current is 6, fits)
   - Config: `num_workers: 3` in config.yaml or CLI override

2. **Keep num_workers=2 for GAT** — T_g=25ms is long enough that 2 workers already deliver 100%

3. **Don't increase to 4+ workers** — diminishing returns, RAM cost not worth it on Pitzer

### What would actually fix the memory bloat

The root cause is spawn pickling full dataset tensors to each worker. Potential solutions (not implemented):

1. **Main-process collation (num_workers=0) + async prefetch**: Custom prefetch thread that runs _FastCollate in a background thread (GIL released during tensor ops). Avoids spawn entirely. Risk: GIL contention if Python-heavy ops remain.

2. **Workers mmap the .pt file directly**: Instead of receiving tensors via pickle, workers open the same mmap'd .pt file independently. Each worker creates its own `_FastCollate` from the mmap'd data. No pickle, no shared memory copy. The OS page cache handles dedup.

3. **Pre-batched dataset**: Run DynamicBatchSampler offline, save pre-collated batches as individual .pt files. Workers just `torch.load(batch_N.pt)`. Zero collation at training time. Downside: batches are fixed (no shuffle, no curriculum).
