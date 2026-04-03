# Training Efficiency — GPU-First Sizing Chain

> Supersedes previous "cap the batch" approach (2026-04-02).
> Workers bumped to 6, SLURM CPUs to 8, cluster mem limits validated (2026-04-02).

## Problem

GPU utilization is 5-22% across all models. The budget system sizes batches to
fill VRAM, then the CPU can't collate fast enough to keep the GPU fed. Previous
approach was to shrink batches — wrong. That reduces GPU throughput per step and
adds per-step overhead. The correct approach: maximize batch size for GPU
efficiency, then scale the CPU pipeline to keep up.

## The Sizing Chain

Every model's resource profile is determined by a single chain. Each step feeds
the next. The budget system already measures steps 1-3 — it just stops there
instead of computing 4-6.

```
┌─────────────────────────────────────────────────────────────┐
│ 1. VRAM CAPACITY                                            │
│    Input:  GPU VRAM (16 GB V100), safety margin (0.85)      │
│    Output: usable_vram = 16 GB × 0.85 = 13.6 GB            │
│                                                             │
│    Note: use ACTIVE peak, not RESERVED. The caching         │
│    allocator's reserved pool is torch.compile overhead,     │
│    not recurring per-step demand. Current probe             │
│    underestimates by 2-3x because it sees post-compile      │
│    free VRAM instead of actual per-step activation cost.    │
├─────────────────────────────────────────────────────────────┤
│ 2. BATCH SIZE                                               │
│    Input:  usable_vram, bytes_per_node (from probe),        │
│            backward_multiplier (measured), bytes_per_edge    │
│    Formula:                                                 │
│      effective_bpn = bytes_per_node × backward_multiplier   │
│                    + bytes_per_edge × edges_per_node_p95    │
│      max_nodes = usable_vram / effective_bpn                │
│    Output: node_budget (target ~80% active VRAM util)       │
│                                                             │
│    Current bug: probe measures free VRAM after              │
│    torch.compile inflates the reserved pool. VGAE gets      │
│    154K nodes (30% active) instead of ~400K (80% active).   │
│    Fix: probe before compile, or use expandable_segments.   │
├─────────────────────────────────────────────────────────────┤
│ 3. PER-STEP TIMING                                          │
│    Input:  node_budget, gamma (collation μs/graph),         │
│            alpha (GPU base ms), beta (GPU μs/node)          │
│    Formula:                                                 │
│      graphs_per_batch = node_budget / mean_nodes_per_graph  │
│      T_collation = gamma × graphs_per_batch                 │
│      T_gpu = alpha + beta × node_budget                     │
│    Output: T_collation, T_gpu                               │
│                                                             │
│    All measurements already exist in BudgetResult           │
│    (gamma, alpha, beta from probe-budget).                  │
├─────────────────────────────────────────────────────────────┤
│ 4. WORKERS                                                  │
│    Input:  T_collation, T_gpu                               │
│    Formula:                                                 │
│      workers = ceil(T_collation / T_gpu)                    │
│    Output: num_workers                                      │
│                                                             │
│    This ensures a new batch is ready every T_gpu ms.        │
│    With W workers collating in parallel, a batch completes  │
│    every T_collation/W ms. Setting W = ceil(T_c/T_gpu)     │
│    guarantees T_c/W <= T_gpu → GPU never starves.           │
├─────────────────────────────────────────────────────────────┤
│ 5. PREFETCH                                                 │
│    Input:  workers, T_collation variance                    │
│    Formula:                                                 │
│      prefetch_factor = 2-4 (absorbs collation variance)     │
│    Output: prefetch_factor                                  │
│                                                             │
│    Prefetch doesn't change throughput — it smooths jitter.  │
│    Without prefetch buffer, a slow batch (outlier graph      │
│    sizes) stalls the GPU. With prefetch_factor=4, there are │
│    workers × 4 batches queued, absorbing variance.          │
├─────────────────────────────────────────────────────────────┤
│ 6. SLURM RESOURCES                                          │
│    Input:  workers, per-worker RSS (measured or estimated)  │
│    Formula:                                                 │
│      cpus = workers + 2  (main process + headroom)          │
│      memory = workers × worker_rss + base_rss + headroom    │
│    Output: --cpus-per-task, --mem for SLURM profile         │
│                                                             │
│    CPU memory is an order of magnitude cheaper than GPU      │
│    time. A 128 GB RAM request that keeps the GPU fed is     │
│    better than a 36 GB request where the GPU idles 90%.     │
└─────────────────────────────────────────────────────────────┘
```

## Projected Impact

Using ablation run measurements (`docs/reference/ablation-resource-profile.md`):

### VGAE Large (currently 3h40m, 5% GPU util)

| Step | Current | Optimized |
|------|---------|-----------|
| 1. Usable VRAM | 9.1 GB (post-compile) | 13.6 GB (pre-compile probe) |
| 2. Node budget | 154K nodes (5,471 graphs) | ~460K nodes (~16,300 graphs) |
| 3a. T_collation | 400 ms | ~1,200 ms |
| 3b. T_gpu | 25 ms | ~75 ms |
| 4. Workers | 2 (hardcoded) | ceil(1200/75) = **16** |
| 5. Prefetch | 2 | 4 |
| 6. CPUs / Memory | 6 / 48 GB | 18 / 128 GB |
| **Throughput** | **12.9K nodes/sec** | **~219K nodes/sec (17x)** |
| **Wall time** | **3h40m** | **~13 min** |

### GAT Large (currently 3h55m, 22% GPU util)

| Step | Current | Optimized |
|------|---------|-----------|
| 1. Usable VRAM | 12.6 GB (post-compile) | 13.6 GB |
| 2. Node budget | 36K nodes (1,289 graphs) | ~42K nodes (~1,500 graphs) |
| 3a. T_collation | 52 ms | ~61 ms |
| 3b. T_gpu | 25 ms | ~29 ms |
| 4. Workers | 2 | ceil(61/29) = **3** |
| 5. Prefetch | 2 | 2 |
| 6. CPUs / Memory | 4 / 36 GB | 5 / 36 GB |
| **Throughput** | **~328K nodes/sec** | **~483K nodes/sec (1.5x)** |
| **Wall time** | **3h55m** | **~2h37m** |

GAT is already closer to optimal — the 2:1 ratio means 3 workers nearly
saturate it. VGAE is where the 17x gain lives.

### Curriculum

Same architecture as GAT but CurriculumDataModule rebuilds the DataLoader
every epoch (re-sorts by difficulty). Worker startup cost (~3-5s per epoch
for torch+PyG import with `spawn`) adds ~7-15 min over 300 epochs.
`persistent_workers=True` doesn't help because the DataLoader is recreated.

Fix: cache the difficulty sort and reuse the DataLoader, only rebuilding
the sampler order. Separate backlog item.

## Implementation

### Phase 1: Fix the probe (biggest single win)

The probe currently runs `torch.compile`, then measures free VRAM. This
underestimates usable VRAM by the compile pool inflation (4-8 GB).

Options:
- **A)** Probe before compile, store result, compile later. Requires
  restructuring the budget→compile order in `_build_loader`.
- **B)** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so the
  allocator returns unused segments. The reserved pool shrinks to match
  actual demand. Simplest change (env var in `_preamble.sh`).
- **C)** Measure `active_bytes.peak` from the first few training steps
  (ResourceProfileCallback) and re-adjust the budget mid-epoch. Complex.

**Recommendation: B first** (one line in `_preamble.sh`), validate with
a smoke test, then decide if A is needed.

### Phase 2: Compute full resource profile from budget

Extend `node_budget()` or add `compute_resource_profile()` that returns:

```python
@dataclass
class ResourceProfile:
    node_budget: int           # from step 2
    graphs_per_batch: int      # node_budget / mean_nodes
    t_collation_ms: float      # gamma × graphs_per_batch
    t_gpu_ms: float            # alpha + beta × node_budget
    workers: int               # ceil(t_collation / t_gpu)
    prefetch_factor: int       # 2-4
    cpus: int                  # workers + 2
    memory_gb: int             # workers × worker_rss + base
```

The budget system already has gamma, alpha, beta, bytes_per_node,
backward_multiplier, edges_per_node_p95. This is arithmetic on existing
measurements.

### Phase 3: Wire into DataModule + resource profiles

- `CANBusDataModule.__init__` gets `num_workers: int | None = None` and
  `prefetch_factor: int = 2`. When `None`, auto-compute from
  `ResourceProfile`.
- Resource profile YAMLs (`config/resources/profiles/*.yaml`) get
  `cpus` and `mem` updated per model from the chain.
- `scripts/submit.sh` reads the profile as it already does.

### Phase 4: Validate

Run VGAE large + GAT large on hcrl_sa (small, fast) with optimized
profiles. Compare:
- samples/sec (should be 10-17x for VGAE, 1.5x for GAT)
- Peak active VRAM (should be ~80% of 16 GB)
- No OOMs
- Training loss curve matches baseline (same hyperparams, just faster)

If convergence differs with larger batches, increase epochs or adjust
learning rate (linear scaling rule: `lr *= batch_scale_factor`).

### Phase 5: Rewrite plot_budget.py

`scripts/plot_budget.py` was built for the old "cap the batch" framing.
Every plot assumes fixed `W=6` and shows artifacts of that assumption
(throughput plateaus, throughput floors). Rewrite to reflect the sizing chain.

**Fixes to existing plots:**

1. **`_throughput_floor()` → `_required_workers()`** — replace the floor
   function (minimum batch to amortize overhead) with a function that
   returns `ceil(T_c / T_gpu)` for a given batch size. The floor concept
   is dead.

2. **`plot_throughput_curves`** — currently shows one curve at fixed W.
   Show three curves: (a) current W=2, (b) fixed W=6, (c) auto W where
   `W = ceil(T_c / T_gpu)` per batch size. The gap between (a) and (c)
   IS the speedup. Annotate VRAM ceiling with required workers at that point.

3. **`plot_regime_map`** — keep the heatmap but add a **rightmost column**
   showing the required W to reach cg_ratio ≤ 1 for each model. This is
   the answer the plot should give: "how many workers do I need?"

4. **`plot_budget_comparison`** — remove throughput floor markers (▼).
   Replace with worker count annotations on each bar: "W=3" for GAT,
   "W=16" for VGAE, etc. The bars (VRAM budget) stay — they're the batch.

5. **`plot_deep_dive`** — GPU utilization panel (bottom) should show TWO
   fills: current W (e.g., W=2, mostly pink/idle) and optimal W (mostly
   green/active). The visual gap between them is the wasted GPU time.
   Top panel: show throughput at current W vs optimal W.

**New plots:**

6. **Workers sweep** — for each model at VRAM-ceiling batch, plot
   throughput vs num_workers (1 to 32). Shows where throughput plateaus
   (= GPU-bound, mission accomplished). One subplot per model, overlay
   the VRAM-ceiling operating point.

7. **Sizing chain summary** — horizontal waterfall or table-as-figure
   showing the full chain per model: VRAM → batch → T_gpu → T_c →
   workers → CPUs → memory. One row per model×scale. This is the
   "answer sheet" plot.

8. **Prefetch jitter tolerance** — for a given model at optimal W, show
   step time distribution with prefetch_factor=1,2,4. X-axis = step time,
   Y-axis = density. Without prefetch, tail batches (slow collation from
   large graphs) stall the GPU. With prefetch=4, the buffer absorbs them.
   Use gamma variance from cache_metadata.json (graph size std dev) to
   model the distribution.

**Prefetch in the throughput model:**

The current `_throughput_model` is deterministic — all batches identical.
Add a stochastic mode that models collation time as
`T_c ~ Normal(gamma × B, sigma × B)` where sigma comes from graph size
variance. Without prefetch buffer, the effective step time is
`E[max(T_c/W, T_gpu)]` which is higher than `max(E[T_c]/W, T_gpu)` due
to Jensen's inequality. With prefetch buffer depth `D = W × prefetch_factor`,
the GPU draws from a queue of D batches, smoothing the variance.

Approximate: `effective_step = max(E[T_c]/W, T_gpu) × (1 + cv² / (2 × D))`
where cv = sigma/mean collation time. For CAN bus (low cv ≈ 0.1-0.15),
prefetch=2 adds <1% overhead. For high-variance datasets, prefetch=4 matters.

Add `prefetch_factor` and `cv_collation` to `ResourceProfile`:

```python
@dataclass
class ResourceProfile:
    node_budget: int
    graphs_per_batch: int
    t_collation_ms: float
    t_gpu_ms: float
    workers: int               # ceil(t_collation / t_gpu)
    prefetch_factor: int       # ceil(cv_collation × 2), min 2
    cv_collation: float        # from cache_metadata.json graph size stats
    cpus: int                  # workers + 2
    memory_gb: int             # workers × worker_rss + prefetch_buffer + base
    prefetch_buffer_mb: float  # workers × prefetch_factor × batch_bytes
```

**CLI changes:**

- `--workers` default changes from `6` to `auto` (compute from chain).
  Keep `--workers N` for fixed-W comparison plots.
- Add `--prefetch` flag (default `auto`, or fixed integer).
- Add `--sweep-workers` flag to generate the workers sweep plot.

**Files:**

- `scripts/plot_budget.py` — rewrite all 4 existing plot functions + add 3 new
- `graphids/core/preprocessing/budget.py` — add `cv_collation` to probe
  (read from `cache_metadata.json` graph size stats, already available)

## What NOT to change

- Fusion stays as-is (no collation — cached state vectors, `num_workers=0`).
- Don't reduce safety margin below 0.80 (OOM headroom for batch variance).
- Don't add graph-size bucketing (incremental polish after the chain works).

## Execution order

| Step | What | Effort | Impact |
|------|------|--------|--------|
| 1 | `expandable_segments:True` in preamble | 1 line | Unlocks accurate VRAM probe |
| 2 | `compute_resource_profile()` in budget.py | ~40 lines | Computes full chain incl. prefetch |
| 3 | Wire num_workers/prefetch into DataModule | ~10 lines | Enables auto-sizing |
| 4 | Update resource profile YAMLs | YAML only | SLURM gets right CPUs/mem |
| 5 | Rewrite plot_budget.py | ~300 lines | Correct visualizations |
| 6 | Validate on hcrl_sa | 1 smoke test | Confirms 10-17x speedup |
