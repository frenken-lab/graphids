# Throughput Model & Budget System

> Cost model for GNN training throughput, GPU optimization framework, and budget implementation.
> Rewritten 2026-04-03 to correct sections 2, 3, 5 (previous version wrongly advocated
> shrinking batches instead of scaling the data pipeline).

---

## 1. Cost Model (First Principles)

### 1.1 Throughput Definition

Throughput $T$ = nodes processed per unit time:

$$T = \frac{N_V^{\text{batch}}}{\Delta t_{\text{step}}}$$

where $N_V^{\text{batch}} = \sum_{i=1}^{B} |V_i|$ is total node count across $B$ graphs. Step time decomposes as:

$$\Delta t_{\text{step}} = \Delta t_{\text{collate}} + \Delta t_{\text{transfer}} + \Delta t_{\text{forward}} + \Delta t_{\text{backward}}$$

With sufficient prefetch depth and `num_workers`, collation of batch $k+1$ overlaps with compute on batch $k$:

$$\Delta t_{\text{step}} \approx \max\!\left(\frac{\Delta t_{\text{collate}}}{W},\ \Delta t_{\text{forward}} + \Delta t_{\text{backward}}\right)$$

where $W$ = number of DataLoader workers producing batches in parallel.

> **Source:** Standard pipeline overlap analysis. Used explicitly in [Tan et al., USENIX ATC 2021](https://www.usenix.org/conference/atc21/presentation/tan-ying). The $\max(\cdot)$ form assumes zero synchronization overhead (idealization).

### 1.2 Collation Cost

PyG's `Batch.from_data_list()` performs three operations per graph ([PyG source](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/batch.py)):

1. Offset `edge_index` by cumulative node count — $O(|E_i|)$
2. Concatenate node features — $O(|V_i| \cdot d_v)$
3. Concatenate edge features — $O(|E_i| \cdot d_e)$

Summing over the batch:

$$\Delta t_{\text{collate}} \propto N_V^{\text{batch}} \cdot d_v + N_E^{\text{batch}} \cdot (1 + d_e)$$

> **Source:** Derived from PyG source inspection. Proportionality constant is hardware-dependent, must be measured.

In practice, for constant edge density (CAN bus ≈ 4.5 edges/node), collation simplifies to:

$$T_{\text{collate}} = \gamma \cdot B$$

where $\gamma$ = seconds/graph (CPU work, model-independent).

### 1.3 Forward Pass Cost

For a standard MPNN ([Gilmer et al., ICML 2017](https://arxiv.org/abs/1704.01212)) with $L$ layers, hidden dim $h$:

$$\Delta t_{\text{forward}} \propto L \cdot h^2 \left(N_E^{\text{batch}} + N_V^{\text{batch}}\right)$$

For **GAT** ([Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903)) with $K$ attention heads:

$$\Delta t_{\text{forward}}^{\text{GAT}} \propto L \left(K \cdot N_E^{\text{batch}} \cdot h + N_E^{\text{batch}} \cdot h^2 + N_V^{\text{batch}} \cdot h^2\right)$$

> **Source:** MPNN/GAT formulations. $h^2$ scaling from linear projections is standard. GPU kernel efficiency for sparse scatter/gather is substantially lower than dense matmul — hardware-dependent.

In practice, GPU cost follows an affine model (kernel launch overhead + per-element work):

$$T_{\text{gpu}}(N) = \alpha + \beta \cdot N$$

where $\alpha$ = per-step overhead (seconds), $\beta$ = per-node cost (seconds/node).

### 1.4 Three Bottleneck Regimes

Every GPU workload sits in one of three regimes ([Horace He, "Making DL Go Brrr"](https://horace.io/brrr_intro.html)):

1. **Compute-bound** — GPU arithmetic units are saturated. Achieved FLOPS near
   peak. Fix: better kernels, tensor cores, mixed precision.
2. **Memory-bound** — GPU HBM bandwidth is the bottleneck. Time moving data
   between HBM and compute exceeds compute time. Fix: operator fusion, increase
   arithmetic intensity (bigger matmuls, larger batches).
3. **Overhead-bound** — everything *outside* the GPU is the bottleneck. Python
   dispatch, CUDA kernel launch overhead, data loading stalls. Fix: give the
   GPU more work (bigger batches) and deliver it faster (more workers, prefetch).

The regime is identified by measuring achieved FLOPS as a percentage of peak.
If FLOPS/peak is high, you're compute-bound. If GPU utilization is low despite
large batches, you're memory-bound. If GPU utilization is low because the GPU
is idle waiting for data, you're overhead-bound.

**Arithmetic intensity** — the ratio of FLOPs to bytes transferred — determines
the boundary between compute-bound and memory-bound. For a V100 (15.7 TFLOPS
FP32, 900 GB/s HBM bandwidth), the threshold is ~17.4 FLOPs/byte. Operations
below this are memory-bound; above are compute-bound.

GNN message passing has inherently low arithmetic intensity due to sparse
scatter/gather operations. This makes GNNs more likely to be memory-bound or
overhead-bound than dense models (transformers, CNNs).

> **Sources:**
> [NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html),
> [JAX Scaling Book — Rooflines](https://jax-ml.github.io/scaling-book/roofline/),
> [Horace He — Making DL Go Brrr](https://horace.io/brrr_intro.html)

### 1.5 Peak VRAM

$$\text{VRAM} \geq \underbrace{L \cdot N_V \cdot h \cdot s}_{\text{activations}} + \underbrace{P \cdot s}_{\text{params}} + \underbrace{2P \cdot s}_{\text{Adam } m_t, v_t} + \underbrace{P \cdot s}_{\text{gradients}} + \underbrace{N_E \cdot 8}_{\text{edge\_index}}$$

where $P$ = parameter count, $s$ = bytes/element. This is a **lower bound** — excludes CUDA allocator fragmentation, intermediate buffers, optimizer workspace.

> **Sources:** Activation retention: [Chen et al., arXiv:1604.06174](https://arxiv.org/abs/1604.06174). Adam state: [Kingma & Ba, ICLR 2015](https://arxiv.org/abs/1412.6980). `edge_index`: exact (2 rows × $N_E$ × 8 bytes).

### 1.6 What Cannot Be Stated as Equations

- **PCIe transfer time for fragmented small tensors.** Many small H2D transfers have non-trivial per-transfer latency. Measurable with nsys; not derivable analytically.
- **SM occupancy vs graph size.** Small graphs produce small kernels with poor warp utilization. Measurable with ncu; no closed-form expression.
- **Proportionality constants** in $\Delta t_{\text{collate}}$ and $\Delta t_{\text{forward}}$. Must be measured on specific hardware and graph distribution.

### 1.7 Diagnostic Protocol

```python
import time, torch

# 1. Pure collation cost (no GPU)
t0 = time.perf_counter()
for batch in loader:
    pass
t_collate = time.perf_counter() - t0

# 2. Collation + H2D transfer
t0 = time.perf_counter()
for batch in loader:
    batch = batch.to(device)
t_transfer = time.perf_counter() - t0

# 3. Forward only (no backward)
t0 = time.perf_counter()
for batch in loader:
    with torch.no_grad():
        _ = model(batch.to(device))
t_forward = time.perf_counter() - t0

# 4. Full training step
t0 = time.perf_counter()
for batch in loader:
    batch = batch.to(device)
    loss = criterion(model(batch), batch.y)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
t_full = time.perf_counter() - t0

print(f"collate:  {t_collate:.3f}s")
print(f"transfer: {t_transfer - t_collate:.3f}s (delta)")
print(f"forward:  {t_forward - t_transfer:.3f}s (delta)")
print(f"backward: {t_full - t_forward:.3f}s (delta)")
```

---

## 2. Application: Small-Graph GNN Regime

### This workload is overhead-bound

Profiled data from the set_01 ablation campaign (V100 16GB, 2 workers):

| Model | Batch (graphs) | T_collate | T_gpu | GPU active % | Regime |
|-------|----------------|-----------|-------|-------------|--------|
| VGAE large | 5,471 | 400 ms | 25 ms | 5% | Overhead-bound |
| GAT large | 1,289 | 52 ms | 25 ms | 22% | Overhead-bound |

The GPU finishes in 25ms and then idles for 175-375ms waiting for the CPU
to collate the next batch. GPU compute units and HBM bandwidth are both
unutilized. No amount of kernel fusion, mixed precision, or operator
optimization helps when the GPU has no work to do.

This is the textbook overhead-bound regime described by
[Horace He](https://horace.io/brrr_intro.html): "the easiest way to tell
if you're overhead bound is to simply increase the size of your data."

### Why these models are overhead-bound

Three factors compound:

1. **Small models, cheap compute.** VGAE has 745K params, GAT has 2.5M.
   Forward+backward on a V100 takes 25ms regardless of batch size because
   the model is too small to saturate the GPU.

2. **Expensive collation.** PyG's `Batch.from_data_list()` iterates every
   graph, concatenating variable-size tensors. For 5,471 graphs at 65μs/graph,
   that's 356ms of serial CPU work. This is O(B) and cannot be reduced without
   changing the data format.

3. **Insufficient pipeline depth.** With 2 workers, effective delivery rate
   is T_collate/2 = 200ms. GPU needs a batch every 25ms. The pipeline is
   8x too slow.

> **Established in literature:**
> - [SALIENT](https://arxiv.org/abs/2110.08450) (Kaler et al., 2021): 3x
>   speedup from pipeline optimization alone. "Only ~28% of time is GPU training."
> - [BGL](https://www.usenix.org/conference/nsdi23/presentation/liu-tianfeng)
>   (Liu et al., NSDI 2023): ~10% GPU utilization in typical DGL training.
> - [PyG #4891](https://github.com/pyg-team/pytorch_geometric/issues/4891):
>   DataLoader is 59-83% of runtime.
> - [ATC 2025](https://www.usenix.org/system/files/atc25-gong.pdf) (Gong et al.):
>   "Training runtime on smaller graphs is dominated by framework overhead."

### The correct fix: GPU-first sizing

The optimization chain starts from the GPU and works outward:

```
VRAM capacity → max batch size → T_gpu per step
                               → T_collation per step
                                   → workers = ceil(T_c / T_gpu)
                                       → CPUs, memory for SLURM
```

**Maximize batch size** to increase GPU utilization and amortize per-step
overhead (α). Larger batches increase arithmetic intensity, pushing the
workload from overhead-bound toward memory-bound or compute-bound — both
of which are more efficient regimes.

**Scale workers** to deliver batches at the rate the GPU consumes them.
CPU memory is an order of magnitude cheaper than GPU time. A 128 GB RAM
request that keeps the GPU fed is better than a 36 GB request where the
GPU idles 90%.

> **Supported by:**
> - [NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html):
>   "Larger layers achieve higher arithmetic intensity" → larger batches
>   increase arithmetic intensity for the same model.
> - [PyTorch Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html):
>   "Set num_workers > 0 to enable asynchronous data loading, tuned based
>   on workload."
> - [JAX Scaling Book](https://jax-ml.github.io/scaling-book/roofline/):
>   Batch size directly determines arithmetic intensity for matmuls.
>   "Become compute-bound when per-replica batch size exceeds ~240 tokens"
>   (for transformers; GNNs need proportionally more due to sparse ops).

### Why "shrink the batch" was wrong

A previous version of this document proposed capping the batch at a
"throughput-optimal" size smaller than VRAM capacity. The reasoning:
with fixed workers (W=6), smaller batches reduce T_collation so the
pipeline can keep up.

This is wrong because:

1. **It holds the wrong variable fixed.** Workers are a cheap, scalable
   resource. Batch size directly affects GPU efficiency. Shrinking the
   batch sacrifices GPU throughput to accommodate an undersized pipeline.

2. **Smaller batches increase per-step overhead.** Each step has fixed
   costs: kernel launches (~3μs each, dozens per step), optimizer update,
   DataLoader synchronization. More steps = more overhead. ([HiFuse, 2024](https://arxiv.org/html/2408.08490v1):
   "substantial overhead and idle time from frequent kernel launches.")

3. **Smaller batches reduce arithmetic intensity.** Less work per byte
   transferred pushes the workload deeper into the memory-bound or
   overhead-bound regime — the opposite of what you want.

4. **The literature says increase batch size when overhead-bound.**
   [NVIDIA](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html),
   [Horace He](https://horace.io/brrr_intro.html), and the
   [roofline model](https://jax-ml.github.io/scaling-book/roofline/)
   all recommend increasing batch size to move out of the overhead-bound
   regime.

### Projected impact (from ablation measurements)

| Model | Current | Optimized | Speedup |
|-------|---------|-----------|---------|
| VGAE large | 3h40m, 5% GPU, 2 workers, 154K nodes | ~13min, ~80% GPU, 16 workers, ~460K nodes | **17x** |
| GAT large | 3h55m, 22% GPU, 2 workers, 36K nodes | ~2h37m, ~50% GPU, 3 workers, ~42K nodes | **1.5x** |

See `docs/backlog/training-efficiency.md` for the full sizing chain with
derivations.

### Additional GPU optimizations (secondary to pipeline fix)

These help once the GPU is actually busy:

| Optimization | Regime it helps | Status | Action |
|-------------|----------------|--------|--------|
| Mixed precision (AMP) | Memory-bound, compute-bound | Enabled | Already done |
| torch.compile | Memory-bound (operator fusion) | Enabled | Consider disabling for VGAE (5% util, wastes 4-8GB VRAM pool) |
| Tensor core alignment | Compute-bound (FP16 dims ÷ 8) | Not audited | Audit hidden dims |
| CUDA Graphs | Overhead-bound (kernel launch) | Not used | Future: capture training step as graph |
| expandable_segments | Overhead-bound (VRAM pool) | Not set | Enable in _preamble.sh |
| pin_memory | Overhead-bound (H2D transfer) | Verify enabled | Check DataModule |
| Gradient accumulation | Memory-bound (simulate larger batch) | Not needed | VRAM headroom exists |

> **Sources:**
> [PyTorch Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
> [NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html),
> [HiFuse — CUDA kernel reduction](https://arxiv.org/html/2408.08490v1),
> [CUDA Graphs for kernel batching](https://arxiv.org/html/2501.09398v1)

### Self-tuning properties

| What changes | Effect on sizing chain |
|---|---|
| Larger model (GAT vs VGAE) | T_gpu ↑ → fewer workers needed to keep up |
| More workers | Pipeline faster → can deliver bigger batches |
| Faster CPU | γ ↓ → fewer workers needed |
| Larger graphs | γ ↑ per graph → more workers needed |
| Faster GPU (V100 → H100) | T_gpu ↓ → more workers needed to keep up |
| torch.compile disabled | VRAM pool shrinks → room for bigger batch |

---

## 3. Budget Implementation (budget.py)

### Current state

`node_budget()` computes the VRAM-limited batch size and measures timing
coefficients (γ, α, β). It does NOT yet compute the full sizing chain
(workers, CPUs, memory). The throughput floor exists in the code but is
vestigial from the old "cap the batch" design — it should be removed in
favor of the GPU-first chain.

```
node_budget()                                    [budget.py]
  ├─ probe()
  │    ├─ Batch.from_data_list(~70 graphs)      ← TIMED → γ
  │    ├─ model.forward(small batch)            ← TIMED
  │    ├─ model.forward(large batch)            ← TIMED → α, β (two-point)
  │    ├─ model.train(); loss.backward()        ← TIMED → backward_multiplier
  │    └─ (bytes_per_node, bytes_per_edge, CostCoefficients)
  ├─ mem_budget = free_vram × 0.85 / effective_bpn
  └─ budget = mem_budget                         ← VRAM is the only batch constraint
```

### What needs to change

The budget system should output a full `ResourceProfile`, not just a node
count. The measurements already exist — it's arithmetic on existing values:

```python
@dataclass
class ResourceProfile:
    # From VRAM probe (existing)
    node_budget: int
    graphs_per_batch: int
    # From timing probe (existing measurements, new computation)
    t_collation_ms: float      # γ × graphs_per_batch
    t_gpu_ms: float            # (α + β × node_budget) × backward_multiplier
    # From the sizing chain (new)
    workers: int               # ceil(t_collation / t_gpu)
    prefetch_factor: int       # 2-4
    cpus: int                  # workers + 2
    memory_gb: int             # workers × worker_rss + base
```

### Known probe bug: torch.compile pool inflation

The probe runs after `torch.compile`, which inflates the VRAM reserved pool
by 4-8 GB. The probe sees reduced free VRAM and underestimates the batch
by 2-3x. Active VRAM utilization is only 19-37% despite the probe trying
to fill it.

Fix: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so the allocator
returns unused segments, or probe before compile.

### Measured probe values

From GPU probe job 46273452, Pitzer V100, set_01:

| model | scale | bpn | bwd_mult | α (ms) | β (μs/node) | γ (μs/graph) |
|---|---|---|---|---|---|---|
| vgae | small | 34,601 | 1.39 | 7.1 | 0.00 | 65 |
| vgae | large | 50,112 | 1.26 | 6.9 | 0.16 | 65 |
| gat | small | 59,838 | 1.29 | 2.7 | 0.85 | 65 |
| gat | large | 223,738 | 1.52 | 4.6 | 0.73 | 65 |
| dgi | small | 13,974 | 2.0* | 7.1 | 0.03 | 65 |
| dgi | large | 80,142 | 2.0* | 6.1 | 0.06 | 65 |

*DGI backward probe fails, falls back to `_GRAD_MULTIPLIER=2.0`.

### Derived sizing chain (target state)

Using the probe values and the GPU-first chain:

| model/scale | mem_budget (nodes) | T_gpu (ms) | T_collate (ms) | Workers needed | CPUs | Est. memory |
|---|---|---|---|---|---|---|
| vgae/small | ~400K | ~7 | ~920 | **132** | 134 | impractical |
| vgae/large | ~230K | ~44 | ~530 | **13** | 15 | ~100 GB |
| gat/small | ~190K | ~164 | ~437 | **3** | 5 | ~36 GB |
| gat/large | ~52K | ~42 | ~120 | **3** | 5 | ~36 GB |
| dgi/small | ~600K | ~25 | ~1,384 | **56** | 58 | impractical |
| dgi/large | ~145K | ~15 | ~334 | **23** | 25 | ~160 GB |

VGAE small and DGI small have such low β (near-zero per-node GPU cost) that
the GPU finishes almost instantly regardless of batch size. For these,
even 132 workers can't keep up because T_gpu ≈ α ≈ 7ms is dominated by
kernel launch overhead, not actual compute. These models are candidates
for **CPU-only training** or **CUDA Graphs** (capture the step as a single
graph to eliminate per-kernel launch overhead).

VGAE large and GAT large/small are the practical targets: 3-13 workers,
feasible on OSC Pitzer.

### Correctness of existing components

| Component | Status | Evidence |
|---|---|---|
| `mem_budget` | **Correct but conservative** | torch.compile pool inflation underestimates by 2-3x |
| `_SAFETY_MARGIN = 0.85` | **Adequate** | Covers allocator frag + optimizer + P/N error |
| `BenchmarkTimer` for GPU | **Proper** | CUDA sync, multi-sample, outlier-robust |
| Linear VRAM extrapolation | **Valid** | P/N error < 0.2% |
| α,β two-point solve | **Correct** | Clamping ≥ 0 handles noise |
| γ measurement | **Bug: contaminated by GPU state** | DGI large γ inflated 3000x |
| `throughput_budget` | **Remove** | Vestigial from old "cap the batch" design |

---

## 4. Literature Assessment

### Established

**GNN training is data-pipeline-bound, not GPU-bound.** Multiple systems
papers measure ~28% GPU time / ~72% data preparation time:

- **SALIENT** (Kaler et al., 2021): 3x speedup from pipeline optimization
  alone. [arXiv:2110.08450](https://arxiv.org/abs/2110.08450)
- **BGL** (Liu et al., NSDI 2023): ~10% GPU utilization in typical DGL
  training. [USENIX](https://www.usenix.org/conference/nsdi23/presentation/liu-tianfeng)
- **BatchGNN** (2023): CPU-side batch preparation dominates distributed GNN
  training. [arXiv:2306.13814](https://arxiv.org/abs/2306.13814)
- **ATC 2025** (Gong et al.): "Training runtime on smaller graphs is dominated
  by framework overhead." [USENIX](https://www.usenix.org/system/files/atc25-gong.pdf)

**`Batch.from_data_list()` is a known PyG bottleneck.**
[PyG #572](https://github.com/pyg-team/pytorch_geometric/issues/572),
[PyG #4891](https://github.com/pyg-team/pytorch_geometric/issues/4891):
DataLoader is 59-83% of runtime.

**The fix is to scale the pipeline, not shrink the batch.** The general
DL optimization literature uniformly recommends increasing batch size when
GPU utilization is low:

- [NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html):
  larger batches increase arithmetic intensity, pushing toward compute-bound.
- [PyTorch Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html):
  "increase batch size aggressively — usually your biggest win."
- [Horace He](https://horace.io/brrr_intro.html): when overhead-bound,
  "increase the size of your data."
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/roofline/):
  batch size determines arithmetic intensity; larger = more efficient.

**Kernel launch overhead is significant for small GNN batches.**
[HiFuse](https://arxiv.org/html/2408.08490v1) (2024): individual kernels
execute in 2.6-3.3μs, creating "substantial overhead from frequent launches."
Reducing kernel count by 43-73% achieved 2.4x speedup.
[CUDA Graphs](https://arxiv.org/html/2501.09398v1) (2025): capturing kernel
sequences eliminates per-launch overhead.

### What this project contributes

| Component | Status |
|---|---|
| GNN training is overhead-bound for small graphs | **Established** — multiple systems papers |
| Fix is to scale pipeline (workers), not shrink batch | **Established** — NVIDIA, PyTorch, roofline model |
| `from_data_list` collation is O(B) | **Established** — PyG design |
| GPU-first sizing chain (VRAM → batch → workers → SLURM) | **Engineering application** of established principles |
| Dual-constraint budget with probe timing | **Implementation detail** — probe measurements are standard |
| Very small models (β ≈ 0) may be better on CPU | **Derived** — correct inference from probe data |

---

## Equation Summary

| Equation | Source |
|---|---|
| $T = N_V / \Delta t_{\text{step}}$ | Definition |
| $\Delta t_{\text{step}} \approx \max(\Delta t_c / W, \Delta t_f + \Delta t_b)$ | Pipeline overlap ([Tan et al.](https://www.usenix.org/conference/atc21/presentation/tan-ying)) |
| $\Delta t_c \propto N_V d_v + N_E(1+d_e)$ | PyG source inspection |
| $\Delta t_f \propto L h^2 (N_E + N_V)$ | [Gilmer et al.](https://arxiv.org/abs/1704.01212) |
| GAT attention: $K \cdot N_E \cdot h$ | [Veličković et al.](https://arxiv.org/abs/1710.10903) |
| VRAM lower bound | [Chen et al.](https://arxiv.org/abs/1604.06174), [Kingma & Ba](https://arxiv.org/abs/1412.6980) |
| Workers $= \lceil T_c / T_{\text{gpu}} \rceil$ | Pipeline saturation condition |
| Arithmetic intensity threshold | [Roofline model](https://jax-ml.github.io/scaling-book/roofline/), [NVIDIA](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html) |
| Proportionality constants | **Must measure** |
