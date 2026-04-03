# Throughput Model & Budget System

> Cost model for GNN training throughput, budget implementation, and literature assessment.
> Consolidates: gnn_throughput_equations.md, throughput-optimal-batching.md, budget-pipeline-analysis.md, budget-cost-model-audit.md

---

## 1. Cost Model (First Principles)

### 1.1 Throughput Definition

Throughput $T$ = nodes processed per unit time:

$$T = \frac{N_V^{\text{batch}}}{\Delta t_{\text{step}}}$$

where $N_V^{\text{batch}} = \sum_{i=1}^{B} |V_i|$ is total node count across $B$ graphs. Step time decomposes as:

$$\Delta t_{\text{step}} = \Delta t_{\text{collate}} + \Delta t_{\text{transfer}} + \Delta t_{\text{forward}} + \Delta t_{\text{backward}}$$

With sufficient prefetch depth and `num_workers`, collation of batch $k+1$ overlaps with compute on batch $k$:

$$\Delta t_{\text{step}} \approx \max\!\left(\Delta t_{\text{collate}},\ \Delta t_{\text{forward}} + \Delta t_{\text{backward}}\right)$$

> **Source:** Standard pipeline overlap analysis. Used explicitly in [Tan et al., USENIX ATC 2021](https://www.usenix.org/conference/atc21/presentation/tan-ying). The $\max(\cdot)$ form assumes zero synchronization overhead (idealization).

### 1.2 Collation Cost

PyG's `Batch.from_data_list()` performs three operations per graph ([PyG source](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/batch.py)):

1. Offset `edge_index` by cumulative node count — $O(|E_i|)$
2. Concatenate node features — $O(|V_i| \cdot d_v)$
3. Concatenate edge features — $O(|E_i| \cdot d_e)$

Summing over the batch:

$$\Delta t_{\text{collate}} \propto N_V^{\text{batch}} \cdot d_v + N_E^{\text{batch}} \cdot (1 + d_e)$$

> **Source:** Derived from PyG source inspection. Proportionality constant is hardware-dependent, must be measured.

**Key implication:** At fixed node budget, packing *more, smaller* graphs increases $B$. If those graphs have non-trivial edge density, $N_E^{\text{batch}}$ grows, increasing collation cost. Node count alone is not the right proxy.

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

### 1.4 Bottleneck Crossover

The pipeline overlap condition is $\Delta t_{\text{collate}} < \Delta t_{\text{forward}} + \Delta t_{\text{backward}}$. Using $\Delta t_{\text{backward}} \approx 2 \cdot \Delta t_{\text{forward}}$ and mean degree $\bar{\rho} = N_E / N_V$:

$$d_v + \bar{\rho}(1 + d_e) \lesssim 3L \cdot h^2 (1 + \bar{\rho})$$

With practical coefficients ($\gamma$, $\beta$, worker count $W$):

| Regime | Condition | Action |
|---|---|---|
| **Collation-dominated** | $\gamma/W > \beta \cdot \bar{m}$ | Optimize pipeline: more workers, pre-caching |
| **Compute-dominated** | $\gamma/W < \beta \cdot \bar{m}$ | Fill VRAM, optimize GPU utilization |
| **Balanced** | Both comparable | Profile to find dominant term |

**The regime is a system property** (model + graph structure + workers), not a batch-size knob. $B$ cancels when comparing collation rate vs GPU rate.

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

### Why "fill VRAM = fast" breaks

"Larger batch = faster training" assumes the GPU is the bottleneck. For images and LLMs, collation is trivial (`torch.stack()` on fixed-shape tensors), so larger batches better utilize GPU parallelism. VRAM is the only constraint.

For GNNs training on many small graphs, this inverts:
- `Batch.from_data_list()` iterates every graph, concatenating variable-size tensors — **O(N_graphs)** CPU work
- GPU compute (message passing) is cheap for small models
- As batch size grows, $T_{\text{collation}}$ grows linearly while $T_{\text{gpu}}$ grows sublinearly. Past a crossover, every additional graph increases wait time more than useful work.

### Profiled evidence (set_01, Pitzer V100)

```
VGAE (fill VRAM → 154K nodes, 5,471 graphs):
  T_collation = 400ms, 2 workers → effective delivery = 200ms/batch
  T_gpu = 25ms
  throughput = 770K nodes/sec
  GPU utilization: 5%

VGAE (throughput-optimal → ~50K nodes, ~1,800 graphs, 6 workers):
  T_collation = ~130ms, 6 workers → effective delivery = ~22ms/batch
  T_gpu = 25ms
  throughput = 2,000K nodes/sec  (2.6x faster)
  GPU utilization: ~100%
```

A 3x smaller batch yields 2.6x higher throughput because the pipeline finally delivers batches as fast as the GPU consumes them.

### The optimal batch size

The crossover batch $B^*$ exists only in the collation-dominated regime (§1.4):

```
B* = α / (γ/W − β·mean_nodes)     (exists only when γ/W > β·mean_nodes)
```

$B^*$ is a **floor**, not a ceiling. Going above $B^*$ doesn't hurt throughput (flat at $\bar{m} \cdot W / \gamma$). Going below wastes GPU overhead.

Budget decision: `budget = max(N_floor, 1)` then `budget = min(budget, mem_budget)`.

### Self-tuning properties

| What changes | Effect on throughput budget |
|---|---|
| Larger model (GAT vs VGAE) | $T_{\text{gpu}} \uparrow$ → larger batches fine (memory binds) |
| More workers | pipeline depth ↑ → larger batches deliverable |
| Faster CPU | $\gamma \downarrow$ → larger batches deliverable |
| Larger graphs | $\gamma \uparrow$ per graph → smaller batches needed |
| **Faster GPU (V100 → H100)** | $T_{\text{gpu}} \downarrow$ → **smaller batches needed** |

The last row is counterintuitive: faster GPUs need smaller batches because they finish sooner and starve faster. Correct under throughput thinking, wrong under "fill VRAM" thinking.

### Applicability

This analysis applies specifically to:
1. **Small models** — GPU compute is fast relative to data prep
2. **Variable-size samples** — batch construction is O(N) not O(1)
3. **Many samples per batch** — thousands of small graphs, not tens of large ones

For large-graph GNNs (molecular dynamics, social networks), batches contain few graphs and collation is fast. VRAM binds. Conventional wisdom holds.

---

## 3. Budget Implementation (budget.py)

### Old (memory-only) vs New (dual-constraint)

**Old:** One question — "What fits in VRAM?"

```
vram_node_budget()                              [deleted]
  ├─ _probe_bytes_per_node()                    [deleted]
  │    ├─ Batch.from_data_list(~70 graphs)      — untimed
  │    └─ model.forward(batch)                  — untimed, VRAM only
  └─ budget = free_vram × 0.85 / bytes_per_node
       → VGAE: 154K nodes (5,471 graphs)  ← fills VRAM, GPU idle 95%
       → GAT:  36K nodes (1,289 graphs)   ← fills VRAM, GPU idle 78%
```

**New:** Two questions, tighter answer wins.

```
node_budget()                                    [budget.py]
  ├─ probe()
  │    ├─ Batch.from_data_list(~70 graphs)      ← TIMED → γ
  │    ├─ model.forward(small batch)            ← TIMED
  │    ├─ model.forward(large batch)            ← TIMED → α, β (two-point)
  │    └─ (bytes_per_node, CostCoefficients)
  ├─ mem_budget = free_vram × 0.85 / bytes_per_node
  ├─ throughput_budget (from affine model)       [NEW]
  └─ budget = min(mem_budget, throughput_budget)
       → VGAE: mem=154K, tput=58K → budget=58K  (throughput binds)
       → GAT:  mem=36K,  tput=210K → budget=36K (memory binds)
```

### What budget.py measures

**γ (collation rate):** `time(Batch.from_data_list(graphs)) / len(graphs)`. Units: seconds/graph. Bug: measured after `model.to(device)` — CUDA context init can stall CPU (DGI large measured γ = 200ms vs expected 65μs). Fix: `torch.cuda.synchronize()` before timing.

**α, β (GPU timing):** Two-point probe using `BenchmarkTimer` (handles CUDA sync, multi-sample median). Measures **forward-only** time in eval mode. Training β ≈ β × backward_multiplier — means `cg_ratio` is ~2× too high (diagnostic only, doesn't affect budget).

**bytes_per_node:** Peak `max_memory_allocated` delta from one forward pass / node count. Linear extrapolation assumption is valid — P/N error < 0.2% for all models. Probe OVERESTIMATES → budget is conservative (safe).

**backward_multiplier:** `(training fwd+bwd peak) / (inference fwd peak)`. Measured values: 1.26-1.55. DGI falls back to `_GRAD_MULTIPLIER=2.0` (dual-encoder `_step` fails).

**mem_budget:** `free_vram × _SAFETY_MARGIN / effective_bpn`. Called after `model.to(device)` but before optimizer — Adam state not yet allocated (gap: 2P bytes, worst case 0.14%, absorbed by 15% margin).

### Correctness

| Component | Status | Evidence |
|---|---|---|
| `budget = mem_budget` | **Correct** | Throughput floor << mem_budget for all configs |
| `_SAFETY_MARGIN = 0.85` | **Adequate** | Covers allocator frag + optimizer + P/N error |
| `BenchmarkTimer` for GPU | **Proper** | CUDA sync, multi-sample, outlier-robust |
| Linear VRAM extrapolation | **Valid** | P/N error < 0.2% |
| α,β two-point solve | **Correct** | Clamping ≥ 0 handles noise |
| `skip_too_big=True` | **Correct** | No CAN bus graph exceeds budget |

### Bugs found (2026-04-02 audit)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | Stale docstrings say `min(mem, throughput)` — code does `mem_budget` only | Misleading | ~10 lines |
| 2 | γ contaminated by GPU state (CUDA context init) | Inflated cg_ratio for some combos | `synchronize()` before timing |
| 3 | `cg_ratio` uses forward-only β instead of training β | Diagnostic ~2× too high | Multiply by `bwd_mult` |
| 4 | `num_steps` uses `int()` (floor) instead of `math.ceil` — can skip 10-15% of data | Slower convergence | 1-line fix |
| 5 | No throughput floor guard (N_floor not enforced) | None currently (mem >> floor) | ~5 lines |

---

## 4. Measured Probe Values

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

### Derived throughput floor (N_floor, nodes) at W=6

| model/scale | γ/(m̄·W) μs/node | β μs/node | regime | N_floor |
|---|---|---|---|---|
| vgae/small | 0.384 | 0.00 | collation | 521 |
| vgae/large | 0.384 | 0.16 | collation | ~705 |
| gat/small | 0.384 | 0.85 | compute | no floor |
| gat/large | 0.384 | 0.73 | compute | no floor |
| dgi/small | 0.384 | 0.03 | collation | ~558 |
| dgi/large | 0.384 | 0.06 | collation | ~533 |

All floors well below smallest mem_budget (GAT large V100 ≈ 54K nodes).

---

## 5. Literature Assessment

### Established

**GNN training is data-pipeline-bound, not GPU-bound.** Multiple systems papers measure ~28% GPU time / ~72% data preparation time:

- **SALIENT** (Kaler et al., 2021): 3x speedup from pipeline optimization alone. [arXiv:2110.08450](https://arxiv.org/abs/2110.08450)
- **BGL** (Liu et al., NSDI 2023): ~10% GPU utilization in typical DGL training. [USENIX](https://www.usenix.org/conference/nsdi23/presentation/liu-tianfeng)
- **BatchGNN** (2023): CPU-side batch preparation dominates distributed GNN training. [arXiv:2306.13814](https://arxiv.org/abs/2306.13814)

**`Batch.from_data_list()` is a known PyG bottleneck.** [PyG #572](https://github.com/pyg-team/pytorch_geometric/issues/572): original 3x slower, rewritten. [PyG #4891](https://github.com/pyg-team/pytorch_geometric/issues/4891): DataLoader is 59-83% of runtime.

**Collation scales linearly with graph count.** Inherent to PyG's block-diagonal batching. [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html).

**Small-graph packing is a recognized problem.** Graphcore's [IPU tutorial](https://docs.graphcore.ai/projects/tutorials/en/latest/pytorch_geometric/4_small_graph_batching_with_packing/README.html) addresses padding waste, but focuses on fixed-size hardware constraints, not throughput-optimal sizing.

### Novel (not found in literature)

| Component | Status |
|---|---|
| GNN training is data-pipeline-bound | **Established** — multiple systems papers |
| `from_data_list` is a bottleneck | **Established** — PyG maintainers acknowledged |
| Collation cost is O(N_graphs) | **Established** — inherent to PyG design |
| Throughput-optimal batch < VRAM-optimal batch | **Derived** — correct inference, not published |
| Dual-constraint budget with probe timing | **Novel** — not found in literature |
| Faster GPU → smaller optimal batch | **Derived** — logical consequence, not published |

No paper frames batch sizing as `T_collation ≤ T_gpu × pipeline_depth`. The systems papers focus on making the pipeline faster, not on reducing batch size to match pipeline capacity. No paper discusses the "many small graphs" regime where collation — not neighborhood sampling — is the bottleneck.

---

## Equation Summary

| Equation | Source |
|---|---|
| $T = N_V / \Delta t_{\text{step}}$ | Definition |
| $\Delta t_{\text{step}} \approx \max(\Delta t_c, \Delta t_f + \Delta t_b)$ | Pipeline overlap ([Tan et al.](https://www.usenix.org/conference/atc21/presentation/tan-ying)) |
| $\Delta t_c \propto N_V d_v + N_E(1+d_e)$ | PyG source inspection |
| $\Delta t_f \propto L h^2 (N_E + N_V)$ | [Gilmer et al.](https://arxiv.org/abs/1704.01212) |
| GAT attention: $K \cdot N_E \cdot h$ | [Veličković et al.](https://arxiv.org/abs/1710.10903) |
| VRAM lower bound | [Chen et al.](https://arxiv.org/abs/1604.06174), [Kingma & Ba](https://arxiv.org/abs/1412.6980) |
| Proportionality constants | **Must measure** |
