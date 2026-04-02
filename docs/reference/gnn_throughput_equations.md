# GNN Training Throughput: A Cost Model

> A principled decomposition of throughput for small-graph GNN workloads, with explicit sourcing and epistemic status for every claim.

---

## Motivation

For small-graph GNN training, the standard intuition from image/text models — that GPU compute is the bottleneck — breaks down. The forward pass over small sparse graphs is fast relative to the CPU work required to assemble each batch. Understanding *where* time goes requires a cost model that separates collation, transfer, and compute. This document builds that model from first principles, notes where derivations are sourced, and explicitly flags what cannot be stated without empirical measurement.

---

## 1. Throughput Definition

Let throughput $T$ be the number of nodes processed per unit time:

$$T = \frac{N_V^{\text{batch}}}{\Delta t_{\text{step}}}$$

where $N_V^{\text{batch}} = \sum_{i=1}^{B} |V_i|$ is the total node count across the $B$ graphs in a batch and $\Delta t_{\text{step}}$ is wall-clock time per training step. This is a definition, not a claim.

The step time decomposes as:

$$\Delta t_{\text{step}} = \Delta t_{\text{collate}} + \Delta t_{\text{transfer}} + \Delta t_{\text{forward}} + \Delta t_{\text{backward}}$$

With sufficient prefetch depth and `num_workers`, collation of batch $k+1$ overlaps with compute on batch $k$. When this overlap is complete, the effective step time collapses to:

$$\Delta t_{\text{step}} \approx \max\!\left(\Delta t_{\text{collate}},\ \Delta t_{\text{forward}} + \Delta t_{\text{backward}}\right)$$

**Epistemic status:** The pipeline overlap model is the standard analysis for prefetch-based data pipelines. It underlies the design of PyTorch's DataLoader prefetch worker architecture and is used explicitly in systems-level GPU utilization analysis [Tan et al., *Improving GPU Utilization in Deep Learning*, USENIX ATC 2021, [link](https://www.usenix.org/conference/atc21/presentation/tan-ying)]. The `max(·)` form assumes zero synchronization overhead between pipeline stages, which is an idealization — in practice, there is a small scheduling gap.

---

## 2. Collation Cost

PyG's `Batch.from_data_list()` assembles a mini-batch by iterating over $B$ graphs and performing three operations per graph $i$ [[PyG source: `torch_geometric/data/batch.py`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/batch.py)]:

1. **Offset `edge_index`** by the cumulative node count — $O(|E_i|)$ integer additions
2. **Concatenate node features** — $O(|V_i| \cdot d_v)$ memory copy
3. **Concatenate edge features** — $O(|E_i| \cdot d_e)$ memory copy

Summing over the batch:

$$\Delta t_{\text{collate}} \propto \sum_{i=1}^{B} \left(|V_i| \cdot d_v + |E_i| \cdot (1 + d_e)\right)$$

Separating node and edge terms:

$$= \underbrace{\left(\sum_{i=1}^{B} |V_i|\right) d_v}_{N_V^{\text{batch}} \cdot d_v} + \underbrace{\left(\sum_{i=1}^{B} |E_i|\right)(1 + d_e)}_{N_E^{\text{batch}} \cdot (1 + d_e)}$$

**Epistemic status:** The $O(\cdot)$ structure is derived directly from reading PyG source. The proportionality constant is hardware-dependent (CPU clock speed, memory bandwidth, Python overhead) and must be measured empirically. There is no paper that formally derives this expression; it follows from code inspection.

**Key implication:** At fixed node budget $N_V^{\text{batch}} = C$, packing *more, smaller* graphs increases $B$. If those graphs have non-trivial edge density, $N_E^{\text{batch}}$ grows (more edge offset operations), increasing $\Delta t_{\text{collate}}$. **Node count is not the right proxy for collation cost when edge density varies across graphs.**

---

## 3. Forward Pass Cost

For a standard Message Passing Neural Network (MPNN) [Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017, [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)] with $L$ layers and hidden dimension $h$, each layer performs:

- **Message computation** (linear projection per edge): $O(N_E^{\text{batch}} \cdot h^2)$
- **Aggregation** (scatter over edges): $O(N_E^{\text{batch}})$ memory accesses
- **Node update** (MLP per node): $O(N_V^{\text{batch}} \cdot h^2)$

Total forward pass cost:

$$\Delta t_{\text{forward}} \propto L \cdot h^2 \left(N_E^{\text{batch}} + N_V^{\text{batch}}\right)$$

For **attention-based aggregation** (GAT) [Veličković et al., *Graph Attention Networks*, ICLR 2018, [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)], add a per-edge attention coefficient computation with $K$ attention heads:

$$\Delta t_{\text{forward}}^{\text{GAT}} \propto L \left(K \cdot N_E^{\text{batch}} \cdot h + N_E^{\text{batch}} \cdot h^2 + N_V^{\text{batch}} \cdot h^2\right)$$

The $K \cdot N_E^{\text{batch}} \cdot h$ term comes from computing and normalizing attention weights per edge per head, which increases forward pass cost relative to simple aggregation independently of $h^2$ scaling.

**Epistemic status:** The $O(\cdot)$ structure follows from the MPNN formulation in Gilmer et al. and the GAT formulation in Veličković et al. The $h^2$ scaling of linear projections is standard linear algebra. The proportionality constants hide GPU kernel efficiency for sparse scatter/gather operations, which is substantially lower than for dense matrix multiplications — this is hardware-dependent and is not derived here.

---

## 4. Bottleneck Crossover

The pipeline overlap condition is $\Delta t_{\text{collate}} < \Delta t_{\text{forward}} + \Delta t_{\text{backward}}$. Assuming $\Delta t_{\text{backward}} \approx 2 \cdot \Delta t_{\text{forward}}$ (standard autodiff rule of thumb for uniform-cost ops), and substituting the cost models from Sections 2 and 3:

$$N_V^{\text{batch}} \cdot d_v + N_E^{\text{batch}} \cdot (1 + d_e) \lesssim 3L \cdot h^2 \left(N_E^{\text{batch}} + N_V^{\text{batch}}\right)$$

Dividing both sides by $N_V^{\text{batch}}$ and defining mean degree $\bar{\rho} = N_E^{\text{batch}} / N_V^{\text{batch}}$:

$$d_v + \bar{\rho}(1 + d_e) \lesssim 3L \cdot h^2 (1 + \bar{\rho})$$

This makes the crossover conditions explicit in terms of observable quantities:

| Regime | Condition | Action |
|---|---|---|
| **Collation-dominated** | $d_v + \bar{\rho}(1+d_e) \gg 3Lh^2(1+\bar{\rho})$ | Optimize data pipeline: `num_workers`, `persistent_workers`, pre-cache `Batch` objects |
| **Compute-dominated** | $3Lh^2(1+\bar{\rho}) \gg d_v + \bar{\rho}(1+d_e)$ | Optimize packing density and GPU utilization |
| **Balanced** | Both sides comparable | Both matter; profile to find dominant term |

For **small GNNs on small graphs** (small $L$, small $h$, potentially large $d_v$), the left side dominates — this is the collation-bottleneck regime. For **image-like models** (large $L$, large $h$, trivial collation), the right side dominates.

**Epistemic status:** This inequality is derived by combining the cost models in Sections 2 and 3. It is a qualitative guide, not a quantitative prediction — the proportionality constants are absorbed and differ by orders of magnitude between CPU collation ops and GPU sparse kernel ops. The $\Delta t_{\text{backward}} \approx 2 \cdot \Delta t_{\text{forward}}$ approximation holds for ops with uniform backward cost (linear layers) but not for all GNN operations.

---

## 5. Peak VRAM

During a forward + backward pass, the following must reside in GPU VRAM simultaneously:

$$\text{VRAM} \geq \underbrace{L \cdot N_V^{\text{batch}} \cdot h \cdot s}_{\text{activations (retained for backprop)}} + \underbrace{P \cdot s}_{\text{parameters}} + \underbrace{2P \cdot s}_{\text{Adam: } m_t, v_t} + \underbrace{P \cdot s}_{\text{gradients}} + \underbrace{N_E^{\text{batch}} \cdot 8}_{\text{edge\_index (int64)}}$$

where $P$ is total parameter count and $s$ is bytes per element ($s = 4$ for float32, $s = 2$ for float16/bfloat16).

This is a **lower bound**. It excludes:
- CUDA memory allocator reserved-but-unused blocks
- Intermediate buffers within GNN kernels (e.g., attention coefficient tensors in GAT)
- Optimizer temporary workspace

**Epistemic status:** The activation retention term ($L \cdot N_V^{\text{batch}} \cdot h$) follows from the fact that standard autodiff retains all intermediate activations for the backward pass — this is the basis of the memory analysis in [Chen et al., *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174](https://arxiv.org/abs/1604.06174), which proposes gradient checkpointing precisely to reduce this term. The Adam state term (2× parameters for first and second moment estimates $m_t$ and $v_t$) is from [Kingma & Ba, *Adam*, ICLR 2015, [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)]. The `edge_index` term is exact: 2 rows × $N_E^{\text{batch}}$ entries × 8 bytes per int64.

**Gradient checkpointing** trades the activation term for recomputation cost: instead of $L \cdot N_V^{\text{batch}} \cdot h \cdot s$, only $\sqrt{L}$ checkpoints are retained, reducing activation memory to $O(\sqrt{L} \cdot N_V^{\text{batch}} \cdot h \cdot s)$ at the cost of one additional forward pass [Chen et al., 2016].

---

## 6. What Cannot Be Stated as Equations

The following are real effects but do not have closed-form expressions I can source:

**PCIe transfer time for fragmented small tensors.** Many small graphs create many small H2D transfers. PCIe has non-trivial per-transfer latency overhead, so the total transfer time is not simply proportional to total bytes. The relationship depends on DMA engine concurrency and driver-level batching. Measurable with `nvprof` or Nsight Systems; not derivable analytically.

**SM occupancy as a function of graph size.** Small graphs produce small kernels with poor warp utilization. `nvidia-smi` reports utilization as "any kernel running" rather than "running at full warp efficiency." The occupancy loss for sparse GNN kernels is measurable with `ncu` (NVIDIA Nsight Compute) but I am not aware of a closed-form expression for it as a function of $|V|$ and $|E|$.

**The proportionality constants** in $\Delta t_{\text{collate}}$ and $\Delta t_{\text{forward}}$.  These must be measured on your specific hardware and graph distribution. The crossover inequality in Section 4 tells you which regime you are in qualitatively; it does not give you a number.

---

## 7. Diagnostic Protocol

Given the above, the right approach before tuning any knob:

```python
import time, torch

# 1. Pure collation cost (no GPU involvement)
t0 = time.perf_counter()
for batch in loader:
    pass
t_collate = time.perf_counter() - t0

# 2. Collation + H2D transfer cost
t0 = time.perf_counter()
for batch in loader:
    batch = batch.to(device)
t_transfer = time.perf_counter() - t0

# 3. Full forward step (no backward)
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

The deltas give you an empirical estimate of each $\Delta t$ term, localizing the dominant bottleneck before any optimization is attempted.

---

## Summary

| Equation | Epistemic Status | Source |
|---|---|---|
| $T = N_V^{\text{batch}} / \Delta t_{\text{step}}$ | Definition | — |
| $\Delta t_{\text{step}} \approx \max(\Delta t_{\text{collate}},\ \Delta t_{\text{fwd}} + \Delta t_{\text{bwd}})$ | First principles, pipeline overlap | [Tan et al., USENIX ATC 2021](https://www.usenix.org/conference/atc21/presentation/tan-ying) |
| $\Delta t_{\text{collate}} \propto N_V d_v + N_E(1+d_e)$ | First principles from PyG source | [PyG `batch.py`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/batch.py) |
| $\Delta t_{\text{forward}} \propto L h^2 (N_E + N_V)$ | First principles from MPNN formulation | [Gilmer et al., ICML 2017](https://arxiv.org/abs/1704.01212) |
| GAT attention term $K \cdot N_E \cdot h$ | First principles from GAT formulation | [Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903) |
| Crossover inequality | Derived — qualitative only | Combination of above |
| VRAM lower bound | First principles + lower bound caveat | [Chen et al. 2016](https://arxiv.org/abs/1604.06174), [Kingma & Ba 2015](https://arxiv.org/abs/1412.6980) |
| Proportionality constants | **Cannot provide — must measure** | — |
| PCIe fragmentation cost | **Cannot provide — must measure** | — |
| SM occupancy vs. graph size | **Cannot provide — must measure** | — |
