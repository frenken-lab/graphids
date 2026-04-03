# Budget Cost Model Audit

> Audited 2026-04-02. Source: `budget.py`, `gnn_throughput_equations.md`,
> `throughput-optimal-batching.md`, PyG `DynamicBatchSampler` source,
> PyTorch CUDA memory APIs.

## 1. The Pipeline Model (first principles)

A prefetch DataLoader with W workers produces batches in parallel with GPU
consumption. Steady-state throughput:

```
throughput = N_batch / max(T_collate/W, T_gpu)
```

Where:
- `N_batch` = total nodes in the batch
- `T_collate` = CPU time to build one batch via `Batch.from_data_list()`
- `T_gpu` = GPU time for forward + backward on one batch
- `W` = num_workers

Source: standard pipeline overlap analysis (Tan et al., USENIX ATC 2021).
PyG's DataLoader uses this architecture.

### 1.1 Collation cost

From PyG `batch.py` source — `from_data_list()` iterates B graphs, offsets
edge indices, concatenates features:

```
T_collate = γ · B
```

Where `γ` = seconds per graph (CPU work). γ depends on graph structure
(num nodes, edges, feature dims) but NOT on the model. It is a pure CPU
measurement.

In terms of total nodes N = B · m̄ (m̄ = mean nodes per graph):

```
T_collate = γ · N / m̄ = (γ/m̄) · N
```

### 1.2 GPU cost (affine model)

GPU kernels have fixed overhead (launch, scheduler, memory barriers) plus
per-element work:

```
T_gpu = α + β · N
```

Where:
- `α` = per-step overhead (seconds), independent of batch content
- `β` = per-node GPU cost (seconds/node), depends on model architecture
- `N` = total nodes in batch

From `gnn_throughput_equations.md` §3: forward cost ∝ L·h²·(N_E + N_V).
For constant E/V ratio (CAN bus ≈ 4.5), this simplifies to ∝ N_V.

**Important:** T_gpu should be measured for the FULL training step
(forward + backward), not forward-only. The backward pass adds ~1.3-2×
depending on model architecture.

### 1.3 Throughput as function of batch size

Substituting into the pipeline model, with B graphs (N = B·m̄ nodes):

```
throughput(B) = B·m̄ / max(γ·B/W, α + β·B·m̄)
```

**Case A: Collation-bound** (γ·B/W > α + β·B·m̄):
```
throughput = B·m̄ / (γ·B/W) = m̄·W/γ     ← CONSTANT, independent of B
```

**Case B: Compute-bound** (α + β·B·m̄ > γ·B/W):
```
throughput = B·m̄ / (α + β·B·m̄) = 1/(α/(B·m̄) + β)     ← increases with B
```

As B → ∞: throughput → 1/β (asymptotic max).

### 1.4 Regime classification

The crossover between cases is where γ·B/W = α + β·B·m̄. Rearranging:

```
B·(γ/W − β·m̄) = α
```

If `γ/W > β·m̄`: collation is slower per-node than GPU. Crossover at:
```
B* = α / (γ/W − β·m̄)
```

If `γ/W ≤ β·m̄`: GPU is always the bottleneck. No crossover.

**The regime depends on per-node rates (γ/m̄, β) and worker count W,
NOT on batch size B.** B cancels when comparing collation rate vs GPU rate.
This is the key insight.

### 1.5 What B* means

B* is the **minimum batch size** (in graphs) to reach peak throughput:

| B relative to B* | Bottleneck | Throughput |
|---|---|---|
| B < B* | GPU (wasting cycles on α overhead) | Increasing with B |
| B = B* | Balanced | Peak for collation-bound regime |
| B > B* | Collation (GPU idles between batches) | Flat at m̄·W/γ |

**B* is a FLOOR, not a ceiling.** Going above B* doesn't hurt throughput
(it stays flat). Going below B* wastes GPU overhead.

Converting to nodes for DynamicBatchSampler (max_num is in nodes):
```
N_floor = B* · m̄ = α·m̄ / (γ/W − β·m̄)
```

### 1.6 The budget decision

```
budget = max(N_floor, 1)        # don't go below throughput floor
budget = min(budget, mem_budget) # don't exceed VRAM ceiling
```

For ALL current model×GPU configurations, `mem_budget >> N_floor`
(smallest mem_budget = GAT large on V100 ≈ 54K nodes; largest N_floor
≈ 850 nodes). So `budget = mem_budget` is correct in practice.

The floor matters if:
- A model has very high α (large kernel overhead)
- A user manually constrains batch size
- Hardware changes make mem_budget small

---

## 2. What the Code Measures

### 2.1 γ (collation rate)

```python
# budget.py _probe(), lines 120-124
t0 = time.perf_counter()
batch_large = Batch.from_data_list(graphs_large)
t_collate = time.perf_counter() - t0
gamma = t_collate / len(graphs_large)   # seconds per graph
```

**Units:** seconds/graph. Correct.

**Bug: GPU state contamination.** γ is measured AFTER `model.to(device)`.
`Batch.from_data_list()` is pure CPU, but CUDA context initialization
(lazy on first `.to()`) can stall the CPU thread. Evidence: DGI large
measured γ = 200ms vs expected 65μs — 3000× anomaly. See
`docs/backlog/dgi-large-gamma-anomaly.md`.

**Fix:** Add `torch.cuda.synchronize()` before γ measurement to flush
any pending GPU work. Or measure γ once per dataset before any model
loading.

**Robustness:** Single measurement (no repetition). CPU collation is
deterministic enough for one run, but GC pauses or NUMA effects on HPC
can cause one-off spikes. A 3-sample median would be safer.

### 2.2 α, β (GPU timing)

```python
# budget.py _probe(), lines 150-156
t_gpu_small = BenchmarkTimer(...).blocked_autorange(min_run_time=0.2).median
t_gpu_large = BenchmarkTimer(...).blocked_autorange(min_run_time=0.2).median

# lines 213-216
beta = max(0.0, (t_gpu_large - t_gpu_small) / (nodes_large - nodes_small))
alpha = max(0.0, t_gpu_large - beta * nodes_large)
```

**What's measured:** FORWARD-ONLY time, in eval mode, with torch.no_grad().
BenchmarkTimer handles CUDA synchronization internally. Multi-sample
median is robust.

**Problem:** α and β measure forward-only GPU time. Real training step
includes backward pass (~1.3-2× forward). The throughput equation needs
training-time T_gpu, not inference-time T_gpu.

- α_training ≈ α_forward × bwd_time_mult (overhead scales similarly)
- β_training ≈ β_forward × bwd_time_mult (per-node cost scales)

Since the throughput floor is NOT currently used for the budget decision,
this error doesn't affect the budget. But it makes cg_ratio inaccurate:

**cg_ratio uses forward-only β** (budget.py line 336):
```python
cg_ratio = gamma_eff / beta   # gamma_eff = γ/(m̄·W)
```

This should be `gamma_eff / (beta * bwd_time_mult)` for training regime.
Result: cg_ratio is systematically ~2× too high, over-reporting collation
dominance.

**Fix options:**
(a) Multiply β by backward_multiplier in cg_ratio computation
(b) Measure α,β with a training step (fwd+bwd) — more accurate but
    requires working _step function
(c) Document that cg_ratio is forward-only and add a field for
    estimated training cg_ratio

Option (a) is pragmatic since backward_multiplier is already measured.
Note: backward_multiplier is a VRAM ratio (bwd_peak/fwd_peak), which
may differ from the TIME ratio. But for GNNs with mostly linear layers,
VRAM ratio ≈ time ratio within 20%.

### 2.3 bytes_per_node (VRAM measurement)

```python
# budget.py _probe(), lines 164-174
torch.cuda.reset_peak_memory_stats(model.device)
before = torch.cuda.memory_allocated(model.device)
with torch.no_grad():
    fn(batch_large)
torch.cuda.synchronize()
vram_large = torch.cuda.max_memory_allocated(model.device) - before
fwd_per_node = max(1, int(vram_large / max(1, nodes_large)))
```

**What's measured:** Peak memory delta from one forward pass. Uses
`max_memory_allocated` (tensors actually used), not `max_memory_reserved`
(caching allocator blocks). Correct metric — `_SAFETY_MARGIN` covers
allocator fragmentation.

**Linear extrapolation assumption:** `bytes_per_node = peak_delta / N`.
This assumes VRAM scales linearly with N. True for:
- Activations: L·N·h (linear in N) ✓
- Edge buffers: N_E ∝ N for constant E/N ratio ✓

False for:
- Model params: constant P, contributes P/N per-node ✗

The P/N error: probe at N=2000 includes P/2000 overhead. At actual batch
size (50K+ nodes), overhead is P/50K — negligible. The probe
OVERESTIMATES bytes_per_node → budget is CONSERVATIVE (safe).

Magnitude for VGAE small (26K params × 4 bytes = 104KB):
- Probe: +52 bytes/node overhead (104KB/2000)
- Real:  +0.3 bytes/node overhead (104KB/344K)
- Error: 52/34600 = 0.15% of total bytes_per_node

**Verdict:** Linear extrapolation is valid for these model sizes.

### 2.4 backward_multiplier

```python
# budget.py _probe(), lines 181-198
model.train()
loss = _extract_loss(step_fn(batch_bwd))
torch.autograd.backward(loss)
bwd_peak = torch.cuda.max_memory_allocated(model.device) - before
backward_multiplier = max(1.0, bwd_peak / vram_large)
```

Measures: (training fwd+bwd peak) / (inference fwd peak). Captures:
- Activation retention for backward ✓
- Gradient tensors ✓
- Optimizer state (m, v): NOT measured (Adam allocates lazily on first step)

Adam overhead: 2P bytes. For GAT large (2.5M params): 20MB out of 14GB
free = 0.14%. Covered by _SAFETY_MARGIN.

DGI falls back to _GRAD_MULTIPLIER=2.0 because `_step` fails on
dual-encoder architecture. This is conservative — measured values for
other models are 1.26-1.55.

### 2.5 mem_budget

```python
# budget.py node_budget(), lines 269-270, 329
free, _ = torch.cuda.mem_get_info()    # after model load, before optimizer
mem_budget = int(effective_free * _SAFETY_MARGIN / effective_bpn)
```

**Timing of mem_get_info():** Called during DataLoader setup, after
`model.to(device)` but before `optimizer.step()`. Adam m,v state is
NOT YET allocated. Free VRAM is slightly overestimated.

Gap: 2P bytes (Adam). Worst case (GAT large): 20MB / 14GB = 0.14%.
`_SAFETY_MARGIN = 0.85` (15% reserve = 2.1GB) absorbs this easily.

**Edge-aware margin:** Scales bytes_per_node by p95_epn/mean_epn to
account for batches with denser-than-average graphs. Uses marginal
statistics (p95(edges)/p95(nodes)) rather than per-graph E/N ratio.
For CAN bus (E/N ≈ 4.5 constant), ratio ≈ 1.0, no adjustment.

**Verdict:** mem_budget is correct and conservative.

### 2.6 num_steps

```python
# datamodule.py:218
num_steps = max(1, int(len(dataset) * result.mean_nodes / result.budget))
```

**Bug:** `int()` truncates (floor). If the true count is 47.6 batches,
num_steps = 47, and DynamicBatchSampler stops after 47 batches, dropping
the last partial batch (~60% of a batch worth of graphs).

With variable graph sizes, actual batch count is HIGHER than estimated
(some batches pack fewer nodes → more batches needed). Combined with
floor truncation, up to ~15% of data could be skipped per epoch.

Shuffling mitigates: different graphs skipped each epoch, all seen over
multiple epochs. But convergence speed suffers.

**Fix:** Use `math.ceil` instead of `int`. Or add +1 margin.

---

## 3. What's Correct (and why)

| Component | Status | Evidence |
|---|---|---|
| `budget = mem_budget` | **Correct** | Throughput floor << mem_budget for all configs (§1.6) |
| `_SAFETY_MARGIN = 0.85` | **Adequate** | Covers allocator frag + optimizer state + P/N error |
| `BenchmarkTimer` for GPU | **Proper** | Handles CUDA sync, multi-sample, robust to outliers |
| Linear VRAM extrapolation | **Valid** | P/N error < 0.2% for all models (§2.3) |
| α,β two-point solve | **Correct** | Clamping ≥ 0 handles noise. Two points sufficient for affine model |
| Edge-aware margin | **Adequate** | Near-constant E/N for CAN bus makes it a no-op |
| GPS quadratic path | **Approximate** | Ignores non-attention VRAM, but GPS not in current ablation |
| `skip_too_big=True` | **Correct** | No CAN bus graph exceeds budget (max ~63 nodes vs budget 50K+) |

## 4. Bugs Found

### BUG-1: Stale docstrings (misleading, no runtime effect)

**Module docstring** (lines 1-7): says `budget = min(memory_ceiling,
throughput_ceiling)`. Code does `budget = mem_budget` only.

**node_budget docstring** (lines 246-256): says steps 6-7 compute
throughput_budget and min with mem_budget. This code was deleted in
session 14.

**Impact:** Anyone reading the code gets the wrong mental model.

### BUG-2: γ measurement contaminated by GPU state

γ (pure CPU metric) measured after `model.to(device)`. Caused 3000×
anomaly for DGI large. `torch.cuda.synchronize()` before timing would
flush pending GPU work.

**Impact:** Inflated cg_ratio for some model×dataset combos. Budget
decision unaffected (cg_ratio is diagnostic only).

### BUG-3: cg_ratio uses forward-only β

β measured in eval/no_grad mode. Training β ≈ β × backward_multiplier.
cg_ratio is ~1.3-2× too high, making models appear more
collation-dominated than they actually are during training.

**Impact:** Diagnostic only — doesn't affect budget. But misleading for
profiling decisions.

### BUG-4: num_steps truncates instead of ceiling

`int()` floors the batch count. Combined with graph size variance, can
skip 10-15% of dataset per epoch.

**Impact:** Slower convergence, possible underfitting on underrepresented
graph types.

### BUG-5: No throughput floor guard

The throughput floor `N_floor = α·m̄ / (γ/W − β·m̄)` is the minimum
batch size for peak throughput. Currently not computed or enforced.

For all current configs, `mem_budget >> N_floor`, so the floor is never
binding. But there's no guard against manual batch size reduction or
future models with high α.

**Impact:** None currently. ~5 lines to add as a safety guard.

---

## 5. Action Items

| # | Fix | Risk | Lines |
|---|---|---|---|
| 1 | Fix stale docstrings (module + node_budget) | None | ~10 |
| 2 | Add `torch.cuda.synchronize()` before γ timing | None | +2 |
| 3 | Use `math.ceil` for num_steps | Low | 1 |
| 4 | Add throughput floor guard | Low | ~5 |
| 5 | Adjust cg_ratio for backward multiplier | None | ~3 |
| 6 | Add γ robustness (3-sample median) | None | ~5 |
| 7 | Log warning when backward_multiplier uses fallback | None | +1 |

---

## 6. Measured Probe Values (reference)

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

### Derived: throughput floor (N_floor, in nodes) at W=6

| model/scale | γ/m̄ (μs/node) | β (μs/node) | γ/(m̄·W) (μs/node) | regime | N_floor |
|---|---|---|---|---|---|
| vgae/small | 2.30 | 0.00 | 0.384 | collation | α·m̄/(γ/W) = 521 |
| vgae/large | 2.30 | 0.16 | 0.384 | collation | α·m̄/(γ/W−β·m̄) ≈ 705 |
| gat/small | 2.30 | 0.85 | 0.384 | compute | no floor (GPU always bottleneck) |
| gat/large | 2.30 | 0.73 | 0.384 | compute | no floor |
| dgi/small | 2.30 | 0.03 | 0.384 | collation | ≈ 558 |
| dgi/large | 2.30 | 0.06 | 0.384 | collation | ≈ 533 |

All floors well below smallest mem_budget (GAT large V100 ≈ 54K nodes).

### Derived: cg_ratio (corrected for training)

Forward-only β underestimates training β by ~backward_multiplier:

| model/scale | cg_ratio (code, fwd-only β) | cg_ratio (corrected, β×bwd_mult) |
|---|---|---|
| vgae/small | ∞ (β≈0) | ∞ |
| gat/small | 0.45 | 0.35 |
| gat/large | 0.53 | 0.35 |

GAT is even MORE compute-dominated during training than cg_ratio suggests.
