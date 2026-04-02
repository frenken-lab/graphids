# Throughput-Optimal Batching for Small-Graph GNNs

> Status: correct application of established principles to a specific regime.
> The individual components are well-sourced; the regime-based budget is a
> novel (to us) synthesis. See "Sources" section for detailed assessment.
>
> **Correction (2026-04-02):** original version treated T_gpu as constant
> regardless of batch size. This is wrong — T_gpu scales with batch content
> (Section 3 of gnn_throughput_equations.md). The corrected model uses an
> affine GPU model T_gpu = α + β·N and solves the fixed-point equation.
> See budget.py for the implementation.

## The Conventional Wisdom

"Larger batch = faster training" assumes the GPU is the bottleneck. For images
and LLMs, data loading is trivial (`torch.stack()` on fixed-shape tensors is
O(1)), so larger batches better utilize GPU parallelism. VRAM is the only
constraint on batch size.

## Where It Inverts

For GNNs training on many small graphs, batch construction is expensive:

- `Batch.from_data_list()` iterates over every graph, concatenates variable-size
  tensors, offsets edge indices — **O(N_graphs)** CPU work
- GPU compute (message passing) is cheap for small models — a few sparse matmuls
  per layer, sublinear scaling with batch size

As batch size grows, T_collation grows linearly while T_gpu grows sublinearly.
Past a crossover point, every additional graph increases wait time more than
useful work. Throughput decreases.

### Evidence from profiled runs (set_01, Pitzer V100)

```
VGAE (fill VRAM → 154K nodes, 5,471 graphs):
  T_collation = 400ms, 2 workers → effective delivery = 200ms/batch
  T_gpu = 25ms
  throughput = 154K nodes / 200ms = 770K nodes/sec
  GPU utilization: 5%

VGAE (throughput-optimal → ~50K nodes, ~1,800 graphs, 6 workers):
  T_collation = ~130ms, 6 workers → effective delivery = ~22ms/batch
  T_gpu = 25ms
  throughput = 50K nodes / 25ms = 2,000K nodes/sec  (2.6x faster)
  GPU utilization: ~100%
```

A 3x smaller batch yields 2.6x higher throughput because the pipeline can
finally deliver batches as fast as the GPU consumes them.

## The Principle

**Optimal batch size maximizes throughput (samples/sec), not VRAM utilization.**

These coincide in compute-bound regimes (data loading ≈ free). They diverge in
data-bound regimes (batch construction cost >> GPU compute).

The correct constraint:

```
T_collation(batch_size) ≤ T_gpu(batch_size) × pipeline_depth

where pipeline_depth = num_workers × prefetch_factor
```

This is a throughput equation, not a memory equation. VRAM remains a hard safety
ceiling (OOM = crash), but the throughput constraint should be the actual sizing
knob.

## Corrected Design: Regime Classification + Affine GPU Model

### Why "T_gpu = constant" is wrong

The original design treated T_gpu as a single number measured from one probe.
But Section 3 of gnn_throughput_equations.md shows T_gpu ∝ L·h²·(N_E + N_V) —
it scales with batch size. Since BOTH T_collation and T_gpu scale with batch
content, their ratio is approximately constant. You can't "optimize" it by
changing batch size.

### What the equations actually tell us

If T_collation and T_gpu both scale linearly with batch content:

```
T_collation(B) / W  vs  T_gpu(B)
(γ × B) / W         vs  (β × B)
γ / W                vs  β           ← B cancels out
```

The crossover is a property of the system, not the batch size:
- γ/W > β → collation-dominated for ALL batch sizes. More workers is the fix.
- γ/W < β → compute-dominated for ALL batch sizes. Fill VRAM.

### Where a finite optimal batch DOES exist

GPUs have kernel launch overhead (α) — a constant cost per step independent
of batch size:

```
T_gpu(N) = α + β·N     (affine, not purely linear)
```

Solving T_collation(B)/W = T_gpu(B):

```
γ·B / W = α + β·B·mean_nodes
B = α / (γ/W - β·mean_nodes)     (exists only when γ/W > β·mean_nodes)
```

The overhead α makes very small batches inefficient (high overhead fraction).
The optimal batch amortizes α while staying within the delivery rate.

### Implementation

The probe runs at two batch sizes to separate α and β:

```python
# Two-point GPU measurement
t_gpu_small = time_forward(batch_small)   # ~200 nodes
t_gpu_large = time_forward(batch_large)   # ~2000 nodes

β = (t_gpu_large - t_gpu_small) / (nodes_large - nodes_small)
α = t_gpu_large - β × nodes_large
```

Budget = `min(mem_budget, throughput_budget)` where throughput_budget exists
only in the collation-dominated regime with measurable α. See `budget.py`.

### Self-tuning properties

| What changes | Effect on throughput budget |
|---|---|
| Larger model (GAT vs VGAE) | T_gpu ↑ → larger batches fine (memory binds) |
| More workers | pipeline_depth ↑ → larger batches deliverable |
| Faster CPU | collate_per_graph ↓ → larger batches deliverable |
| Larger graphs | collate_per_graph ↑ → smaller batches needed |
| Faster GPU (V100 → H100) | T_gpu ↓ → smaller batches needed |

The last row is counterintuitive: faster GPUs need smaller batches because
they finish sooner and starve faster. Correct under throughput thinking,
wrong under "fill VRAM" thinking.

### When memory still binds

For GAT (2.5M params), T_gpu is large enough that the throughput budget exceeds
the memory budget. The memory constraint is already the tighter one, and current
behavior is correct. The throughput constraint only bites for small models
(VGAE 745K, DGI) where GPU compute is trivially fast.

## Applicability

This analysis applies specifically to:
1. **Small models** — GPU compute per batch is fast relative to data prep
2. **Variable-size samples** — batch construction is O(N) not O(1)
3. **Many samples per batch** — thousands of small graphs, not tens of large ones

For large-graph GNNs (molecular dynamics, social networks), batches contain
few graphs and collation is fast. VRAM binds. The conventional wisdom holds.

## Sources and Literature Assessment

### What IS established

**GNN training is data-pipeline-bound, not GPU-bound.**
Multiple systems papers measure ~28% GPU time / ~72% data preparation time.
This is the most consistent finding across the GNN systems literature.

- SALIENT (Kaler et al., 2021): "mini-batch preparation and transfer [are]
  major performance bottlenecks hitherto under-explored." 3x speedup from
  pipeline optimization alone. [arXiv:2110.08450](https://arxiv.org/abs/2110.08450)
- BGL (Liu et al., NSDI 2023): "huge gap between data I/O and preprocessing
  speed versus GPU computation speed... pipelining can only hide a small
  fraction." Only ~10% GPU utilization in typical DGL training jobs.
  [USENIX](https://www.usenix.org/conference/nsdi23/presentation/liu-tianfeng)
- BatchGNN (2023): CPU-side batch preparation dominates distributed GNN training.
  [arXiv:2306.13814](https://arxiv.org/abs/2306.13814)

**`Batch.from_data_list()` is a known PyG bottleneck.**

- [PyG issue #572](https://github.com/pyg-team/pytorch_geometric/issues/572):
  original implementation used Python loops, was 3x slower than restructured
  version. Maintainers acknowledged and rewrote.
- [PyG CPU roadmap #4891](https://github.com/pyg-team/pytorch_geometric/issues/4891):
  DataLoader identified as 59-83% of total runtime. `from_numpy` and type
  conversion are the dominant costs.

**Collation scales linearly with graph count.**
Inherent to PyG's block-diagonal batching: concatenate variable-size tensors,
offset edge indices per graph. [PyG batching docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html).

**Data-bound vs compute-bound is standard systems analysis.**
Amdahl's law directly applies — the serial CPU portion caps speedup regardless
of GPU speed. Well-covered in general GPU optimization literature.
[Unite.AI: 124x Slower](https://www.unite.ai/pytorch-dataloader-slowdown-gpu-starvation-kernel-trace/):
"The GPU wasn't slow — it was starving."

**Small-graph packing is a recognized problem.**
Graphcore's [IPU tutorial](https://docs.graphcore.ai/projects/tutorials/en/latest/pytorch_geometric/4_small_graph_batching_with_packing/README.html)
addresses padding waste (39.91% → 84.23% packing efficiency), but focuses on
fixed-size hardware constraints, not throughput-optimal sizing.

### What is NOT in the literature

- **No paper frames batch sizing as `T_collation ≤ T_gpu × pipeline_depth`.**
  The systems papers focus on making the pipeline faster (better sampling,
  caching, GPU-side preprocessing), not on reducing batch size to match
  pipeline capacity.

- **No paper discusses the counterintuitive implication that faster GPUs
  need smaller batches** in data-bound regimes.

- **The dual-constraint budget (`min(memory, throughput)`) with probe-derived
  timing is not a published design pattern.** It is a novel (to us) synthesis
  of established principles.

- **No paper specifically addresses the "many small graphs" regime** where
  collation — not neighborhood sampling — is the bottleneck. The GNN systems
  literature focuses on large graphs (social networks, citation graphs, OGB)
  where sampling dominates.

### Why this wasn't obvious earlier

`DynamicBatchSampler` was designed to prevent OOM on variable-size graphs.
It does its job — no OOM. But the implicit assumption that "fill VRAM = fast"
went unquestioned because it IS correct for the large-graph regime the tool
was designed for. Our workload — millions of small CAN bus graphs with a tiny
model — hits a different regime where collation dominates compute by 16:1.
This regime is underrepresented in the literature because most GNN benchmarks
use large graphs.

### Classification

| Component | Status |
|-----------|--------|
| GNN training is data-pipeline-bound | **Established** — multiple systems papers |
| `from_data_list` is a bottleneck | **Established** — PyG maintainers acknowledged |
| Collation cost is O(N_graphs) | **Established** — inherent to PyG design |
| Data-bound vs compute-bound regimes | **Established** — standard systems analysis |
| Throughput-optimal batch < VRAM-optimal batch | **Derived** — correct inference, not published |
| Dual-constraint budget with probe timing | **Novel** — not found in literature |
| Faster GPU → smaller optimal batch | **Derived** — logical consequence, not published |
