# Training Efficiency — CPU-Bound DataLoader Bottleneck

> Workers bumped to 6, SLURM CPUs to 8, cluster mem limits validated (2026-04-02).
> See git history for details.

## Diagnosis

GPU utilization is 5-22%. Bottleneck is CPU-side `Batch.from_data_list()` collation.
GAT: 2:1 collation-to-GPU ratio (closable with pipeline depth).
VGAE: 16:1 ratio (batch too large for compute — budget cap needed).

## Next — Principled fixes

### 1. Cap VGAE/DGI node budget (addresses the 16:1 ratio)

`vram_node_budget` returns 154K nodes (5,471 graphs) for VGAE. Collation of 5K
graphs takes ~400ms while GPU takes ~25ms. Target: ~3,500 graphs (~100K nodes)
to bring collation within 300ms pipeline window (6 workers × prefetch 2 × T_gpu).

Add `max_node_budget: int | None = None` to `CANBusDataModule.__init__`, apply
`budget = min(budget, max_node_budget)` in `_build_loader`. Set in autoencoder YAML.

### 2. Add prefetch_factor parameter

Default `prefetch_factor=2`. Add to `CANBusDataModule.__init__`, pass through
`make_graph_loader`, set to 4 in stage YAMLs. **Must land after step 1** (memory).

### 3. Per-model worker count (optional refinement)

GAT only needs ~3 workers. VGAE after cap might need 4-6. Stage-specific
`num_workers` in YAML would right-size CPU allocations. Low priority.

## Deferred — CPU-only training for autoencoders

VGAE: 745K params, 5% GPU util. CPU training eliminates GPU queue contention.
Requires spike: measure CPU throughput, verify wall time doesn't regress >30%.
Prerequisites: `cpu_train` execution mode, CPU-aware budget, disable compile+AMP.

## What NOT to change

- GAT/curriculum stays on GPU (2.5M params, 2:1 ratio closable).
- Fusion stays on GPU (no collation, cached state vectors).
- Don't reduce `_GRAD_MULTIPLIER` for VGAE (worsens collation problem).
- Don't add graph-size bucketing yet (incremental polish after cap).
