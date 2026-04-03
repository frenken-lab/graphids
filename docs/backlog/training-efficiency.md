# Training Efficiency — CPU-Bound DataLoader Bottleneck

> Sources: `docs/reference/ablation-resource-profile.md`, `docs/guides/cpu-gpu-gnn-training.md`

## Diagnosis

GPU utilization is 5-22% across all graph stages. The bottleneck is CPU-side
`Batch.from_data_list()` collation, not GPU compute.

| Model | T_collation | T_gpu | Ratio | GPU active % |
|-------|------------|-------|-------|-------------|
| GAT (large) | 52 ms | 25 ms | 2:1 | 22% |
| VGAE (large) | ~400 ms | 10-25 ms | 16:1 | 5% |

These are **two structurally different problems** requiring different fixes.
GAT's ratio is closable with pipeline depth. VGAE's ratio means the batch is
fundamentally too large for the compute — no amount of prefetching fixes a 16:1
collation-to-compute ratio.

Root cause for VGAE: `vram_node_budget` optimizes for "fill GPU VRAM" when
the real constraint is "collation time the pipeline can hide." The budget should
be set by throughput, not memory capacity.

## Done (2026-04-02)

### Workers 2 → 6, SLURM CPUs 4 → 8

- Stage YAMLs: `num_workers: 6` in all 4 stages
- Resource profiles: `cpus: 8, workers: 6` across GAT, VGAE, DGI, temporal
- Memory bumped for extra worker RSS: GAT 36→40G, VGAE large 48→52G, etc.
- Temporal: 24→28G (small), 36→40G (large)

**Assessment:** correct direction for GAT (3 workers would suffice to hide 2:1
ratio, 6 gives headroom). For VGAE, this alone won't fix the 16:1 ratio —
budget cap (below) is required. The number 6 was chosen as a general-purpose
default across stages, not derived per-model.

### Cluster memory limits in config

- `clusters.yaml`: added `mem_per_cpu` (MB) per execution mode for all 3 clusters
- `slurm/resources.py`: validates `mem <= cpus × mem_per_cpu` at profile resolution
- Fixed Cardinal partition: `batch` → `gpu`
- Reference: `docs/reference/osc-cluster-memory-limits.md`

**Finding:** Ascend `nextgen` has only 4,027 MB/CPU. Our 52G profiles fail at
8 CPUs (ceiling 31.5G). Needs 14 CPUs or `quad` partition on Ascend.

## Next — Principled fixes

### 1. Cap VGAE/DGI node budget (addresses the 16:1 ratio)

This is the real fix for autoencoder GPU waste. `vram_node_budget` returns 154K
nodes (5,471 graphs) for VGAE because the model is small. But collation of 5K
graphs takes ~400ms while GPU compute takes ~25ms — the batch is too large for
the pipeline to absorb.

**Principle:** batch size should be set by `T_collation ≈ T_gpu × pipeline_depth`,
not by VRAM capacity. With 6 workers × prefetch_factor=2, pipeline depth = 12.
Target T_collation ≤ 12 × T_gpu = 300ms. Collation scales linearly with graph
count, so:

```
current:  5,471 graphs → ~400 ms collation
target:   ~3,500 graphs → ~260 ms collation  (within 300ms pipeline window)
          → ~100K nodes (at 28.2 nodes/graph mean)
```

Implementation: add `max_node_budget: int | None = None` to
`CANBusDataModule.__init__`, apply `budget = min(budget, max_node_budget)` in
`_build_loader`. Set in autoencoder stage YAML. GAT doesn't need a cap — its
1,289-graph batches already produce collation within pipeline tolerance.

**Validate:** compare steps/sec and GPU util before/after on a short run.

### 2. Add prefetch_factor parameter

`make_graph_loader` uses PyTorch's default `prefetch_factor=2`. Increasing to 4
doubles pipeline depth: 6 workers × 4 = 24 batches queued.

- Add `prefetch_factor: int = 2` to `CANBusDataModule.__init__`
- Pass through in `make_graph_loader`
- Set `prefetch_factor: 4` in stage YAMLs

**Caveat:** each prefetched batch consumes worker RSS. With VGAE's current
5,471-graph batches, `prefetch_factor=4` would queue ~22K graphs per worker —
significant memory pressure. Budget cap (above) must land first to make
prefetch_factor safe to increase.

### 3. Per-model worker count (optional refinement)

Current: all stages use `num_workers: 6`. But GAT only needs ~3 workers to hide
its 2:1 ratio. VGAE (after budget cap) might need 4-6. Making `num_workers`
stage-specific would right-size CPU allocations per model.

Not urgent — 6 workers is safe for all models. Revisit if CPU allocation pressure
becomes a problem on Ascend (where CPUs are shared with more jobs).

## Deferred — Evaluate for next campaign

### CPU-only training for autoencoders

VGAE: 745K params, 5% GPU util. Even after budget cap, GPU compute is trivial.
CPU training eliminates GPU queue contention and saves ~24 GPU-hours per campaign.

Requires spike: measure CPU forward+backward throughput for VGAE to verify wall
time doesn't regress more than ~30%. See `docs/guides/cpu-gpu-gnn-training.md`
for the analysis framework.

Prerequisites: `cpu_train` execution mode, CPU-aware budget in `vram_node_budget`,
disable `torch.compile` + AMP on CPU.

## What NOT to change

- **GAT/curriculum stays on GPU.** 2.5M params benefits from GPU. 2:1 ratio
  is already closable with current worker count.
- **Fusion stays on GPU.** `num_workers: 0`, cached state vectors. No collation.
- **Don't reduce `_GRAD_MULTIPLIER` for VGAE.** Larger budget worsens the
  collation problem. The fix is a budget *cap*, not a more aggressive probe.
- **Don't add graph-size bucketing yet.** Budget cap + workers address the
  acute bottleneck. Bucketing is incremental polish.

## Execution order

| Step | What | Effort | Blocks on |
|------|------|--------|-----------|
| 1 | Cap VGAE/DGI node budget | ~5 lines + YAML | Nothing — do first |
| 2 | Add prefetch_factor | ~10 lines + YAML | Step 1 (memory safety) |
| 3 | Per-model worker count | YAML only | Profile data from steps 1-2 |
| 4 | CPU training spike | ~50 lines + config | Steps 1-2 validated |

Steps 1-2 can be validated together in one `hcrl_sa` smoke test comparing
steps/sec and GPU util against the profiled baseline (9.03 it/s GAT, 2.35 it/s VGAE).
