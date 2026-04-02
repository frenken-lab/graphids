# Budget Pipeline Analysis: Old vs New

> Theoretical model using profiled constants from set_01 ablation (2026-04-02).
> Real numbers will have overhead (~2x) from spawn workers, GIL, memory allocation.
>
> **Correction (2026-04-02):** original version assumed T_gpu = 25ms constant
> regardless of batch size. This is wrong — T_gpu scales with batch content.
> Both T_collation and T_gpu scale with batch size, so their ratio is roughly
> constant (batch-size-independent). The regime (collation vs compute dominated)
> is determined by model architecture + worker count, not batch size.
> See corrected analysis below.

## Old vs New: How Training Works

### Old: Memory-only budget

```
Trainer.train_dataloader()
  └─ CANBusDataModule._build_loader()
       └─ vram_node_budget()                          [datamodule.py — deleted]
            ├─ cache_metadata.json → mean_nodes
            ├─ torch.cuda.mem_get_info() → free VRAM
            ├─ _probe_bytes_per_node()                [datamodule.py — deleted]
            │    ├─ Batch.from_data_list(~70 graphs)  — untimed
            │    ├─ model.forward(batch)              — untimed, VRAM only
            │    └─ bytes_per_node × _GRAD_MULTIPLIER
            └─ budget = free_vram × 0.85 / bytes_per_node
                     │
              VGAE: 154,074 nodes (5,471 graphs)  ← fills VRAM
              GAT:  36,314 nodes (1,289 graphs)    ← fills VRAM
                     │
       DynamicBatchSampler(max_num=budget)
                     │
       GPU idle 95% (VGAE) — 400ms collation, 25ms compute
       GPU idle 78% (GAT)  — 52ms collation, 25ms compute
```

One question asked: **"What fits in VRAM?"** Answer was always "a lot" for small
models. Monster batches the CPU couldn't deliver fast enough.

### New: Dual-constraint budget

```
Trainer.train_dataloader()
  └─ CANBusDataModule._build_loader()
       ├─ pipeline_depth = num_workers × prefetch_factor   [from hparams]
       └─ node_budget()                                     [budget.py — new]
            ├─ cache_metadata.json → mean_nodes
            ├─ torch.cuda.mem_get_info() → free VRAM
            ├─ probe()                                      [budget.py — new]
            │    ├─ Batch.from_data_list(~70 graphs) ← TIMED → collate_per_graph
            │    ├─ model.forward(batch) warmup
            │    ├─ model.forward(batch)             ← TIMED → gpu_step_s
            │    │                                   ← VRAM measured (same as before)
            │    └─ (bytes_per_node, CostCoefficients)
            │
            ├─ mem_budget = free_vram × 0.85 / bytes_per_node    [same as before]
            ├─ throughput_optimal_graphs()                        [NEW]
            │    = gpu_step × num_workers / collate_per_graph
            │    → max graphs workers can deliver per GPU step
            ├─ throughput_budget = optimal_graphs × mean_nodes
            │
            └─ budget = min(mem_budget, throughput_budget)
                     │
              VGAE: mem=154K, throughput=58K → budget=58K  (throughput binds)
              GAT:  mem=36K,  throughput=210K → budget=36K (memory binds)
                     │
       DynamicBatchSampler(max_num=budget)
                     │
       BudgetResult logged: binding, regime, coefficients
```

Two questions asked, tighter answer wins:
1. **"What fits in VRAM?"** → `mem_budget` (hard ceiling, prevents OOM)
2. **"What can workers deliver before GPU finishes?"** → `throughput_budget`

For GAT (2.5M params): memory still binds. Behavior identical to before.
For VGAE/DGI (745K params): throughput binds. Batch shrinks to match pipeline.

## Effect of Workers on the Two Regimes

### The corrected model

Both T_collation and T_gpu scale with batch size. Their per-node rates are:

```
γ = collation cost per graph             (measured: ~73 µs/graph for VGAE)
β = GPU cost per node                    (measured by two-point probe)
α = GPU kernel overhead per step         (measured: the y-intercept of affine fit)

T_collation(B) = γ × B
T_gpu(B)       = α + β × B × mean_nodes
T_delivery(B)  = T_collation(B) / W = γ × B / W
```

At steady state: `T_step = max(T_delivery, T_gpu)`.

The regime is determined by the per-graph rates, NOT batch size:

```
γ/W  vs  β × mean_nodes

γ/W > β·mean_nodes → collation-dominated (for ALL batch sizes)
γ/W < β·mean_nodes → compute-dominated (for ALL batch sizes)
```

Adding workers (increasing W) moves you from collation-dominated toward
compute-dominated. Changing batch size does NOT change the regime.

### VGAE — Collation-dominated regime

Using profiled constants: γ = 73 µs/graph, mean_nodes = 28.2.
β and α from two-point probe (not yet measured — values below are illustrative).

With old budget (fill VRAM = 5,471 graphs), adding workers reduces T_delivery
but T_gpu also scales with the large batch:

| Workers | T_delivery | T_gpu (affine) | T_step | GPU util |
|---------|-----------|---------------|--------|----------|
| 2 | 200 ms | α + β·154K | max(200, T_gpu) | T_gpu/T_step |
| 6 | 67 ms | α + β·154K | max(67, T_gpu) | T_gpu/T_step |

The absolute GPU utilization depends on measured α and β. What we know from
profiling: at 5,471 graphs (154K nodes), T_step = 453ms and T_gpu ≈ 25ms
with 2 workers. This means T_delivery ≈ 200ms dominated T_step after overhead.

**The throughput budget (from the affine model) caps the batch** at the point
where T_delivery = T_gpu. This is a smaller batch that both the pipeline and GPU
can process at matched rates, eliminating GPU starvation. The exact size depends
on measured α and β — the probe computes this at runtime.

### GAT — Compute-dominated regime

GAT's 2.5M params produce a large enough β that γ/W < β·mean_nodes even at
2 workers. Memory binds for all practical worker counts. The budget system
correctly returns mem_budget and binding="memory".

### What workers actually change

Workers change the **regime**, not the batch size. The correct mental model:

| Workers | γ/W vs β·mean | Regime | Budget decision |
|---------|-------------|--------|-----------------|
| 2 (VGAE) | γ/2 >> β·28.2 | collation-dominated | throughput_budget < mem_budget |
| 6 (VGAE) | γ/6 > β·28.2 | still collation-dominated | throughput_budget < mem_budget |
| ~15+ (VGAE) | γ/15 ≈ β·28.2 | balanced → compute | mem_budget binds |
| 2 (GAT) | γ/2 ≈ β·28.2 | borderline | depends on probe |
| 3+ (GAT) | γ/3 < β·28.2 | compute-dominated | mem_budget binds |

Throughput scales with workers because more workers = higher CPU parallelism =
more nodes/sec through the system. This is true regardless of budget system.
The budget system just ensures the GPU isn't starved while this happens.

## Key Takeaways

**1. The regime is a system property, not a batch size knob.**
Both T_collation and T_gpu scale with batch size. Their ratio is determined by
model architecture, graph structure, and worker count. You can't "optimize out"
a collation bottleneck by choosing a different batch size — you need more workers
(or faster collation, or CPU training).

**2. The affine GPU model (α + β·N) creates a finite optimal batch.**
The kernel overhead α makes very small batches wasteful (high overhead fraction).
The throughput budget balances amortizing α against pipeline delivery rate.
This exists ONLY in the collation-dominated regime with measurable α.

**3. More workers is the real fix for collation-dominated models.**
Workers change γ/W, moving you toward the compute-dominated regime where the
budget system correctly says "fill VRAM." The budget system is a guard that
prevents GPU starvation while you're still collation-dominated.

**4. GAT is already near the boundary at 2 workers.**
The profiled 52ms collation vs 25ms GPU (at the memory-limited batch) puts GAT
close to balanced. 3 workers tips it into compute-dominated. No budget system
change needed — memory is already the correct ceiling.

**5. Actual utilization requires measured α and β.**
The two-point probe in budget.py measures these at runtime. Until validated on
GPU (needs SLURM job), the exact budget numbers are estimates. The regime
classification and directional conclusions are robust to measurement error.
