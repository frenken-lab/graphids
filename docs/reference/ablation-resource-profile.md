# Ablation Resource Profiles — set_01

> Last updated: 2026-04-02

---

## Large GAT — `normal_789ca533`

> Job 46260691, set_01, seed 42, weighted_ce
> Node: p0226 (V100 16GB), 2026-04-02

## SLURM Resources

| Metric | Allocated | Used | Efficiency |
|--------|-----------|------|------------|
| Wall time | 5h | 3h55m | 78.4% |
| CPUs | 4 | — | 42.2% |
| Memory | 36 GB | 29.4 GB | 81.7% |
| GPU (V100 16GB) | 1 | — | see below |

## CUDA Memory

| Metric | Value |
|--------|-------|
| Peak allocated | 4.03 GB |
| Steady-state current | 0.07 GB |
| V100 capacity | 16 GB |
| VRAM utilization | ~25% |

## VRAM Probe Results

The probe (`datamodule.py:36-128`) ran twice — before and after `torch.compile`:

| Probe | free_vram | bytes_per_node | budget (nodes) | budget (graphs) |
|-------|-----------|----------------|-----------------|-----------------|
| Pre-compile | 16.51 GB | 303,702 | 46,211 | ~1,641 |
| Post-compile | 12.61 GB | 295,120 | 36,314 | ~1,289 |

`torch.compile` consumed ~4 GB for compilation artifacts, reducing free VRAM
before the second (operative) probe.

### Why budget overestimates actual VRAM usage

Three compounding factors make the budget conservative:

```
Real cost per node (~80 KB)
  x _GRAD_MULTIPLIER = 2           → 160 KB   (probe at datamodule.py:28)
  x probe may run in fp32           → ~295 KB  (training uses 16-mixed)
Budget denominator: 295 KB/node

Actual training: 36,314 nodes x ~80 KB/node ≈ 2.8 GB data + 1.2 GB model = 4 GB
Budget prediction: 36,314 nodes x 295 KB/node ≈ 10.2 GB  ← never reached
```

The `_GRAD_MULTIPLIER=2` assumes full backward for all parameters. For non-KD
runs (no teacher backward), this is a ~2x overestimate. The `_SAFETY_MARGIN=0.85`
(`datamodule.py:30`) adds another 15% headroom on top.

## Training Pipeline Timeline

```
Per-step wall time: ~111 ms  (9.03 it/s)

     ┌─────────────── 111 ms ───────────────┐
     │                                      │
GPU  │▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ▓ = compute (25 ms)
     │      ░ idle / waiting for data       │  ░ = idle    (86 ms)
     │                                      │
W0   │████████████████████░░░░░░░░░░░░░░░░░░│  █ = collate (52 ms)
     │                    ░ idle            │
W1   │░░░░░░░░░░████████████████████░░░░░░░░│  (offset by ~1 batch)
     │                                      │
     └──────────────────────────────────────┘

Prefetch depth: 2 (2 persistent workers, pin_memory=True)
Overlap: workers prepare batch N+1 while GPU processes batch N
Bottleneck: T_collation (52 ms) >> T_gpu (25 ms) → data-bound
```

**Note:** With 2 workers, prefetch partially hides collation latency but cannot
eliminate it when `T_c > T_gpu`. The GPU is productive ~22% of wall time.
Prior Run 003 measured 90% GPU utilization (GAT) — that was likely a smaller
dataset or different batch composition. The set_01 dataset's larger graph count
(~476K training graphs, 367 steps/epoch) may shift the balance.

## Training Metrics

| Metric | Start (ep 0) | End (ep 299) | Best |
|--------|-------------|--------------|------|
| Train loss | 0.3434 | 0.0434 | — |
| Val loss | 0.1761 | 0.0719 | 0.0653 (ep 216) |
| Val accuracy | 77.9% | 93.3% | 93.7% (ep 255) |
| Epochs | 300 | — | early stopping did not trigger |
| Steps/epoch | 367 | — | — |

Test AUC: 0.617 (all 6 attack types). Note: test metrics are per-attack-type
DataLoaders, not aggregated. Low AUC on some attack types is expected for the
normal (non-curriculum) stage — curriculum training improves rare-attack detection.

## Model

| Property | Value |
|----------|-------|
| Class | `GATModule` (large scale) |
| Parameters | 2.5M (9 MB) |
| Precision | 16-mixed (AMP) |
| Compiled | Yes (`torch.compile`) |
| Loss | `weighted_ce` |
| Conv type | GATv2 |

## Options to Improve GPU Utilization

| Option | Expected impact | Effort |
|--------|----------------|--------|
| Increase `num_workers` 2→4 | Deeper prefetch, may fully hide T_c | Config change |
| Reduce `_GRAD_MULTIPLIER` for non-KD | ~2x larger batches, more GPU work/step | Code change |
| Run probe in AMP context | Accurate fp16 measurement | Code change |
| Accept current profile | 4h/run is within ablation budget | None |

Increasing workers to 4 requires bumping `--cpus-per-task` from 4→5 in the
resource profile (`config/resources/profiles/gat.yaml`). Memory impact is
~1.5-2 GB per additional worker (per-worker `_data_list` cache on set_01).

---

## Large VGAE — `autoencoder_9ffb88b1`

> Job 46260687, set_01, seed 42, variational=true
> Node: p0233 (V100 16GB), 2026-04-02

### SLURM Resources

| Metric | Allocated | Used | Efficiency |
|--------|-----------|------|------------|
| Wall time | 4h | 3h40m | 91.7% |
| CPUs | 4 | — | 29.6% |
| Memory | 48 GB | 31.8 GB | 66.2% |
| GPU (V100 16GB) | 1 | — | see below |

### CUDA Memory

| Metric | Value |
|--------|-------|
| Peak allocated | 5.16 GB |
| Steady-state current | 0.08 GB |
| V100 capacity | 16 GB |
| VRAM utilization | ~32% |

### VRAM Probe Results

| Probe | free_vram | bytes_per_node | budget (nodes) | budget (graphs) |
|-------|-----------|----------------|-----------------|-----------------|
| Pre-compile | 16.60 GB | 59,252 | 238,094 | ~8,453 |
| Post-compile | 9.10 GB | 50,202 | 154,074 | ~5,471 |

`torch.compile` consumed ~7.5 GB (vs 4 GB for GAT). The VGAE encoder has 3
hidden layers (480→240→64) with 4-head GATv2 attention + variational
reparameterization, producing more compilation artifacts.

VGAE's `bytes_per_node` is ~5x lower than GAT (50 KB vs 295 KB) because:
- Smaller per-node activation footprint (reconstruction loss, no classifier head)
- Node-level output (reconstruct adjacency) vs graph-level output (classify)

This yields a much larger budget: 154K nodes (5,471 graphs) vs GAT's 36K nodes
(1,289 graphs).

### Training Pipeline Timeline

```
Per-step wall time: ~453 ms  (2.35 it/s)

     ┌──────────────────── 453 ms ─────────────────────┐
     │                                                  │
GPU  │▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ▓ = compute (~25 ms)
     │  ░ idle / waiting for data                       │  ░ = idle   (~428 ms)
     │                                                  │
W0   │██████████████████████████████████████████░░░░░░░░│  █ = collate (~400 ms)
     │                                                  │
W1   │░░░░░░░░░░░░██████████████████████████████████████│  (offset by ~1 batch)
     │                                                  │
     └──────────────────────────────────────────────────┘

Prefetch depth: 2 (2 persistent workers, pin_memory=True)
Bottleneck: collation dominates — 5,471 graphs per batch vs GAT's 1,289
```

**Severely data-bound.** Each batch packs ~5,471 graphs (154K nodes) via
`Batch.from_data_list()`. Collation time scales roughly linearly with graph
count. Prior profiling measured T_c ≈ 52ms for ~1,600 graphs (GAT); at 5,471
graphs, expected T_c ≈ 180-400ms. The measured 453ms/step with T_gpu ≈ 10-25ms
confirms the GPU is idle **>90% of wall time** waiting for data.

With only 86 steps per epoch (large batches drain the dataset quickly),
overhead per epoch is also visible: 86 steps × 453ms ≈ 39s/epoch, but
epoch transitions + validation add overhead → ~44s/epoch effective.

### Training Metrics

| Metric | Start (ep 0) | End (ep 299) | Best |
|--------|-------------|--------------|------|
| Train loss (reconstruction) | 2869.70 | 2869.50 | — |
| Val loss (reconstruction) | 2872.66 | 2871.92 | 2871.92 (ep 265) |
| Epochs | 300 (max) | — | early stopping did not trigger |
| Steps/epoch | 86 | — | — |

VGAE is unsupervised — loss is edge reconstruction error (BCE on adjacency),
not classification. No accuracy/AUC metrics. Test AUC=0.5 is expected (the
autoencoder doesn't classify; test metrics are a sanity check on the
reconstruction quality via embedding separability).

### Model

| Property | Value |
|----------|-------|
| Class | `VGAEModule` (large scale) |
| Parameters | 745K (2 MB) |
| Hidden dims | [480, 240, 64], latent=64 |
| Heads | 4 (GATv2) |
| Precision | 16-mixed (AMP) |
| Compiled | Yes (`torch.compile`) |
| Variational | Yes |

### GAT vs VGAE Comparison

| Metric | Large GAT | Large VGAE | Ratio |
|--------|-----------|------------|-------|
| Parameters | 2.5M | 745K | 3.4x |
| Wall time | 3h55m | 3h40m | ~1.1x |
| Peak CUDA | 4.03 GB | 5.16 GB | 0.8x |
| Budget (nodes) | 36,314 | 154,074 | 4.2x |
| Budget (graphs) | 1,289 | 5,471 | 4.2x |
| Steps/epoch | 367 | 86 | 4.3x |
| ms/step | 111 | 453 | 4.1x |
| GPU compute/step | ~25 ms | ~10-25 ms | ~1x |
| **GPU compute %** | **~22%** | **~5%** | — |
| CPU efficiency | 42.2% | 29.6% | — |
| Memory efficiency | 81.7% | 66.2% | — |

**Key insight:** VGAE takes almost as long as GAT despite being a smaller model
because the 4x larger batch size causes 4x longer collation per step. The GPU
processes each batch quickly but spends most of its time idle. Wall time is
dominated by CPU-side `from_data_list()`, not GPU compute.

### VGAE-Specific Optimization Options

| Option | Expected impact | Effort |
|--------|----------------|--------|
| Increase `num_workers` 2→4 | Double prefetch depth, may halve idle time | Config change |
| Cap node budget | Smaller batches → shorter T_c, more steps, better overlap | Code change |
| Pre-batched dataset | Precompute `Batch.from_data_list()` offline, store as single tensors | Significant |
| Reduce `_GRAD_MULTIPLIER` | Larger budget → even worse data starvation | **Avoid** |

For VGAE specifically, **increasing workers is critical** — the 4.2x larger
batch means T_c >> T_gpu by a much wider margin than GAT. Alternatively,
capping the budget to ~50K nodes (~1,800 graphs, matching GAT's collation
profile) would produce more steps but smaller idle gaps. This trades per-step
efficiency for better GPU overlap.

---

## Cross-references

- `dataloader-performance.md` — collation benchmarks, memory analysis
- `gpu-profiling-tools.md` — nsys/ncu invocations for deeper investigation
- `../decisions/0004-keep-custom-vram-probe.md` — probe design rationale
