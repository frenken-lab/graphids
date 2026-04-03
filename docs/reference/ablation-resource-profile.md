# Ablation Resource Profiles — set_01

> Last updated: 2026-04-02
> For cost model theory and regime analysis, see `throughput-model.md`.

---

## Large GAT — `normal_789ca533`

> Job 46260691, set_01, seed 42, weighted_ce, node p0226 (V100 16GB)

### SLURM Resources

| Metric | Allocated | Used | Efficiency |
|--------|-----------|------|------------|
| Wall time | 5h | 3h55m | 78.4% |
| CPUs | 4 | — | 42.2% |
| Memory | 36 GB | 29.4 GB | 81.7% |
| GPU (V100 16GB) | 1 | — | see below |

### CUDA Memory

| Metric | Value |
|--------|-------|
| Peak allocated | 4.03 GB |
| Steady-state current | 0.07 GB |
| VRAM utilization | ~25% (of 16 GB) |

### VRAM Probe

| Probe | free_vram | bytes_per_node | budget (nodes) | budget (graphs) |
|-------|-----------|----------------|-----------------|-----------------|
| Pre-compile | 16.51 GB | 303,702 | 46,211 | ~1,641 |
| Post-compile | 12.61 GB | 295,120 | 36,314 | ~1,289 |

`torch.compile` consumed ~4 GB for compilation artifacts, reducing free VRAM.

### Training Pipeline

```
Per-step: ~111 ms  (9.03 it/s)

GPU  │▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ▓ = compute (25 ms)
W0   │████████████████████░░░░░░░░░░░░░░░░░░│  █ = collate (52 ms)
W1   │░░░░░░░░░░████████████████████░░░░░░░░│  (offset by ~1 batch)

Bottleneck: T_c (52 ms) >> T_gpu (25 ms) → data-bound, GPU ~22% productive
```

### Training Metrics

| Metric | Start (ep 0) | End (ep 299) | Best |
|--------|-------------|--------------|------|
| Val loss | 0.1761 | 0.0719 | 0.0653 (ep 216) |
| Val accuracy | 77.9% | 93.3% | 93.7% (ep 255) |
| Test AUC | — | 0.617 | (per-attack-type, not aggregated) |

300 epochs, 367 steps/epoch, early stopping did not trigger.

### Model

| Property | Value |
|----------|-------|
| Class | `GATModule` (large) |
| Parameters | 2.5M |
| Precision | 16-mixed, compiled |
| Loss | `weighted_ce`, conv GATv2 |

---

## Large VGAE — `autoencoder_9ffb88b1`

> Job 46260687, set_01, seed 42, variational=true, node p0233 (V100 16GB)

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
| VRAM utilization | ~32% (of 16 GB) |

### VRAM Probe

| Probe | free_vram | bytes_per_node | budget (nodes) | budget (graphs) |
|-------|-----------|----------------|-----------------|-----------------|
| Pre-compile | 16.60 GB | 59,252 | 238,094 | ~8,453 |
| Post-compile | 9.10 GB | 50,202 | 154,074 | ~5,471 |

`torch.compile` consumed ~7.5 GB (vs 4 GB for GAT — VGAE's 3-layer encoder with 4-head GATv2 + reparameterization produces more compilation artifacts).

### Training Pipeline

```
Per-step: ~453 ms  (2.35 it/s)

GPU  │▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ▓ = compute (~25 ms)
W0   │██████████████████████████████████████████░░░░░░░░│  █ = collate (~400 ms)
W1   │░░░░░░░░░░░░██████████████████████████████████████│  (offset by ~1 batch)

Bottleneck: 5,471 graphs/batch → T_c ~400ms >> T_gpu ~25ms → GPU idle >90%
```

### Training Metrics

| Metric | Start (ep 0) | End (ep 299) | Best |
|--------|-------------|--------------|------|
| Val loss (recon) | 2872.66 | 2871.92 | 2871.92 (ep 265) |

300 epochs, 86 steps/epoch, unsupervised (no accuracy/AUC).

### Model

| Property | Value |
|----------|-------|
| Class | `VGAEModule` (large) |
| Parameters | 745K |
| Hidden dims | [480, 240, 64], latent=64, 4 heads |
| Precision | 16-mixed, compiled, variational |

---

## GAT vs VGAE Comparison

| Metric | Large GAT | Large VGAE | Ratio |
|--------|-----------|------------|-------|
| Parameters | 2.5M | 745K | 3.4x |
| Wall time | 3h55m | 3h40m | ~1.1x |
| Peak CUDA | 4.03 GB | 5.16 GB | 0.8x |
| Budget (nodes) | 36,314 | 154,074 | 4.2x |
| Steps/epoch | 367 | 86 | 4.3x |
| ms/step | 111 | 453 | 4.1x |
| **GPU compute %** | **~22%** | **~5%** | — |

VGAE takes almost as long as GAT despite being smaller because the 4x larger batch causes 4x longer collation. Wall time is dominated by CPU-side `from_data_list()`.

---

## Cross-references

- `throughput-model.md` — cost model, regime analysis, optimization options
- `dataloader-performance.md` — collation benchmarks, memory analysis
- `observability.md` — profiling tool invocations
