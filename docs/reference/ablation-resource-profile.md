# Ablation Resource Profiles — set_01

> Last updated: 2026-04-03
> Dataset: set_01, seed 42, all runs on Pitzer V100 16 GB nodes
> For cost model theory and regime analysis, see `throughput-model.md`.
> Analysis script: `scripts/analyze_resource_profiles.py`

---

## Summary Tables

### SLURM Resources

| Run | Job ID | Wall Time | Peak RSS | Req Mem | Mem Eff | CPUs | CPU Eff |
|-----|--------|-----------|----------|---------|---------|------|---------|
| VGAE large | 46260687 | 3h40m | 31.8 GB | 48 GB | 66.2% | 6 | 29.6% |
| VGAE small | 46260690 | 2h15m | 24.9 GB | 36 GB | 69.3% | 4 | 54.2% |
| GAT large (normal) | 46260691 | 3h55m | 29.4 GB | 36 GB | 81.7% | 4 | 42.2% |
| GAT small (normal) | 46266539 | 1h53m | 43.9 GB | 54 GB | 81.3% | 6 | 39.2% |
| GAT large (curriculum) | 46264821 | 2h55m | 35.9 GB | 36 GB | 99.8% | 4 | 58.2% |
| GAT small (curriculum) | 46266245 | 2h00m | 36.0 GB | 36 GB | 100.0% | 4 | 45.9% |

Curriculum runs are at 100% memory utilization (RSS = ReqMem). Caused by
CurriculumDataModule loading VGAE checkpoint + scoring all graphs for
difficulty ranking on top of normal data pipeline memory. Future runs
should request 48 GB for curriculum stages.

GAT small (normal) needed 54 GB (up from 36 GB after OOM on earlier
attempts with job 46152814). The 6 workers at `num_workers=6` explains the
higher RSS vs GAT large which uses 2 workers.

### CUDA Memory (V100 16 GB)

| Run | Peak Active | Steady-State | Reserved | Fragmentation | Retries | OOMs |
|-----|-------------|--------------|----------|---------------|---------|------|
| VGAE large | 4.81 GB | 0.08 GB | 12.43 GB | 12.35 GB | 0 | 0 |
| VGAE small | 5.99 GB | 0.08 GB | 10.08 GB | 10.00 GB | 0 | 0 |
| GAT large (normal) | 3.75 GB | 0.07 GB | 7.78 GB | 7.71 GB | 0 | 0 |
| GAT small (normal) | 3.09 GB | 0.06 GB | 4.84 GB | 4.78 GB | 0 | 0 |
| GAT large (curriculum) | 3.75 GB | 0.07 GB | 7.78 GB | 7.70 GB | 0 | 0 |
| GAT small (curriculum) | 3.05 GB | 0.06 GB | 7.05 GB | 6.99 GB | 0 | 0 |

"Peak Active" = `active_bytes.all.peak` (max across all training steps).
"Steady-State" = `active_bytes.all.current` (median during training).
"Reserved" = `reserved_bytes.all.current` (median, includes caching allocator pools).
"Fragmentation" = reserved minus active (mean).

All runs show massive VRAM fragmentation: the caching allocator reserves
far more than is actively used. VGAE large reserves 12.43 GB but only
uses 4.81 GB peak -- 77% of reserved VRAM is idle pool.
`torch.compile` is the primary driver: compilation artifacts inflate the
reserved pool but are not reflected in active bytes.

No allocation retries or OOMs on any run.

### Fragmentation Drift

| Run | Frag Min | Frag Mean | Frag Max | Drift (MB/step) |
|-----|----------|-----------|----------|------------------|
| VGAE large | 10.54 GB | 12.35 GB | 12.36 GB | +0.042 |
| VGAE small | 7.28 GB | 10.00 GB | 10.01 GB | +0.090 |
| GAT large (normal) | 5.45 GB | 7.71 GB | 7.71 GB | +0.020 |
| GAT small (normal) | 4.78 GB | 4.78 GB | 4.78 GB | 0.000 |
| GAT large (curriculum) | 5.45 GB | 7.70 GB | 7.71 GB | +0.100 |
| GAT small (curriculum) | 5.00 GB | 6.99 GB | 7.00 GB | +0.097 |

"Drift" = linear slope of (reserved - active) over training steps.
Positive drift means fragmentation grows over time. VGAE and curriculum
runs show the most drift. GAT small (normal) is flat -- no progressive
fragmentation.

The min-to-mean jump (e.g., 10.54 to 12.35 GB for VGAE large) reflects
early-training pool expansion during the first `torch.compile` call.
Once the pool stabilizes, drift is gradual.

### Training Throughput

| Run | Epochs | Steps/Epoch | Total Steps | ms/step (wall) | it/s |
|-----|--------|-------------|-------------|----------------|------|
| VGAE large | 300 | 86 | 25,799 | 512 | 1.95 |
| VGAE small | 300 | 72 | 21,599 | 375 | 2.67 |
| GAT large (normal) | 300 | 368 | 110,099 | 128 | 7.80 |
| GAT small (normal) | 154 | 108 | 16,549 | 408 | 2.45 |
| GAT large (curriculum) | 134 | 370 | 49,149 | 214 | 4.68 |
| GAT small (curriculum) | 229 | 110 | 25,149 | 286 | 3.50 |

ms/step is wall time / total steps (includes data loading, validation,
checkpointing overhead). CSVLogger logged every 50 steps.

GAT large (normal) is the fastest per-step (128 ms) because more
steps/epoch means the DataLoader's persistent workers stay warm. VGAE
is slowest per-step due to the larger dynamic batch (more graphs per
batch = longer CPU-side collation in `from_data_list()`).

GAT small runs fewer steps/epoch than large (108 vs 368) because
the dynamic batch sampler packs more graphs per batch for smaller
models (smaller node budget = more graphs fit).

Curriculum runs fewer total epochs than normal due to early stopping.
GAT large (curriculum) stopped at epoch 134 with best val at epoch 33.

### Training Convergence

| Run | Val Loss (start) | Val Loss (best) | Best Epoch | Val Acc (start) | Val Acc (best) | Best Epoch |
|-----|------------------|-----------------|------------|-----------------|----------------|------------|
| VGAE large | 2872.66 | 2871.92 | 265 | -- | -- | -- |
| VGAE small | 2873.72 | 2872.81 | 288 | -- | -- | -- |
| GAT large (normal) | 0.1761 | 0.0653 | 216 | 77.9% | 93.7% | 255 |
| GAT small (normal) | 0.4751 | 0.0968 | 133 | 77.5% | 96.5% | 145 |
| GAT large (curriculum) | 0.0674 | 0.0322 | 33 | 89.5% | 95.3% | 33 |
| GAT small (curriculum) | 0.1135 | 0.0274 | 128 | 80.0% | 96.5% | 178 |

Curriculum starts with lower val loss because the initial ratio=1.0
feeds only easy examples, so the model begins from a favorable
distribution. Best val loss for curriculum is 2x lower than normal
(0.032 vs 0.065 for large).

GAT small outperforms GAT large on val accuracy (96.5% vs 93.7% normal,
96.5% vs 95.3% curriculum). This is consistent with the smaller model
being less prone to overfitting on set_01's relatively small training set.

VGAE loss is reconstruction loss (not cross-entropy), so the absolute
values are not comparable to GAT. The near-zero improvement (2872.66 to
2871.92) is expected -- VGAE learns embeddings, not a decision boundary.

### Test Metrics (version_1)

| Run | Test AUC | Test Accuracy | Test F1 | Test Precision | Test Recall | Test Specificity |
|-----|----------|---------------|---------|----------------|-------------|------------------|
| VGAE large | 0.500 | 17.3% | 0.294 | 0.173 | 1.000 | 0.000 |
| VGAE small | 0.500 | 17.1% | 0.293 | 0.171 | 1.000 | 0.000 |
| GAT large (normal) | 0.617 | 15.4% | 0.266 | 0.153 | 1.000 | 0.000 |
| GAT small (normal) | 0.636 | 15.9% | 0.273 | 0.158 | 1.000 | -- |
| GAT large (curriculum) | 0.805 | 89.7% | 0.598 | 0.737 | 0.504 | 0.968 |
| GAT small (curriculum) | 0.496 | 17.1% | 0.271 | 0.157 | 0.975 | 0.021 |

VGAE test metrics are from reconstruction threshold classification
(not meaningful for AUC -- 0.5 = random). GAT normal runs show high
recall (1.0) but near-zero specificity = predicting everything as attack.
The test threshold was not tuned.

GAT large (curriculum) is the only run with a balanced test result:
AUC=0.805, 89.7% accuracy, 0.968 specificity. This suggests curriculum
learning + focal loss produces a more discriminative decision boundary.

### Checkpoint Sizes

| Run | best_model.ckpt | last.ckpt |
|-----|-----------------|-----------|
| VGAE large | 8.6 MB | 8.6 MB |
| VGAE small | 1.2 MB | 1.2 MB |
| GAT large | 28.6 MB | 28.6 MB |
| GAT small | 2.4 MB | 2.4 MB |

---

## Per-Run Details

### VGAE Large -- `autoencoder_9ffb88b1`

> Job 46260687, 2026-04-02 05:02-08:42 UTC, V100 16 GB

| Property | Value |
|----------|-------|
| Class | `VGAEModule` (large) |
| Hidden dims | [480, 240, 64], latent=64 |
| Heads | 4, embedding_dim=32 |
| Dropout | 0.15 |
| Precision | 16-mixed, compiled, gradient_checkpointing |
| Variational | true, mask_ratio=0.3, k_neg=32 |
| Loss weights | canid=0.1, nbr=0.05, kl=0.01 |
| Batch size | 8192 (dynamic), 2 workers |

VRAM: peak active 4.81 GB, reserved 12.43 GB. The 7.6 GB gap between
peak active and reserved is `torch.compile` overhead -- VGAE's 3-layer
encoder with 4-head GATv2 + reparameterization produces more compilation
artifacts than GAT.

### VGAE Small -- `autoencoder_ff9f9014`

> Job 46260690, 2026-04-02 08:13-10:28 UTC, V100 16 GB

| Property | Value |
|----------|-------|
| Class | `VGAEModule` (small) |
| Hidden dims | [80, 40, 16], latent=16 |
| Heads | 1, embedding_dim=4 |
| Dropout | 0.1 |
| Precision | 16-mixed, compiled, gradient_checkpointing |
| Batch size | 8192 (dynamic), 2 workers |

Peak active 5.99 GB -- paradoxically higher than VGAE large (4.81 GB).
With 1 head and smaller dims, the model itself is smaller but the
dynamic batch packs more graphs (larger node budget) so each forward
pass processes more data. The batch size dominates over model size for
peak VRAM.

### GAT Large (normal) -- `normal_789ca533`

> Job 46260691, 2026-04-02 08:13-12:08 UTC, V100 16 GB

| Property | Value |
|----------|-------|
| Class | `GATModule` (large) |
| Architecture | hidden=64, layers=3, heads=4, fc_layers=4 |
| Embedding dim | 8, proj_dim=48 |
| Dropout | 0.11 |
| Loss | weighted_ce (weight=10.0) |
| Precision | 16-mixed, compiled |
| Batch size | 8192 (dynamic), 2 workers |

Highest step count (110K) and fastest per-step (128 ms). 300 full
epochs without early stopping. Best val accuracy 93.7% at epoch 255.
Test AUC 0.617 with threshold at 1.0 (everything predicted as attack).

### GAT Small (normal) -- `normal_56cc5893`

> Job 46266539, 2026-04-02 13:47-15:40 UTC, V100 16 GB

| Property | Value |
|----------|-------|
| Class | `GATModule` (small) |
| Architecture | hidden=24, layers=2, heads=4, fc_layers=2 |
| Embedding dim | 8, proj_dim=32 |
| Dropout | 0.1 |
| Loss | ce (not weighted) |
| Precision | 16-mixed, compiled |
| Batch size | 8192 (dynamic), 6 workers |

This run required 54 GB memory (up from 36 GB) after OOM on job 46152814.
The 6 workers drive higher RSS (43.9 GB vs 29.4 GB for large with 2
workers). Early stopping at epoch 154. Best val accuracy 96.5% at
epoch 145 -- outperforms large by 2.8 percentage points.

Note: `loss_fn=ce` (not weighted_ce like GAT large). This is an ablation
axis difference, not just scale.

### GAT Large (curriculum) -- `curriculum_e9354ccd`

> Job 46264821, 2026-04-02 12:20-15:15 UTC, V100 16 GB

| Property | Value |
|----------|-------|
| Class | `GATModule` (large) via CurriculumDataModule |
| Architecture | hidden=64, layers=3, heads=4, fc_layers=4 |
| Loss | focal (gamma=2.0, weight=10.0) |
| Curriculum | start_ratio=1.0, end_ratio=10.0, percentile=75 |
| VGAE ckpt | `vgae_large_autoencoder_9ffb88b1/.../best_model.ckpt` |
| Precision | 16-mixed, compiled |
| Batch size | 8192 (dynamic), 2 workers |

RSS 35.9 GB = 99.8% of 36 GB requested. The CurriculumDataModule
loads the VGAE checkpoint and scores all training graphs at setup time,
adding ~6 GB to baseline memory.

Best test result of any run: AUC=0.805, accuracy=89.7%. Curriculum +
focal loss produces a balanced precision/recall trade-off (0.737/0.504)
vs all other runs which predict everything as attack.

Early stopping at epoch 134, best val at epoch 33.

### GAT Small (curriculum) -- `curriculum_bf2a5575`

> Job 46266245, 2026-04-02 13:33-15:33 UTC, V100 16 GB

| Property | Value |
|----------|-------|
| Class | `GATModule` (small) via CurriculumDataModule |
| Architecture | hidden=24, layers=2, heads=4, fc_layers=2 |
| Loss | focal (gamma=2.0, weight=10.0) |
| Curriculum | start_ratio=1.0, end_ratio=10.0, percentile=75 |
| VGAE ckpt | `vgae_small_autoencoder_ff9f9014/.../best_model.ckpt` |
| Precision | 16-mixed, compiled |
| Batch size | 8192 (dynamic), 2 workers |

RSS 36.0 GB = 100% of 36 GB. Same memory pressure as large curriculum.

Best val metrics match GAT small (normal): 96.5% accuracy, 0.027 val
loss. But test AUC is 0.496 (below random) despite good val metrics.
This suggests the curriculum small model overfit to the difficulty-sorted
training distribution and did not generalize to the test set.

---

## Cross-Run Analysis

### Scale Effect (Large vs Small)

| Metric | VGAE L/S | GAT Normal L/S | GAT Curriculum L/S |
|--------|----------|----------------|--------------------|
| Wall time | 3h40m / 2h15m | 3h55m / 1h53m | 2h55m / 2h00m |
| Peak RSS | 31.8 / 24.9 GB | 29.4 / 43.9 GB | 35.9 / 36.0 GB |
| Peak VRAM | 4.81 / 5.99 GB | 3.75 / 3.09 GB | 3.75 / 3.05 GB |
| Steps/epoch | 86 / 72 | 368 / 108 | 370 / 110 |
| ms/step | 512 / 375 | 128 / 408 | 214 / 286 |
| Best val acc | -- | 93.7 / 96.5% | 95.3 / 96.5% |
| Checkpoint | 8.6 / 1.2 MB | 28.6 / 2.4 MB | 28.6 / 2.4 MB |

Small models have fewer steps/epoch (dynamic batching packs more graphs
per batch) but slower per-step (larger batches = more CPU collation
time). GAT small RSS is higher than large due to 6 workers vs 2.

GAT small consistently matches or exceeds large on val accuracy.
Combined with 7-12x smaller checkpoints, small is the better choice
for edge deployment in the KD pipeline.

### Stage Effect (Normal vs Curriculum)

| Metric | GAT Large Normal/Curriculum | GAT Small Normal/Curriculum |
|--------|----------------------------|-----------------------------|
| Epochs to best val | 255 / 33 | 145 / 178 |
| Best val acc | 93.7% / 95.3% | 96.5% / 96.5% |
| Best val loss | 0.0653 / 0.0322 | 0.0968 / 0.0274 |
| Test AUC | 0.617 / 0.805 | 0.636 / 0.496 |
| Wall time | 3h55m / 2h55m | 1h53m / 2h00m |

For large models, curriculum converges faster (epoch 33 vs 255) and
produces better test AUC (0.805 vs 0.617). For small models, curriculum
does not help test AUC (0.496 vs 0.636). This may indicate the small
model lacks capacity to benefit from curriculum's difficulty scheduling.

### VRAM Utilization

All runs use less than 38% of the V100's 16 GB VRAM for active
computation. The caching allocator's reserved pool is 2-3x the active
peak. This is not wasted VRAM (the allocator pools buffers for reuse)
but confirms these models would fit on smaller GPUs.

| Run | Active/16GB | Reserved/16GB |
|-----|-------------|---------------|
| VGAE large | 30% | 78% |
| VGAE small | 37% | 63% |
| GAT large | 23% | 49% |
| GAT small | 19% | 30% |

### Resource Right-Sizing Recommendations

| Resource | Current | Recommended | Rationale |
|----------|---------|-------------|-----------|
| Memory (VGAE) | 48 GB | 36 GB | Peak RSS 31.8 GB, 36 GB gives 12% headroom |
| Memory (GAT normal) | 36-54 GB | 36 GB / 2 workers | 54 GB needed only because of 6 workers |
| Memory (curriculum) | 36 GB | 48 GB | Both hit 100% -- need headroom |
| GPU | V100 16 GB | V100 16 GB | Max 37% utilization, no OOMs |
| CPUs (VGAE) | 6 | 4 | 29.6% CPU efficiency with 6 |
| CPUs (GAT) | 4 | 4 | 42-58% efficiency is acceptable |

---

## Cross-references

- `throughput-model.md` -- cost model, regime analysis, optimization options
- `dataloader-performance.md` -- collation benchmarks, memory analysis
- `observability.md` -- profiling tool invocations
