# Ablation Run 001 — Training Efficiency Issues

> Researched: 2026-03-24

## Context

Ablation run 001 (submitted 2026-03-24 00:17) exposed three efficiency issues across 69 jobs. This plan addresses:

1. **VRAM underutilization** — non-GPS small models peak at 4-6 GB on 16 GB V100 (33-42%)
2. **GPS OOM** — GPS conv_type jobs attempted 105-169 GB allocations and crashed immediately
3. **Data staging bottleneck** — `cp -r` from scratch to TMPDIR consumed the full 30-min wall for CPU eval jobs

Prior plan (`~/plans/gpu-utilization-and-training-efficiency.md`, 2026-03-23) addressed `safety_factor`-based batch sizing. That factor no longer exists — it was replaced by `DynamicBatchSampler` with a p95-based node budget (`compute_node_budget()` in `data_loading.py:31-57`). The issues below are specific to that new mechanism.

## Issue 1: VRAM Underutilization — Is p95 Too Conservative?

### Current mechanism

`compute_node_budget()` computes: `budget = batch_size * p95_nodes`

For set_01: `batch_size=4096, p95=38 nodes → budget=155,648 nodes`
For set_02: `batch_size=4096, p95=45 nodes → budget=184,320 nodes`

`DynamicBatchSampler` then greedily packs graphs until their cumulative node count hits the budget. Since most graphs are near the **mean** (32 or 38 nodes), not p95, the typical batch packs: `155,648 / 32 ≈ 4,864 graphs` (set_01) or `184,320 / 38 ≈ 4,851 graphs` (set_02). The budget is headroom for the rare batch that draws many large graphs.

### Why p95 wastes VRAM

The p95 budget means 95% of batches use **less** than the budget. The actual distribution matters:

| Dataset | min | mean | median | p95 | p99 | max |
|---------|-----|------|--------|-----|-----|-----|
| set_01  | ?   | 32.0 | 32     | 38  | (in metadata) | 50 |
| set_02  | ?   | 38.3 | 37     | 45  | (in metadata) | 78 |

The coefficient of variation is low: `std/mean` is roughly `(p95 - mean) / (1.645 * mean)` ≈ 0.11 for set_01. With a tight distribution, the p95 budget barely exceeds the mean-based budget:
- Mean-based: `4096 * 32 = 131,072` vs p95-based: `4096 * 38 = 155,648` — only 19% overhead.

The actual VRAM underutilization is **not because p95 is too conservative** — it's because `batch_size=4096` itself is too small for 100K-parameter models on a V100. The p95 multiplier just adds 19% headroom on top of an already undersized base.

### Evidence: batch_size is the real bottleneck

From the ablation log:
- Non-GPS models peak at 4-6 GB VRAM with `batch_size=4096`
- GPS at `batch_size=512` peaked at 3.9 GB (set_01) and 14.8 GB (set_02)
- GPS at `batch_size=1024` peaked at ~12 GB (set_01) — already close to capacity

For non-GPS models with ~100K params and forward pass memory proportional to batch node count, the relationship is roughly: `VRAM ≈ model_params + activations(batch_nodes)`. With models this small, activations dominate. Doubling `batch_size` from 4096 to 8192 would roughly double VRAM to 8-12 GB — much better utilization.

### Recommendation

**Increase `batch_size` in `TrainingConfig` default and model presets**, not change the percentile:

| Model type | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| vgae (small) | 4096 | 8192 | p95 budget = 311K nodes → ~9-10 GB VRAM est. |
| gat (small) | 4096 | 8192 | Same architecture scale |
| gat (large) | 8192 | 8192 | Already higher; keep |
| dgi (small) | 4096 | 8192 | Same as VGAE |
| GPS (small) | 512 (manual) | 256 (set_02) / 512 (set_01) | See Issue 2 |

Keep p95 as the budget percentile. Switching to p99 gains only ~5% more packing per batch but exposes the 1% of batches to near-OOM — not worth the risk for negligible throughput gain. The `OOM-retry` mechanism from the prior plan (Phase 3.1) is a better safety net.

**Alternative: variance-adjusted budget.** Instead of `batch_size * p95`, use `batch_size * (mean + k * std)` where `k` is tuned per-dataset. This is theoretically tighter but: (a) requires storing `std` in metadata, (b) is harder to reason about, (c) gives almost the same result when variance is low. Not worth the complexity for CAN bus graphs where `p95/mean ≈ 1.2`.

## Issue 2: GPS OOM — Immediate or Gradual? Attention-Aware Budgeting

### Root cause: O(N^2) global attention in GPSConv

`GPSConv` with `attn_type="multihead"` uses `torch.nn.MultiheadAttention`, which computes a **full N x N attention matrix** across all nodes in the batch. This is fundamentally different from message-passing convolutions (GATv2, etc.) that are O(E).

For DynamicBatchSampler, the batch is a single mega-graph where N = total nodes across all packed graphs. The attention computation sees ALL nodes, not per-graph attention — this is how PyG's batching works with global attention.

Evidence from ablation run 001 (GPS OOM table):

| batch_size | p95 | Budget (nodes) | Attempted alloc | Dataset |
|-----------|-----|----------------|-----------------|---------|
| 4096 | 38 | 155,648 (≈ 4096*38) | 105 GB | set_01 |
| 4096 | 45 | 184,320 (≈ 4096*45) | 169 GB | set_02 |
| 1024 | ~52 | ~53,248 | 10.5 GB | set_02 |

The attention matrix for N=155,648 nodes in fp16: `N^2 * 2 bytes = 155648^2 * 2 ≈ 48.5 GB` — just the attention weights alone, before keys/queries/values. This confirms **immediate OOM on the first batch**, not a gradual leak.

### Why batch_size=512 worked for set_01 but not set_02

- set_01 at batch_size=512: budget = `512 * 38 = 19,456 nodes`. Attention matrix: `19456^2 * 2 ≈ 756 MB`. Total VRAM ~3.9 GB. Fits easily.
- set_02 at batch_size=512: budget = `512 * 45 = 23,040 nodes`. Attention matrix: `23040^2 * 2 ≈ 1.06 GB`. This should also fit — the 14.8 GB peak for set_02 (job 45973219, which completed) suggests it ran but was tight.
- set_02 at batch_size=1024: budget = `1024 * 45 = 46,080 nodes` (but metadata shows p95=52 for this particular cache version?). Attention matrix: `46080^2 * 2 ≈ 4.24 GB`. Total reported as 10.5 GB attempted — consistent with Q/K/V buffers + activations on top.

### Recommendation: model-type-aware budget cap

GPS needs a **separate budget ceiling** that accounts for O(N^2) memory. The key insight: DynamicBatchSampler's `max_num` controls the total node count per batch. For GPS, this must be capped at a level where `N^2` fits in VRAM.

For V100 (16 GB), reserving 12 GB for attention (rest for model + activations):
- `N_max = sqrt(12 GB / (2 bytes * num_heads * 3))` — 3 for Q,K,V matrices
- With heads=4 (from _make_conv): `N_max = sqrt(12e9 / 24) ≈ 22,360 nodes`
- Safe margin: **N_max ≈ 15,000-20,000 nodes**

Implementation options:

**Option A: GPS-specific batch_size cap in models.yaml** (recommended)
Add GPS model presets to `models.yaml` with batch_size tuned for O(N^2):
```yaml
# In a GPS-specific preset or as conv_type override
conv_gps_small:
  training:
    batch_size: 384  # → budget ≈ 384 * 38 = 14,592 nodes (set_01), 384 * 45 = 17,280 (set_02)
```
This keeps the infrastructure unchanged. The ablation builder already does `"training.batch_size": 512` for conv_gps — just needs right-sizing per dataset.

**Option B: attn_type="performer" for O(N) memory**
PyG's GPSConv supports `attn_type="performer"` which uses Performer attention (Choromanski et al., 2020) — linear in N instead of quadratic. This would let GPS use the same batch sizes as GATv2. Tradeoff: approximation quality vs. exact attention. The GPS paper (Rampasek et al., 2022) reports "huge memory improvement and no major performance drops" when switching to linear attention.

Implementation: Change `_make_conv()` in `_utils.py:101`:
```python
gps = GPSConv(channels, inner, heads=heads, attn_type="performer", dropout=0.1)
```

**Option C: Per-graph attention isolation** (not feasible with current PyG)
PyG's batching creates a single mega-graph. Global attention sees all nodes across all graphs. Isolating attention per-graph would require passing `batch` indices to the attention layer and masking — GPSConv doesn't support this natively.

**Recommendation**: Use **Option A** (right-sized batch_size) for the immediate resubmit, and test **Option B** (performer attention) as a follow-up experiment. Option A is zero-code-change; Option B changes model semantics and needs a validation comparison.

### Profiling: Is there a memory leak?

The ablation data shows:
- Completed GPS jobs hit stable peak VRAM early (3.9 GB for set_01 batch=512, 14.8 GB for set_02 batch=512)
- Failed GPS jobs attempted single allocations of 105-169 GB — these are immediate first-batch failures, not gradual leaks

For non-GPS models, the 4-6 GB peak was consistent across the entire training run (DeviceStatsMonitor logged via Lightning callbacks). No evidence of gradual memory growth.

**Verdict: No memory leak detected.** OOM is a first-batch phenomenon for GPS due to quadratic attention. Non-GPS models are stable but underutilized.

To definitively rule out leaks for future runs, add a lightweight callback:
```python
# In trainer_factory.py callbacks
class VRAMWatchdog(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            peak = torch.cuda.max_memory_allocated() / 1e9
            pl_module.log("vram_peak_gb", peak)
```
DeviceStatsMonitor already does this, but the above is lighter-weight and logs explicitly to the CSV logger.

## Issue 3: Data Staging — Multi-Process Copy and Shared Staging

### Current flow

```
ESS (/fs/ess/PAS1266/kd-gat/) → Scratch (/fs/scratch/PAS1266/kd-gat-data/) → TMPDIR ($TMPDIR/kd-gat-data/)
       rsync (incremental, smart marker)              cp -r (single-threaded)
```

- ESS→Scratch: rsync with `.staged_marker` file count check. Fast when marker is fresh (skip). Slow on first run or after 90-day purge.
- Scratch→TMPDIR: `cp -r` every job. **This is the bottleneck.**

### Data sizes (measured)

| Source | Size |
|--------|------|
| Scratch raw/ | 11 GB |
| Scratch cache/ total | 86 GB |
| set_01 cache | 4.4 GB |
| set_02 cache | 5.9 GB |
| Per-dataset (relevant for ablation) | 4-6 GB each |

Training jobs stage with `--cache` flag, so they copy the entire cache/ directory (86 GB) or could be scoped to per-dataset. The ablation CPU eval jobs timed out at 30 min because `cp -r` of 86 GB on local disk takes ~15-30 min.

### Sub-issue 3a: Multi-process the copy

**Can we parallelize ESS→Scratch?**

ESS is NFS, Scratch is GPFS. rsync is single-threaded by design. Options:

| Tool | How | Fit for HPC? |
|------|-----|--------------|
| `parsyncfp` | Splits file list, runs N rsyncs in parallel | Yes — designed for HPC. Uses `fpart` for file list splitting. Available on GitHub. |
| `msrsync` | Buckets files into 1GB chunks, N parallel rsyncs | Yes — lightweight wrapper, no dependencies beyond Python. |
| `xargs -P N` + `rsync` per subdirectory | Manual splitting by top-level dirs | Simple but coarse — set_01/ and set_02/ are 4-6 GB each. |
| GNU `parallel` + `rsync` | Per-file parallelism | Over-parallelizes small files, high metadata overhead on NFS. |

**Recommendation: `xargs -P` with per-dataset rsync** — the simplest approach that matches the data layout. Since each dataset is a separate directory (set_01/, set_02/, etc.), parallelize at the dataset level:

```bash
# In stage_data.sh, replace single rsync with parallel per-dataset
datasets=$(ls "${DATA_ROOT}/cache/")
echo "$datasets" | xargs -P 4 -I{} rsync -a "${DATA_ROOT}/cache/{}/" "${SCRATCH_DATA}/cache/{}/"
```

This gives 4x speedup for the initial ESS→Scratch copy (4 datasets in parallel). For the Scratch→TMPDIR copy, `cp -r` can similarly be parallelized:

```bash
ls "${SCRATCH_DATA}/cache/" | xargs -P 4 -I{} cp -r "${SCRATCH_DATA}/cache/{}" "${TMPDIR_DATA}/cache/"
```

**But the real fix is sub-issue 3b — don't copy per-job at all.**

### Sub-issue 3b: Write-once-read-N for the DAG

**Is TMPDIR per-job or per-node?**

Per OSC docs: TMPDIR is **per-job, per-node**. "The batch system creates this directory when your job starts and deletes it when your job ends." Each SLURM job gets its own isolated TMPDIR. Two jobs on the same physical node cannot share a TMPDIR.

**Can jobs on the same node share?**

Not via TMPDIR. But OSC offers **PFSDIR** (`--gres=pfsdir`) — a per-job directory on the parallel scratch filesystem, shared across all nodes in a multi-node job. However, PFSDIR is per-job too, not per-DAG.

**Options for write-once-read-N:**

| Option | Mechanism | Speedup | Complexity |
|--------|-----------|---------|------------|
| **A: Skip TMPDIR, read from Scratch** | Set `KD_GAT_CACHE_ROOT=$SCRATCH_DATA/cache` | Eliminates copy entirely | Trivial — `SKIP_STAGE_DATA=1` or just don't stage to TMPDIR |
| **B: Shared scratch subdirectory** | Staging job writes to scratch, training jobs read from scratch | Same as A | Already how ESS→Scratch works |
| **C: Dedicated staging job in DAG** | First job in DAG stages data to a shared location, downstream jobs read it | Centralizes the copy | Medium — needs DAG dependency |
| **D: Dataset-scoped staging** | Each job only copies its own dataset, not all 86 GB | 86 GB → 4-6 GB per job | Small change to stage_data.sh |

**Recommendation: Option D (dataset-scoped) + Option A (skip TMPDIR for CPU eval)**

The current `stage_data.sh` copies the **entire** cache/ directory (86 GB) even though each job only needs one dataset (4-6 GB). Fix:

1. **Add `--dataset` flag to `stage_data.sh`** — only copy the specified dataset's subdirectory:
   ```bash
   # stage_data.sh --cache --dataset set_01
   # copies only cache/set_01/ (4.4 GB) instead of cache/ (86 GB)
   ```

2. **Skip TMPDIR for CPU eval/fusion jobs** — these jobs are I/O-light (single inference pass). Reading from Scratch (GPFS) is fast enough. The ablation's CPU eval jobs timed out because they were copying 86 GB when they only needed one ~5 GB dataset for a 2-minute evaluation. Set `SKIP_STAGE_DATA=1` or `STAGE_DATA_ARGS="--skip-tmpdir"` for eval/fusion jobs.

3. **For GPU training jobs** that benefit from local SSD: keep TMPDIR staging but scope to the single dataset. Copying 5 GB takes ~10-15 seconds on local SSD, vs. 15-30 minutes for 86 GB.

Latency comparison for set_02 (5.9 GB):

| Method | Time estimate |
|--------|--------------|
| Current: `cp -r` 86 GB to TMPDIR | 15-30 min |
| Dataset-scoped: `cp -r` 5.9 GB to TMPDIR | 15-30 sec |
| Skip TMPDIR, read from Scratch GPFS | 0 sec copy, ~10% slower reads |

The manifest orchestrator already knows the dataset per job (it's in the `StageJob.dataset` field). Passing `--dataset $DATASET` to `_preamble.sh` is straightforward.

## Implementation Sketch

### For immediate resubmit (ablation run 002)

1. **Increase batch_size** to 8192 in `TrainingConfig` default (line 136 of `__init__.py`)
2. **GPS batch_size** to 256 for set_02 and 512 for set_01 in `build_ablation.py` (dataset-conditional), or conservatively 256 for both
3. **Bump wall time** to 240 min for training, 60 min for eval (already partly done in resources.yaml)
4. **Dataset-scoped staging**: Add `STAGE_DATASET` env var to `stage_data.sh`, scope TMPDIR copy to single dataset
5. **Skip TMPDIR for CPU eval**: Add `SKIP_STAGE_DATA=1` to eval/fusion resource profiles or job scripts

### For follow-up (post-ablation)

6. **Test Performer attention** (`attn_type="performer"`) for GPS — if quality holds, GPS can use standard batch sizes
7. **Parallel ESS→Scratch rsync** with `xargs -P` per dataset
8. **VRAM watchdog callback** for systematic memory profiling across all model types

## Source Files (read during implementation)

| File | Why |
|------|-----|
| `graphids/config/__init__.py:132-170` | `TrainingConfig` — `batch_size` default, `dynamic_batching` |
| `graphids/config/models.yaml` | Model presets — per-model batch_size overrides |
| `graphids/pipeline/stages/data_loading.py:31-57` | `compute_node_budget()` — p95 budget calculation |
| `graphids/pipeline/stages/data_loading.py:60-101` | `make_dataloader()` — DynamicBatchSampler wiring |
| `graphids/core/preprocessing/datasets/can_bus.py:115-159` | `_write_cache_metadata()` — graph stats written during preprocessing |
| `graphids/core/models/_utils.py:94-104` | `_make_conv()` GPS branch — attn_type parameter |
| `scripts/data/stage_data.sh` | Data staging — TMPDIR copy logic to scope per-dataset |
| `scripts/slurm/_preamble.sh` | SLURM preamble — `STAGE_DATA_ARGS` passthrough |
| `scripts/build_ablation.py:58` | GPS config — `training.batch_size` override |
| `graphids/config/resources.yaml` | SLURM resource profiles — wall times, memory |
| `graphids/pipeline/orchestration/manifest.py` | Manifest → SLURM DAG — `StageJob.dataset` available for staging |
| `graphids/core/preprocessing/datamodule.py:167-181` | `_build_loader()` — where node budget flows to DataLoader |
| `graphids/pipeline/stages/modules.py:420-454` | `CurriculumDynamicBatchSampler.__iter__()` — node budget packing |

## Open Questions

1. **Per-dataset batch_size for GPS**: Should the ablation builder emit dataset-conditional batch_size for conv_gps (256 for set_02, 512 for set_01), or use a single conservative value (256) for both? The latter is simpler but sacrifices ~50% throughput on set_01.

2. **Performer attention quality**: Does `attn_type="performer"` preserve GPS's advantage over GATv2 for CAN bus graphs? The GPS paper shows negligible drops on molecular benchmarks, but CAN bus graphs are much smaller (~30-50 nodes) where exact attention might matter more. Needs an ablation point.

3. **PFSDIR on OSC**: Does PAS1266 have access to `--gres=pfsdir`? This would enable a per-job shared scratch that's faster than GPFS but shared across nodes. Worth testing for multi-node jobs if the project scales up.

4. **Cache directory cleanup**: Scratch has 86 GB in cache/ including stale versioned directories (v3.0.0, v4.0.0, v5.0.0, v7.0.0 = 64 GB total). Deleting these before the next ablation would make even full-cache staging feasible in ~1 minute.

## Cross-Repo Impact

None. All changes are internal to `KD-GAT`. No impact on `kd-gat-paper`, `dotfiles`, `lab-setup-guide`, or `osc-usage`.
