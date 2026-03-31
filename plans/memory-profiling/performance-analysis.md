# DataLoader Performance Analysis — Consolidated

> Consolidates: investigation.md, investigation_v2.md, scenario.md,
> resource_plan_2026_03_27.md, analysis_2026_03_27_collation.md,
> analysis_2026_03_27_multiprocessing.md
>
> Last updated: 2026-03-30

## Current State

| Setting | Value | Where configured |
|---------|-------|-----------------|
| Start method | `spawn` | `__main__.py` (process-global) |
| Sharing strategy | `file_system` | `__main__.py` (process-global) |
| num_workers | 2 | Stage YAML `data.init_args.num_workers` |
| persistent_workers | True | `make_graph_loader` default |
| Collation | PyG standard `Batch.from_data_list()` | PyG DataLoader default |
| Batch sampler | `DynamicBatchSampler` | `make_graph_loader` |
| Precision | 16-mixed | `trainer.yaml` |

**Steady-state collation cost:** T_c ≈ 52ms (warm `_data_list` cache, epoch 2+).
**GPU compute:** T_g ≈ 10ms (VGAE) / 25ms (GAT).
**Measured GPU utilization:** 83% (VGAE set_01) / 90% (GAT set_02) — Run 003 full training.

---

## Performance Model

```
GPU_util = min(1.0, num_workers × T_gpu / T_collate)
RAM      = M_base + num_workers × D + overhead

where:
  T_collate ≈ 52ms   (warm cache, epoch 2+)
  T_gpu     ≈ 10ms   (VGAE) / 25ms (GAT)
  M_base    ≈ 15G    (PyTorch + CUDA context + mmap'd dataset)
  D         ≈ dataset on-disk size (5.9G for set_02)
  overhead  ≈ 4G     (DynamicBatchSampler cache, Python/OS)
```

**Validation against measurements:**

| Configuration | Predicted | Measured | Source |
|---------------|-----------|----------|--------|
| 2 workers, old collate (T_c=70ms) | 29% | 30% | Early spike jobs |
| 2 workers, FastCollate (T_c=25ms) | 80% | 82% | Spike job 45985264 |
| 2 workers, warm cache (T_c=52ms) | 38% | 83-90% | Run 003 (full training) |

The warm-cache measured util exceeds the simple model prediction because prefetch
buffering + batch size variance smooth the pipeline. The model is conservative.

---

## Investigation Timeline

| Date | Finding | Evidence | Status |
|------|---------|----------|--------|
| Pre-03-25 | `Batch.from_data_list()` T_c≈70ms, GPU util 30% with 2 workers | Spike job 45984077 | **Superseded** — warm cache faster |
| 03-25 | `_FastCollate` (tensor slicing) achieves T_c≈25ms, 82% GPU util | Spike job 45985003, profile 45985264 | **Superseded** — slower than warm cache for long runs |
| 03-25 | FastCollate + PrefetchLoader both deleted in cleanup | Commits `7ece283`, `b73ae3d` | Resolved — deletion was correct |
| 03-27 | FastCollate regression discovered — T_c back to 70ms | resource_plan analysis | **Superseded** — warm cache not measured initially |
| 03-27 | **FastCollate (85ms/batch) slower than warm cache (52ms/batch)** | Login node benchmark: 10K graphs, 1000-graph batches | **Current** — warm _data_list cache wins |
| 03-27 | Run 003 confirms 83-90% GPU util with standard collation | PLAN.md, jobs 45985737-45985750 | **Current** — no collation fix needed |
| 03-27 | spawn workers share tensors via ForkingPickler mmap, not copies | `reduce_storage()` analysis + empirical test | **Current** |
| 03-27 | RSS double-counts shared mmap pages; PSS is real metric | smaps analysis, login-node PSS test | **Current** — needs GPU-node verification |
| 03-27 | forkserver 23% faster cold start but identical steady state | Login-node benchmark: spawn vs forkserver vs 0 workers | **Current** — not worth complexity |

---

## Key Findings

### 1. Collation: warm cache beats FastCollate

| Path | Time/batch | When it runs |
|------|-----------|-------------|
| FastCollate (vectorized slicing) | 85ms | Every batch, every epoch |
| `from_data_list` cold (`_data_list=None`) | 166ms | Epoch 1 only |
| `from_data_list` warm (`_data_list` cached) | 52ms | Epoch 2-300 |

**Why:** Lightning caches DataLoaders (`reload_dataloaders_every_n_epochs=0`).
With `persistent_workers=True`, workers survive across epochs. After epoch 1,
`_data_list[i]` is a cache hit — no `separate()` needed. FastCollate is 2x faster
than cold but **1.6x slower than warm**. Over 300 epochs, net negative.

The original 82% GPU util measurement was on a 5-epoch spike where epoch 1 dominated.
Run 003 (full training, standard collation) measured 83-90% — confirming no fix needed.

**Why FastCollate was deleted correctly:** Commit `7ece283` removed it during the
"Replace custom DataLoader/collation/assembly with PyG APIs" cleanup. The performance
concern that motivated restoration investigation turned out to be based on short-run
profiling that didn't capture warm-cache behavior.

### 2. Memory: RSS is inflated, PSS is the real metric

PyTorch's `ForkingPickler` uses `reduce_storage()`, which creates shared memory files
(file_system strategy) or shared file descriptors (file_descriptor strategy). Workers
mmap the same file — **tensor data is shared, not copied**.

```
Estimated physical memory (PSS-based, 2 workers, set_02):
  Main:    3G import + 4G CUDA + 2G shared + 0.5G DBS = ~9.5G
  Worker1: 3G import + 2G shared + 1.5G _data_list    = ~6.5G
  Worker2: 3G import + 2G shared + 1.5G _data_list    = ~6.5G
  Total unique physical: ~22.5G

vs RSS-reported: 37.7G (double/triple counts shared 5.9G)
```

**Implication:** `--mem` requests of 36-48G for 2 workers were ~2x over-provisioned.
24-28G should suffice based on PSS. The 24G OOM (job 45984063) may have been from
`_data_list` cache bloat rather than tensor copies.

**Needs GPU-node verification** — submit job with `cat /proc/self/smaps_rollup | grep Pss`
in `worker_init_fn` to confirm PSS behavior under real conditions.

### 3. Multiprocessing: current setup is correct

| Option | Speed | Memory | Complexity | CUDA safe? |
|--------|-------|--------|------------|-----------|
| **spawn + file_system + persistent_workers** | Good (warm epoch 2+) | ~22G PSS (2w) | **Current** | Yes |
| forkserver + file_system | Same steady state | Same | Medium | Yes |
| num_workers=0 | No overlap | ~15G | Simplest | Yes |
| fork + COW | Fastest startup | ~16G | Simplest | **NO — crashes** |

- **spawn** required (CUDA initialized before DataLoader)
- **file_system** required on OSC (`vm.max_map_count=65530` breaks file_descriptor with large datasets)
- **forkserver** not worth it (23% cold start improvement on epoch 1 of 300)
- **persistent_workers** essential for warm-cache performance

### 4. Job failure root causes (March 2026)

| Category | Count | Root cause | Status |
|----------|-------|-----------|--------|
| Submitit orchestrator | 340 | Undersized resources, code bugs during migration | **Gone** — not using submitit |
| CPU OOM (preprocessing) | 18 | Requested 48G, peak RSS 47-56G | Fix: request 64G+ |
| GPU worker memory bloat | 8 | Spawn pickle overhead at 24G | Fix: PSS-based sizing |
| Code iteration bugs | ~100 | Normal R&D — fix and resubmit | Not a resource problem |

---

## Resource Profiles

Based on steady-state T_c≈52ms (warm cache). GPU util model is conservative;
actual measured util is higher due to prefetch buffering.

### Pitzer: 1x V100-16GB (current setup)

| Workers | GPU util (VGAE) | GPU util (GAT) | CPUs | RAM (PSS) | RAM (RSS) |
|---------|----------------|----------------|------|-----------|-----------|
| 0 | ~16% | ~33% | 1 | 15G | 15G |
| 1 | 19% | 48% | 2 | 18G | 21G |
| **2** | **38%** | **96%** | **3** | **22G** | **28G** |
| 3 | 58% | 100% | 4 | 28G | 34G |
| 4 | 77% | 100% | 5 | 34G | 39G |

**Measured (2 workers): 83-90% GPU util** — model is conservative. Current setup is good.

Sweet spot VGAE: y=2 (current), measured 83%. Increasing to 3 would help marginally.
Sweet spot GAT: y=2 (current), measured 90%. Already near-saturated.

### Cluster scaling (steady-state T_c≈52ms)

| Cluster | Config | Predicted util | Practical? | Notes |
|---------|--------|---------------|------------|-------|
| **Pitzer 1x V100** | y=2, 3 CPUs, 28G | 38% (meas: 83%) | **Current, works** | Best $/util for ablation |
| Pitzer 2x V100 DDP | y=2/gpu, 6 CPUs, 56G | 38% × 2 GPUs | Yes | ~1.9x throughput |
| Ascend 2x A100 | y=6/gpu, 14 CPUs, 92G | 35% per GPU | Overkill | A100 3x faster → harder to feed |
| Cardinal 4x H100 | y=10/gpu, 44 CPUs, 296G | 48% per GPU | Wasteful | 94GB VRAM 95% unused |

**Key insight:** Faster GPUs are HARDER to feed because T_c is CPU-bound. The workload
is best suited for V100 where T_g/T_c ratio is most favorable. A100/H100 only make
sense for larger models or if collation is eliminated entirely.

### Concrete YAML profiles (for resources.yaml)

```yaml
# GPU training (T_c≈52ms warm cache, current standard collation)
# RAM: PSS-based estimates. Add 30% headroom for RSS-based SLURM accounting.

vgae:
  medium:
    autoencoder:
      partition: gpu
      gres: "gpu:1"
      time: "03:00:00"
      mem: "36G"              # 22G PSS + 30% headroom + safety
      cpus_per_task: 3        # 2 workers + 1 main
      num_workers: 2
    curriculum:
      partition: gpu
      gres: "gpu:1"
      time: "03:00:00"
      mem: "36G"
      cpus_per_task: 3
      num_workers: 2

gat:
  medium:
    normal:
      partition: gpu
      gres: "gpu:1"
      time: "03:00:00"
      mem: "36G"
      cpus_per_task: 3
      num_workers: 2
    curriculum:
      partition: gpu
      gres: "gpu:1"
      time: "03:00:00"
      mem: "36G"
      cpus_per_task: 3
      num_workers: 2

dqn/bandit:
  medium:
    fusion:
      partition: gpu
      gres: "gpu:1"
      time: "01:00:00"
      mem: "16G"              # flat TensorDataset, no PyG DataLoader
      cpus_per_task: 2
      num_workers: 0

preprocess:
  any:
    partition: cpu
    time: "02:00:00"
    mem: "72G"                # measured 56G peak, 1.3x headroom
    cpus_per_task: 8

test:
  any:
    partition: cpu
    time: "00:30:00"
    mem: "16G"
    cpus_per_task: 8
```

### Dataset-specific memory scaling

| Dataset | Size on disk | RAM (2 workers, PSS) | RAM (2 workers, RSS) |
|---------|-------------|---------------------|---------------------|
| hcrl_ch | ~0.3G | 17G | 20G |
| hcrl_sa | ~0.5G | 17G | 21G |
| set_01 | ~3.0G | 20G | 25G |
| set_02 | ~5.9G | 22G | 28G |
| set_03 | ~5.0G | 21G | 27G |
| set_04 | ~5.0G | 21G | 27G |

For small datasets (hcrl_*), 20G mem suffices. For large datasets, 36G with headroom.

---

## Open Items

### CurriculumDataModule rebuilds DataLoader every epoch

From project memory (research_spawn_mmap_hpc.md): `CurriculumDataModule` rebuilds
DataLoader every epoch → kills persistent workers → 3-5s spawn per worker per epoch.
300 epochs × 2 workers × 4s = **40 min of pure spawn overhead**.

**Fix:** Create DataLoader once with `persistent_workers=True`. Update
`CurriculumSampler.set_epoch()` internal state only. ~10-line fix in `curriculum.py`.
Tracked in `plans/architecture/preprocessing-consolidation.md`.

### PSS verification on GPU node

The RSS vs PSS analysis was done on login node with synthetic data. Submit a short
training job with PSS logging in `worker_init_fn` to confirm behavior under real
SLURM memory accounting. If confirmed, reduce `--mem` in resource profiles.

```bash
# In worker_init_fn:
cat /proc/self/smaps_rollup | grep Pss
```

### Per-worker `_data_list` cache bloat

Each worker builds its own `_data_list` cache (~1.5-2G per worker on set_02) via
`InMemoryDataset.get()`. With `persistent_workers=True`, this cache persists — good
for speed, costs memory. Clearing it only helps the main process. No fix identified
that doesn't sacrifice warm-cache performance.

---

## Cross-references

- `plans/memory-profiling/pipeline-data-flow.md` — full pipeline data flow diagrams
- `plans/research/profiling-and-observability.md` — logging inventory, tool evaluations, observability plan
- `plans/research/nvidia-gpu-profiling-tools.md` — nsys, ncu, torch.cuda memory APIs
- `plans/research/lightning-profiler-vram-research.md` — why Lightning profilers can't replace _probe_bytes_per_node()
- `plans/open_issues.md` — CurriculumDataModule rebuild + PSS verification tracked there
