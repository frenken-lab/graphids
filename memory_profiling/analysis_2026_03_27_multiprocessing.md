# DataLoader Multiprocessing Strategy Matrix (2026-03-27)

## The three dimensions

### 1. Start method (`multiprocessing_context`)

| Method | How workers start | CUDA safe? | Memory inheritance |
|---|---|---|---|
| `fork` | Clone parent via `fork()` | **NO** — segfault after CUDA init | Copy-on-write: workers share parent's pages until written |
| `spawn` | Fresh process, pickle dataset to worker | Yes | No inheritance — all state transferred via pickle |
| `forkserver` | Fork from a pristine server process (no CUDA) | Yes (if server created before CUDA) | Clean fork from pre-CUDA server. Tensors still pickled. |

**Current:** `spawn` (required — we init CUDA before DataLoader).

### 2. Sharing strategy (`torch.multiprocessing.set_sharing_strategy`)

| Strategy | IPC mechanism | OSC constraint |
|---|---|---|
| `file_descriptor` | `/dev/shm` mmap entries | **Hits `vm.max_map_count=65530`** on large datasets. 700K graphs × 6 tensors = 4.2M entries → OOM. |
| `file_system` | `/tmp` file-backed mmap | No mmap count limit. Files cleaned on process exit. |

**Current:** `file_system` (required on OSC — verified fix for mmap OOM).

### 3. Tensor serialization path

**Critical finding:** There are TWO different pickle paths, and they behave differently.

| Path | When used | What happens |
|---|---|---|
| `tensor.__reduce_ex__()` | Standard `pickle.dump` | Serializes full tensor bytes inline. **Always copies.** |
| `torch.multiprocessing.reductions.reduce_storage()` | `ForkingPickler` in DataLoader workers | Creates shared memory file, passes filename. **Workers mmap the same file.** |

**The DataLoader uses `ForkingPickler` → `reduce_storage`**, NOT standard pickle.

## Verified: `reduce_storage` shares tensor data (not copies)

```python
# file_system strategy + reduce_storage
result = reduce_storage(tensor.untyped_storage())
# → rebuild_storage_filename(cls, bytes(38), bytes(26), nbytes=700000)
# The 38 bytes is a /tmp filename, not the data.

# Two rebuilds from the same reduction:
s1 = rebuild(*args)
s2 = rebuild(*args)
s1.data_ptr() == s2.data_ptr()  # True — SAME virtual address
t1[0,0] = 12345; t2[0,0] == 12345  # True — write-through
```

| Strategy | Reduce function | What's sent to worker | Workers share memory? |
|---|---|---|---|
| `file_system` | `rebuild_storage_filename` | Filename (38 bytes) + size | **Yes** — all workers mmap same file |
| `file_descriptor` | `rebuild_storage_fd` | File descriptor + size | **Yes** — all workers dup the same fd |

**Both strategies share tensor data across workers. Neither copies the tensor bytes.**

## So where does the 37.7G RSS come from?

If `_data` tensors (5.9G on set_02) are shared via mmap, the per-worker copy cost is zero.
The 37.7G measured RSS must come from:

| Source | Estimated size | Per-worker? |
|---|---|---|
| PyTorch + PyG + scipy import | ~3G | Yes (each spawn worker reimports) |
| CUDA context (not in workers) | ~2-4G | Main process only |
| `_data` tensors (shared via mmap) | 5.9G | **Shared** — counted once in RSS but pages shared |
| `_data_list` cache (separate() results) | ~1.5-2G per worker | Yes (each worker builds own cache) |
| DynamicBatchSampler internal state | ~0.5G | Main process only |
| Python/OS overhead | ~1-2G | Yes |

**Key insight:** RSS includes mmap'd pages. Linux counts shared mmap pages in EACH
process's RSS even though the physical memory is shared. So the 37.7G RSS with 2 workers
is NOT 37.7G of unique physical memory — it double-counts the shared `_data` tensors.

To see actual unique physical memory, check `PSS` (Proportional Set Size) instead of `RSS`:
```bash
# On compute node:
cat /proc/<pid>/smaps_rollup | grep Pss
```

PSS divides shared pages proportionally. With 3 processes sharing 5.9G:
- RSS per process: includes full 5.9G
- PSS per process: includes 5.9G/3 = 2G

Estimated actual physical memory (PSS-based):
```
Main:    3G import + 4G CUDA + 5.9G/3 shared + 0.5G DBS = ~9.5G
Worker1: 3G import + 5.9G/3 shared + 1.5G _data_list = ~6.5G
Worker2: 3G import + 5.9G/3 shared + 1.5G _data_list = ~6.5G
Total unique physical: ~22.5G
```

vs RSS-reported: 37.7G (double/triple counts the shared 5.9G).

## What this means for resource profiles

If the RSS bloat is from RSS double-counting shared mmap pages, then:
- **`--mem` should be based on PSS, not RSS** — 24-28G for 2 workers, not 36-48G
- The 48G/54G requests that "worked" were 2× over-provisioned
- The 24G OOM (job 45984063) may have been a real limit hit or from _data_list bloat

**This needs empirical verification on a GPU node** — compare RSS vs PSS vs actual
SLURM memory accounting (`sacct MaxRSS` vs `sstat MaxRSS`).

## Action items

1. **Verify PSS vs RSS on GPU node** — submit a short training job with:
   ```bash
   # In worker_init_fn, log PSS:
   cat /proc/self/smaps_rollup | grep Pss
   ```
2. **If PSS confirms sharing works**, reduce `--mem` in resource profiles
3. **The `_data_list` per-worker cache is the remaining bloat source** — clearing
   it only helps the main process. Workers populate their own copy via `get()`.
   With `persistent_workers=True`, this cache persists across epochs (good for speed,
   costs ~1.5G per worker on set_02).

## Multiprocessing options ranked

| Option | Speed | Memory (actual physical) | Complexity | CUDA safe? |
|---|---|---|---|---|
| **`spawn` + `file_system` + `persistent_workers`** | Good (warm cache epoch 2+) | ~22G (2w, set_02) | **Current setup** | Yes |
| `forkserver` + `file_system` + `persistent_workers` | Same | ~16G (workers fork clean, share parent COW) | Medium | Yes if server created before CUDA |
| `num_workers=0` + main-thread collation | Slower (no overlap) | ~15G (no workers) | Simplest | Yes |
| `fork` + COW | Fastest startup | ~16G (COW pages) | Simplest | **NO — crashes with CUDA** |

## Empirical results: spawn vs forkserver vs num_workers=0

Tested on login node with 5000 synthetic graphs (35.6MB tensors), 2 workers.

| Metric | spawn (2w) | forkserver (2w) | num_workers=0 |
|---|---|---|---|
| Cold epoch | 7.45s | 5.74s (**23% faster**) | 0.57s |
| Warm epoch | 0.33s | 0.33s (identical) | 0.57s |
| Worker RSS | 713MB | 713MB (same) | N/A |
| Worker PSS | 461MB | 461MB (same) | N/A |

**Conclusions:**
1. `forkserver` has faster cold start (23%) but identical steady state.
2. Worker memory is the same — forkserver still pickles the dataset via ForkingPickler.
3. PSS < RSS confirms shared mmap pages (252MB shared per worker on this dataset).
4. `num_workers=0` is fastest on small data — worker overhead only pays off on large batches.

**`forkserver` is not worth the complexity.** The 23% cold-start improvement only affects
epoch 1 of a 300-epoch run. Steady-state throughput is identical.

GPU-node test submitted as job `46018204` (50K graphs, 3 workers) for full-scale validation.

## Final recommendation

**Keep current setup: `spawn` + `file_system` + `persistent_workers=True`.**

It's already doing the right things:
- Tensors are shared via mmap'd files (not copied) through ForkingPickler
- Workers persist across epochs (warm cache from epoch 2)
- `file_system` strategy avoids vm.max_map_count limits on OSC
- RSS appears inflated due to shared page double-counting; actual PSS is lower

The "37.7G RSS" bloat is partially a measurement artifact. Use PSS for accurate
memory accounting and right-size `--mem` accordingly.

### Where to configure

| Setting | Where | Current | Lightning component |
|---|---|---|---|
| Start method | `__main__.py:8` | `spawn` | Process-global, before Trainer |
| Sharing strategy | `__main__.py:9` | `file_system` | Process-global, before Trainer |
| num_workers | DataModule YAML | `2` | `data.init_args.num_workers` |
| persistent_workers | `make_graph_loader` default | `True` | DataLoader kwarg |
| multiprocessing_context | `make_graph_loader` default | `spawn` | DataLoader kwarg |
| accelerator | Trainer YAML | `gpu` | `trainer.accelerator` |
| strategy | Trainer YAML | `auto` (SingleDevice) | `trainer.strategy` |
| SLURM environment | Trainer YAML | `SLURMEnvironment` | `trainer.plugins` |
| precision | Trainer YAML | `16-mixed` | `trainer.precision` |

All configuration lives in the stage YAML. No code changes needed to tune resources —
just edit `data.init_args.num_workers`, `slurm.mem`, `slurm.cpus_per_task` in the config.
