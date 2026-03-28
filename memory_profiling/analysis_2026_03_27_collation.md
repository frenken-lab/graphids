# Collation Analysis — Fast Collation Is Not the Fix (2026-03-27)

## Summary

Fast collation (`_FastCollate`, deleted in `7ece283`) was investigated for restoration.
Benchmarking reveals it's a **net slowdown** for the actual training workload because
Lightning caches DataLoaders and `persistent_workers=True` keeps workers alive across epochs.

## Key finding: Lightning calls `train_dataloader()` once

`reload_dataloaders_every_n_epochs` defaults to 0. Lightning calls `train_dataloader()` at
the start of `trainer.fit()`, caches the returned DataLoader, and reuses it for all epochs.
Workers with `persistent_workers=True` survive across epochs.

This means the DataLoader lifecycle is:
```
Epoch 1:   workers cold → separate() per graph → populate _data_list cache → slow
Epoch 2+:  workers warm → _data_list[i] cache hit → fast (52ms/batch)
```

For a 300-epoch run, epoch 1 is <1% of total training time.

## Benchmark results (login node, 10K synthetic graphs, 1000-graph batches)

| Path | Time/batch | When it runs |
|------|-----------|-------------|
| Fast collation (vectorized slicing) | 85ms | Always the same |
| `from_data_list` cold (`_data_list=None`) | 166ms | Epoch 1 only |
| `from_data_list` warm (`_data_list` cached) | 52ms | Epoch 2-300 |

**Fast collation is 2× faster than cold, but 1.6× SLOWER than warm.**

Over 300 epochs: fast collation costs extra time on 299 epochs to save time on 1.
Net negative.

## Why the original profiling showed 82% util with _FastCollate

The original investigation (investigation_v2.md) measured 82% GPU util with
`_FastCollate` vs 30% without. This was measured on a **short profiling run** (5 epochs)
where the cold first epoch dominated. On a full 300-epoch training run with
`persistent_workers=True`, the warm cache would dominate and standard collation
would be faster.

The Run 003 GPU profile (PLAN.md, jobs 45985737-45985750) confirms this:
- VGAE set_01: **83% GPU util** — this was with standard collation + persistent_workers
- GAT set_02: **90% GPU util** — same
- These numbers are from actual full training runs, not short spikes

The 30% number came from early experiments before `persistent_workers=True` was set,
or from epoch-1-dominated short runs.

## What actually matters

The real bottleneck is **worker memory bloat**, not collation speed:
- Each spawn worker pickles the full dataset: 5.9G per worker on set_02
- 4 workers = 15G base + 4×5.9G + overhead = 43G RAM
- This is why we OOM at 24G and need 36G+ for 2 workers

The collation is fast enough in steady state. The memory is the constraint.

## Action taken

- Reverted `_make_fast_collate`, `_IndexDataset` from `datamodule.py`
- `make_graph_loader` restored to standard PyG DataLoader with spawn/persistent defaults
- Next investigation: verify whether workers actually copy dataset tensors or share them

## Verified: spawn workers ALWAYS duplicate dataset tensors

Tested `torch.Storage.__reduce_ex__` (the serialization path DataLoader uses):

```
Normal storage reduce:  _load_from_bytes, 700254 bytes  (full copy)
Shared storage reduce:  _load_from_bytes, 700254 bytes  (full copy)
```

Both normal and `share_memory_()` tensors serialize the full byte content through
pickle's `__reduce_ex__`. The `file_system` sharing strategy does NOT help — it only
affects `multiprocessing.Queue` IPC, not DataLoader's dataset pickling.

**This means:** each spawn worker receives a full pickle copy of `dataset._data` tensors.
On set_02 (5.9G tensors), 3 workers = 17.7G of copies on top of the 15G base process.
This is unavoidable with spawn multiprocessing + PyG InMemoryDataset.

### What sharing strategies actually do

| Strategy | Mechanism | Helps DataLoader workers? |
|---|---|---|
| `file_system` | Uses `/tmp` files instead of `/dev/shm` for IPC | Avoids mmap count limits. Does NOT reduce copies. |
| `share_memory_()` | Moves tensor to shared memory segment | Only helps with `fork` (inherited memory). Spawn pickles anyway. |
| `mmap=True` on `torch.load` | OS memory-maps the .pt file | Mmap'd pages are shared across forks. Spawn pickles the data out of mmap. |

### The only ways to avoid per-worker copies

1. **`num_workers=0`** — no workers, no copies. But single-threaded collation.
2. **`fork` instead of `spawn`** — workers inherit mmap'd pages. But fork + CUDA = crash.
3. **Workers re-load from disk independently** — each worker `torch.load(path, mmap=True)`.
   OS page cache deduplicates. Requires custom `worker_init_fn` + dataset redesign.
4. **Pre-batched files** — save batches offline, workers just load one file per batch.
   No dataset in worker memory at all.

Options 3 and 4 are the real fixes. Option 3 is the most practical — see next investigation.
