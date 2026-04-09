# Pre-batch Timing Rationale — hcrl_sa (probe 2026-04-07)

Justifies the choice of prebatched training + workers=0 over per-step collation
with workers. Numbers are from a single probe run; see `data-flow.md` for current
architecture.

## Dataset and hardware

- hcrl_sa: 19,085 graphs, mean 38.3 nodes, mean 98 edges, ~10.8 KB/graph
- V100 16GB, PCIe 3.0 x16 (~12 GB/s), pin_memory ~20 GB/s

## Probe numbers (optimizer+compile warmup included)

| Model      | Budget (nodes) | Graphs/batch | Batch MB | T_collation (old) | T_gpu  | H2D (est) | Pin (est) |
|------------|---------------|-------------|----------|-------------------|--------|-----------|-----------|
| VGAE small | 404,718       | 10,557      | 113.9    | 380.7 ms          | 154.9 ms | 9.5 ms  | 5.7 ms  |
| VGAE large | 343,522       | 8,959       | 96.7     | 327.1 ms          | 286.8 ms | 8.1 ms  | 4.8 ms  |
| GAT small  | 233,339       | 6,086       | 65.7     | 218.5 ms          | 261.5 ms | 5.5 ms  | 3.3 ms  |
| GAT large  | 62,439        | 1,629       | 17.6     | 58.9 ms           | 121.6 ms | 1.5 ms  | 0.9 ms  |

T_collation = `Batch.from_data_list()` cost **on the old per-step path only**.
H2D = batch_MB / 12 GB/s.  Pin = batch_MB / 20 GB/s.

## Why prebatch wins

**Old path (per-step collation, 3 workers):** `T_step = max(T_collation/workers, T_gpu) + T_H2D`.
For VGAE small: max(381/3, 155) + 10 = max(127, 155) + 10 = 165 ms/step, GPU util ~60-83%.
Workers needed because T_collation (381 ms) >> T_gpu (155 ms).

**New path (prebatch at setup, workers=0):** `_prebatched_train_dataloader` calls
`NodeBudgetBatchSampler` once at setup to plan batches, then `Batch.from_data_list()`
once per batch (not per step, not per epoch). The training loop iterates over a
`list[Batch]` with `num_workers=0` wrapped by `PrefetchLoader` for async H2D.

Per-step CPU work: pin(5.7 ms) + H2D(9.5 ms) = 15.2 ms. GPU step = 154.6 ms.
CPU finishes preparing next batch in 15.2 ms, waits 139 ms. GPU never idles.

Worst case across all model/scale combos: VGAE small at 15.2/154.6 = 9.8% overhead.

|                | OLD (3 workers) | NEW (prebatched) |
|----------------|-----------------|------------------|
| T_step         | ~165 ms         | ~155 ms          |
| GPU util       | ~60-83%         | ~100%            |
| CPUs           | 5               | 1-2              |

Workers add IPC serialization overhead (~2-5 ms) for zero benefit when each
`__getitem__` is O(1) list lookup. See `graphids/core/data/datamodule/graph.py:328`.
