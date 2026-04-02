# ADR 0008: No Custom Collation — Warm Cache Beats FastCollate

> Date: 2026-03-30 | Status: **Accepted**

## Context

Early profiling showed 30% GPU utilization with 2 workers. A custom `_FastCollate`
(vectorized tensor slicing) achieved 82% in short spike runs. After full training
measurement (Run 003, 300 epochs), standard PyG collation measured 83-90%.

## Decision

**Use standard PyG `Batch.from_data_list()`. No custom collation.**

## Rationale

| Path | Time/batch | When |
|------|-----------|------|
| FastCollate (vectorized) | 85ms | Every batch, every epoch |
| `from_data_list` cold | 166ms | Epoch 1 only |
| `from_data_list` warm (`_data_list` cached) | 52ms | Epoch 2-300 |

With `persistent_workers=True`, workers survive across epochs. After epoch 1,
`_data_list[i]` is a cache hit. FastCollate is 2x faster than cold but **1.6x slower
than warm**. Over 300 epochs, net negative.

The original 82% measurement was from a 5-epoch spike where epoch 1 dominated. Run 003
(full training) confirmed 83-90% with standard collation.

## Consequences

- `_FastCollate` correctly deleted (commit `7ece283`)
- `persistent_workers=True` is essential (enables warm cache)
- `spawn` + `file_system` sharing strategy required (OSC constraint)
- V100 is the sweet spot — faster GPUs are harder to feed (T_c is CPU-bound)

## Sources

- Run 003 measurements: 83% VGAE, 90% GAT GPU utilization
- Spike job 45985264: FastCollate profiling
- `docs/reference/dataloader-performance.md` — full performance model
