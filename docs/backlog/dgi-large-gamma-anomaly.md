# DGI Large γ Anomaly

## Problem

`probe-budget` (job 46273452) measured γ (collation rate) for DGI large at
20,000-208,000 μs/graph. Every other model×scale combo gets 60-75 μs/graph
on the same datasets.

DGI small (probed immediately after DGI large on same data) gets normal γ.
Not a cold-start artifact.

## Data

| model/scale | hcrl_ch | hcrl_sa | set_01 | set_02 |
|-------------|---------|---------|--------|--------|
| dgi/large | 19,972 μs | 6,459 μs | 206,062 μs | 208,481 μs |
| dgi/small | 60 μs | 62 μs | 70 μs | 74 μs |
| gat/large | 87 μs | 64 μs | 68 μs | 74 μs |
| vgae/large | 59 μs | 63 μs | 68 μs | 76 μs |

## Hypothesis

γ is measured as `Batch.from_data_list()` wall time — pure CPU, should be
model-independent. DGI large is the first combo probed (alphabetical sort).
Model is on GPU (345K params, 1.4MB) before γ measurement. Cause unknown —
possibly CUDA context or allocator side effect from `model.to(device)`.

## Impact

DGI large gets a wildly inflated cg_ratio, making it look collation-bound
when it's not. Budget is still safe (throughput ceiling is conservative).

## To investigate

1. Swap probe order (probe VGAE first) — does the anomaly follow DGI or first position?
2. Add `torch.cuda.synchronize()` before γ measurement
3. Measure γ separately from model probe loop (once per dataset, no model on GPU)
