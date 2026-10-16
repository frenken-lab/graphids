# Continuous event pipeline and single-A100 gate — 2026-10-16

The active pipeline is exclusively `can_event` / `can_event_dm` / `event_graph_mamba`. The exact pre-consolidation working tree, including uncommitted source, configurations, tests and dependency locks, is preserved in `archives/snapshot-2026-10-16.tar.gz`. All extracted file hashes matched; Python compilation and manifest/config import smoke checks passed before removal. Datasets, reviewed provenance, clock repairs, checkpoints, calibration and historical results remain in place.

Training now carries both Mamba caches, latest per-ID graph memory and previous-bus state across chronological chunks, detaching learned history at chunk boundaries. Equal-length independent recording streams batch on one GPU; tails and replacements preserve exact physical row coverage. Each optimizer window normalizes classification and verified-normal payload objectives over separate complete eligible-target totals. Epoch checkpoints preserve source/config, optimizer and RNG state, with exact resume parity and explicit rejection of partial epochs. Evaluation writes bounded score shards and computes exact AP from disk-mapped scalar columns.

The source preset selects 2,048 events × four independent streams, accumulating 8,192 events per optimizer update. Width, layers, objectives, loss weights, seed 42, FP32 and `val/classification_frame_ap` selection remain unchanged. CUDA operators and dependency locks are unchanged.

All 76 tests passed in Slurm step `7891556.8`, including official/reference outputs and gradients, nonzero incoming caches, arbitrary boundaries, untruncated versus functional gradients, truncated state, current-payload exclusion, future perturbations, distinct ID sets, short tails, replacement streams, masked/unequal target counts, exact coverage, and checkpoint resume through both manual and validation-selected saves. The source-only public Ray/Lightning launch completed a tiny synthetic epoch and wrote exact validation coverage.

`gx can verify-events` (CPU job `56290736`) independently compared all **21,285,666 source-query frames across 21 recordings** with their CSV rows. All fields, row order, timestamps and duplicate frames matched. No target recording was opened. The active provenance and clock-repair tests are retained separately from the retired snapshot tests.

The final benchmark used an NVIDIA A100-PCIE-40GB in allocation `7891556`. Every configuration processed the same 81,920 scored training frames over ten optimizer steps, after a warmup step and reset of weights, optimizer, RNG and history. Initial weight SHA256 was identical across configurations. Ten reviewed training recordings covered both vehicles, verified-normal traffic and positive attack frames. For each attack recording, the timed segment contains its first positive label; all preceding history was replayed from physical row zero under identical initial weights. That history is cloned per trial and is not charged to steady training throughput. Neither validation accuracy nor target data selected the configuration.

The baseline uses the archived event graph readout on the same correct continuous scan and common loader, at 512 events × one stream. It does not represent the retired snapshot architecture or claim a controlled learning comparison. End-to-end measured time includes CPU preparation, pinned transfers, forward/loss, backward, clipping and optimizer work, with synchronization for phase timing. Source cache materialization, initial history replay, validation, logging, and checkpoint I/O are outside the epoch projection.

| Configuration | Events/s | Step latency | Peak reserved | Projected source training epoch |
|---|---:|---:|---:|---:|
| Baseline 512 × 1 | 46,828 | 174.94 ms | 0.94 GiB | 182.1 s |
| Selected 2048 × 4 | 178,979 | 45.77 ms | 1.87 GiB | 47.6 s |

The measured speedup is **3.82×**. Projection scales the short warm-cache benchmark to 8,526,701 training frames; it is not a measured full epoch or a detection result. Short timing windows and later recording behavior can change the estimate. All configurations passed serial-versus-batched outputs, final caches and gradients and remained below the 32 GiB reserved-memory ceiling.

| Phase per 8,192-event optimizer window | Baseline | Selected |
|---|---:|---:|
| CPU preparation | 1.819 ms | 0.671 ms |
| Host to device | 2.080 ms | 0.624 ms |
| Forward and losses | 81.979 ms | 25.526 ms |
| Backward | 87.324 ms | 17.251 ms |
| Clipping and AdamW | 1.312 ms | 1.314 ms |

The initial profile attributed about 62% of CUDA time to generic repeated-index backward kernels. Projecting graph keys/values once per bank and using `index_select` preserved gradients while replacing that bottleneck; the final index-add kernels account for about 8%. Bulk NumPy ID conversion reduced selected-configuration CPU preparation from about 14 ms to 0.67 ms per optimizer window. Outgoing SSM gradient reconstruction is skipped only when returned history is explicitly detached; functional state gradients remain available and tested. Graph multiplications/reductions and host kernel dispatch remain the main costs. No precision change, compilation or new runtime dependency was introduced.

The instrumented selected trace captured 13,127 kernels, 234.9 ms kernel activity across 746.5 ms (31.5%), with 511.6 ms in gaps; median gap 18.72 µs, maximum 9.41 ms. Profiling adds overhead, so these activity figures are diagnostic and were not used as a throughput acceptance criterion.

| Chunk events | Streams | Events/s | Peak reserved GiB |
|---:|---:|---:|---:|
| 256 | 1 | 25,223 | 0.16 |
| 256 | 2 | 35,976 | 0.25 |
| 256 | 4 | 43,871 | 0.31 |
| 256 | 8 | 48,178 | 0.49 |
| 512 | 1 | 48,215 | 0.38 |
| 512 | 2 | 67,300 | 0.46 |
| 512 | 4 | 78,249 | 0.54 |
| 512 | 8 | 84,586 | 0.89 |
| 1024 | 1 | 87,916 | 0.66 |
| 1024 | 2 | 111,040 | 0.83 |
| 1024 | 4 | 126,383 | 1.00 |
| 1024 | 8 | 135,260 | 1.77 |
| 2048 | 1 | 136,533 | 1.30 |
| 2048 | 2 | 164,932 | 1.23 |
| 2048 | 4 | 178,979 | 1.87 |

Ruff, compilation, configuration resolution, `uv lock --check --offline` (155 packages) and the MkDocs build passed. `pyproject.toml` and `uv.lock` hashes match the pre-change archive. An installed unrelated `canlab` package has nine pre-existing missing/version-conflicting requirements; the project environment was not altered to repair that package. Documentation dependencies were installed only in a temporary environment from the existing `requirements-docs.txt`.

The A100 allocation ran for 19m08s and was explicitly released after acceptance checks, with two earlier preflight failures totaling 35 seconds (module spelling and a stale batch-shape test). The CPU audit took 5m19s. No multi-GPU allocation, new dataset admission, target evaluation, or full learning run occurred.

Reproduce CPU checks with `python scripts/event_model_gate.py --mode cpu --output <directory>` inside Slurm. `sbatch scripts/slurm/event_model_gate.sh` runs official-CUDA checks, the full bounded matrix, a profile, and the public Ray launch. `gx can verify-events --config configs/experiments/graph_mamba_events_source.yml --output <file>` audits all source inputs.

[Machine-readable evidence](event-pipeline-2026-10-16.json) contains the complete matrix, source verification and launch summary. Full logs, CUDA traces, operator tables and launch artifacts are retained at:

`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-consolidation-2026-10-16/`
