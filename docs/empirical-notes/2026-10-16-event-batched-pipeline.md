# Batched event-training pipeline — 2026-10-16

The production preset keeps 16,384-event chunks, one stream and a 16,384-event
optimizer window. It now uses two persistent CPU workers, prefetch factor 2,
pinned memory and synchronous device transfer. CUDA prefetch remains opt-in.
At the unchanged production window, median throughput improved **1.41×**
(243,099 → 343,527 events/s), with the same **40 optimizer updates**.
Peak reserved memory fell from 5.36 to 3.87 GiB.

One A100-PCIE-40GB allocation (`7892928`) ran FP32, seed 42, TF32 disabled and
the pinned official CUDA operators. Each candidate scored the same 655,360
frames across the existing ten reviewed recordings: both vehicles, normal
traffic and attacks, with preceding history replayed from row zero. Three
unprofiled repetitions reset weights, optimizer, caches and RNG after warmup.
All weight hashes matched. Timing synchronizes only at measurement boundaries.

The controlled stream sweep fixes the optimizer window at 163,840 events
(**four updates**). It is separate from the production comparison above.

| Streams | Serial, 0 workers events/s | Batched, 0 workers events/s | Batched, 2 workers events/s | Batched, 2 workers reserved GiB |
|---:|---:|---:|---:|---:|
| 1 | 251,923 | 312,701 | 317,023 | 3.93 |
| 2 | 233,047 | 261,732 | 259,384 | 7.81 |
| 4 | 223,025 | 239,128 | 236,632 | 15.37 |
| 8 | 208,230 | 222,150 | 222,842 | 29.62 |
| 10 | 169,521 | OOM | OOM | failed |

Ten streams failed the enforced 32 GiB ceiling in the batched implementation;
the failure was retained without shrinking the configuration. More streams
remain slower. Packing pads routing indices/masks, while retained attention
activations still grow with the largest routed ID set in each group.

The production-window loader comparison also fixes scored frames and update count:

| Execution | Median events/s | Optimizer step ms | Peak allocated / reserved GiB |
|---|---:|---:|---:|
| Serial, 0 workers | 243,099 | 67.40 | 4.70 / 5.36 |
| Batched, 0 workers | 289,509 | 56.59 | 3.60 / 3.87 |
| Batched, 0 workers + CUDA prefetch | 287,492 | 56.99 | 3.61 / 3.93 |
| Batched, 2 persistent workers | 343,527 | 47.69 | 3.60 / 3.87 |
| Batched, 2 workers + CUDA prefetch | 337,128 | 48.60 | 3.61 / 3.93 |

The initial 64 MiB graph-tile pass measured 238,908 events/s with batched
graph/loss and 255,124 with prepared routing (one stream, controlled window).
It showed no useful worker/prefetch gain. Splitting packed projection banks once
before local tile gathers and using bounded **256 MiB graph tiles** improved
the final path. A dense NumPy routing-scan/transpose experiment regressed to
130,260 events/s and was removed. CPU routing retains the faster row loop.
Mamba's **64 MiB correction workspace**, official operators and tensor math stay intact.

Separate profiles cover one, four and the fastest larger passing candidate,
eight streams. CUDA-event phases cover the entire cohort; operator traces cover
only the first two stream groups. Trace processing was shortened after the
initial full trace proved expensive; profiled throughput never drives selection.

| Streams | Forward + loss s | Backward s | Forward incoming correction μs/event in trace |
|---:|---:|---:|---:|
| 1 | 0.547 | 1.102 | 0.308 |
| 4 | 0.571 | 1.702 | 0.311 |
| 8 | 0.600 | 1.947 | 0.306 |

Backward time grows substantially with streams; traces show index-add, copy,
add and fill work. Forward incoming correction costs are nearly flat per event,
so this pass did not change its tensor operations. The steady source-epoch
projection is 24.82s versus 35.07s; it excludes setup/history replay,
validation and checkpointing, and is not a full training result.

Verification includes serial outputs/losses/caches, input/history/parameter
gradients, forced tiles crossing recording banks, empty/nonzero history,
unseen IDs, ties, arbitrary boundaries/tails, replacement/isolation, causality,
current-payload exclusion, masked/unequal objectives, truncation and exact coverage.
Worker order/shutdown/errors, sync/prefetch dropout RNG and epoch-boundary resume
are covered. Both production and selected public Ray/Lightning smoke fits passed.
The final full suite passed **106 tests**, including official CUDA cases, in
33.05s; lint and diff checks also passed. See `accepted-correctness.log` below.

Full measurements, repetitions, configuration/diff snapshots, failed experiments,
operator tables, traces and smoke logs live outside the repository:

`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-batched-pipeline-2026-10-16/`

Use `summary.json`, `selected-streams/benchmark.json`, `actual-*/benchmark.json`
and `selection.json` there. Reproduce with
`sbatch scripts/slurm/event_model_gate.sh` or, inside one A100 allocation,
`python scripts/event_model_gate.py --mode pipeline --repeats 3 --output <directory>`.
No target evaluation, full source training, dependency changes, compilation or
custom CUDA kernels were introduced. Existing uncommitted work remains uncommitted.

## Subsequent verification correction

During runtime consolidation, `scan(reference=True)` was found to execute the
optimized graph path. Those scan comparisons were therefore not independent;
the recorded throughput and memory measurements are unchanged. The replacement
serial routing/readout/loss oracle and corrected verification are documented in
[the consolidation report](2026-10-16-event-runtime-consolidation.md). Historical
benchmark code is preserved in the external pre-cleanup source snapshot.
