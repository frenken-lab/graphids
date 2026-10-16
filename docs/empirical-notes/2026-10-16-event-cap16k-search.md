# Expanded 16,384-event single-A100 search — 2026-10-16

The user requested doubling the microbatch cap from 8,192 to 16,384 events and searching larger combinations. The expanded search selected **16,384 events × one stream**, with a 16,384-event optimizer window. Longer chronological chunks won this search despite substantial free GPU memory. This is a throughput selection; no full learning run, target evaluation or accuracy tuning was performed.

The implementation now admits 1–16 streams under the doubled cap. The reviewed source-training cohort contains ten distinct recordings, so benchmarks used up to ten actual independent recordings and did not replicate recordings to invent extra streams. Stream groups can split between independent streams at optimizer-window boundaries, preserving each chronological chunk and giving the ten-stream trial the same complete target window. A regression test checks complete, unrepeated coverage for that schedule.

The matrix used FP32 on an NVIDIA A100-PCIE-40GB in Slurm allocation `7891746`. Each configuration ran three independently reset repetitions of 20 optimizer updates, scoring the same 327,680 frames (32,768 per recording). Both vehicles, verified normal traffic and positive attack frames are included. Incoming continuous history was replayed from physical row zero, then cloned separately per trial. All configurations share initial weight SHA256 `e7f7803385e520e1c0581f161241989fe6ef89557383f945f6dadc513d00541f`. Source identities, model dimensions, objectives, seed 42 and checkpoint metric are unchanged.

The comparison control is the previous 2,048 × four-stream winner, remeasured with the same larger cohort and 16,384-event optimizer window. Median throughput across repetitions selects the winner. All 15 configurations passed batched-versus-serial outputs, nonzero caches and gradient checks. Every repetition remained below the existing 32 GiB reserved-memory ceiling. A separate run with explicit row-contiguity and complete-cohort assertions reproduced the winner and verified all 327,680 scored frames exactly once.

| Chunk events | Streams | Median events/s | Peak reserved GiB |
|---:|---:|---:|---:|
| 2048 | 4 | 178,360 | 1.85 |
| 1024 | 1 | 84,322 | 0.40 |
| 1024 | 2 | 107,752 | 0.58 |
| 1024 | 4 | 122,572 | 0.96 |
| 1024 | 8 | 132,439 | 1.67 |
| 1024 | 10 | 137,566 | 2.10 |
| 2048 | 1 | 132,876 | 0.67 |
| 2048 | 2 | 161,445 | 1.18 |
| 2048 | 8 | 181,802 | 3.26 |
| 4096 | 1 | 188,261 | 1.32 |
| 4096 | 2 | 202,559 | 2.30 |
| 4096 | 4 | 208,230 | 3.57 |
| 8192 | 1 | 224,932 | 2.63 |
| 8192 | 2 | 219,545 | 4.56 |
| 16384 | 1 | 237,272 | 5.46 |

The selected configuration measured **237,272 events/s**, versus **178,360** for the control: **33.0% faster**. Median optimizer-window latency was 69.05 ms and peak reserved memory was 5.46 GiB. The three winning repetitions measured 237,160, 237,272, 237,606 events/s.

A training-only projection for 8,526,701 source training frames is 35.9 seconds. It excludes cache/source setup, initial history replay, validation, checkpoint I/O and logging, and is not a measured full epoch. Timing and later recording behavior can change it.

The graph-token capacity increases explicitly from 1,048,576 to 2,097,152 to admit a 16,384-frame chunk with the largest reviewed source ID set (98 IDs). The pinned CUDA operators, bounded 64 MiB scan workspaces, FP32 and independent 32 GiB GPU-memory ceiling remain unchanged. No precision changes, compilation or dependencies were introduced. The selected chunk also lengthens the truncated-backpropagation horizon; the effective optimizer window doubles. These changes have not been evaluated for detection quality. The independent confirmation, using the default allocator rather than the production expandable-segment setting, reserved 6.96 GiB; it also remained below the ceiling.

| Selected optimizer-window phase | Time |
|---|---:|
| CPU preparation | 0.687 ms |
| Host-to-device transfer | 0.313 ms |
| Forward and objective | 38.870 ms |
| Backward | 27.119 ms |
| Clipping and AdamW | 1.458 ms |

The instrumented selected trace recorded 20,955 kernels, 760.0 ms kernel activity over 1689.4 ms (45.0%). Median inter-kernel gap was 1.28 µs. Profiling overhead affects these values; throughput, correctness and memory determined acceptance.

The full Slurm regression suite passed **82 tests**, including five newly accepted larger-cap combinations and the ten-stream accumulation schedule. The public Ray/Lightning smoke fit also passed with the selected 128-wide, two-layer model, 16,384-event chunk and 16,384-event optimizer-window settings; its complete three-frame synthetic validation coverage was verified. Configuration validation, Ruff, compilation and the MkDocs build passed. Runtime dependencies and lockfiles are unchanged.

Reproduce the search inside a one-A100 Slurm allocation:

```bash
python scripts/event_model_gate.py --mode benchmark --steps 20 --repeats 3 \
  --max-microbatch-events 16384 --effective-events 16384 \
  --chunks 1024 2048 4096 8192 16384 --streams 1 2 4 8 10 \
  --output <directory>
```

`--mode gpu` additionally runs a small public Ray/Lightning fit using the selected preset against synthetic reviewed recordings. Full logs, profiles and public-launch artifacts live under:

`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-cap16k-2026-10-16/`

[Machine-readable measurements](event-cap16k-2026-10-16.json) retain every repetition and the complete coverage confirmation. The [original 8,192-event-cap results](2026-10-16-event-pipeline-consolidation.md) remain unchanged.
