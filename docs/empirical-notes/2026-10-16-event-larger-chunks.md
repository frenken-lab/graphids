# Larger event chunks — 2026-10-16

**16,384 events remains fastest.** Increasing the chunk to 32,768 reduced
throughput by 4.9%; 65,536 reduced it by 16.7%. All candidates passed correctness
and stayed under the enforced 32 GiB reserved-memory ceiling. Production settings
are unchanged: 16,384-event chunks/window, one stream, two persistent workers,
prefetch factor 2, pinned memory and CUDA prefetch disabled.

One A100-PCIE-40GB allocation (`7893790`) compared the current batched pipeline
in FP32 with TF32 disabled, seed 42 and unchanged model dimensions/objectives.
Every configuration scored **the same 655,360 frames**, 65,536 from each of the
same ten reviewed source recordings, covering both vehicles, normal traffic and
attacks. Continuous preceding history was replayed from physical row zero.

The common optimizer window was **65,536 events: exactly ten updates per trial**.
Three independent repetitions reset weights, optimizer, states and RNG after
warmup. Initial weight hashes and row coverage matched. Throughput includes CPU
preparation, transfers, forward/loss, backward and optimizer work; synchronization
occurs only at measurement boundaries. Median unprofiled events/s determines rank.

| Chunk events | Median events/s | Change vs 16K | Optimizer step ms | Peak allocated / reserved GiB |
|---:|---:|---:|---:|---:|
| 16,384 | 369,757 | +0.0% | 177.24 | 3.63 / 3.89 |
| 32,768 | 351,611 | -4.9% | 186.39 | 7.18 / 7.43 |
| 65,536 | 307,849 | -16.7% | 212.88 | 14.28 / 14.78 |

Repetition ranges were 368,137–370,809, 351,060–353,496 and 304,078–308,858
events/s respectively. The ranking is consistent across repetitions.

Separate profiling runs measured these CUDA-event phases over the same cohort;
operator traces cover the first two stream groups and do not drive selection:

| Chunk events | Forward + loss s | Backward s |
|---:|---:|---:|
| 16,384 | 0.607 | 1.131 |
| 32,768 | 0.535 | 1.244 |
| 65,536 | 0.548 | 1.493 |

The additional backward cost outweighs reduced forward overhead. Each tiled
gather's backward accumulates into a recording bank that grows with chunk length;
the fixed temporary-workspace bound does not remove this gradient-buffer work.
Larger chunks also extend the truncated-gradient horizon; this was a throughput
and correctness experiment, with no training-quality or target evaluation.
The 65,536-event comparison window is separate from production's 16,384-event window.

The gate gained `--events-per-recording` to keep the timed cohort fixed. Its
usual two-chunk segment minimum would otherwise change the work for 64K chunks.
Full-chunk nonzero-history parity diagnostics may read additional frames from
the same reviewed source recordings outside that timed cohort. Routing capacity
was raised to **8,388,608 graph tokens** and the microbatch cap to **65,536** only
in the benchmark; model dimensions and the 32 GiB ceiling were unchanged.

The event regression suite passed **62 tests**, including official CUDA cases.
At every tested chunk length, outputs, objective gradients and outgoing graph/
Mamba caches matched the serial functional reference with the gate's existing
tolerances. Exact scored coverage and optimizer-step counts were asserted.
Lint and diff checks passed. No production configuration or event-cache identity changed.

Full repetitions, correctness logs, configuration/diff snapshots and phase/operator
traces are saved outside the repository:

`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-larger-chunks-2026-10-16/`

Reproduce inside a one-A100 Slurm allocation:

```bash
python scripts/event_model_gate.py --mode benchmark \
  --chunks 16384 32768 65536 --streams 1 --workers 2 --repeats 3 \
  --control-chunk 16384 --control-streams 1 \
  --max-microbatch-events 65536 --effective-events 65536 \
  --events-per-recording 65536 --steps 10 --no-profiles --output <directory>
```

## Subsequent verification correction

During runtime consolidation, `scan(reference=True)` was found to execute the
optimized graph path. Those scan comparisons were therefore not independent;
the recorded throughput and memory measurements are unchanged. The replacement
serial routing/readout/loss oracle and corrected verification are documented in
[the consolidation report](2026-10-16-event-runtime-consolidation.md). Historical
benchmark code is preserved in the external pre-cleanup source snapshot.
