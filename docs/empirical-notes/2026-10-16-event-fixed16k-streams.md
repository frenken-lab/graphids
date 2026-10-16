# Fixed 16,384-event chunks with additional streams — 2026-10-16

The user requested holding chunk length at 16,384 and adding independent streams. This sweep therefore removed the prior 16,384-event total microbatch restriction: it tested 1, 2, 4, 8 and 10 streams, allowing 163,840 events per microbatch. The reviewed source cohort has only ten distinct training recordings, so no recordings were duplicated to invent more streams.

**One stream remained fastest.** Increasing streams consumed more memory and increased backward time. The source preset remains 16,384 events × one stream, with its existing 16,384-event optimizer window. The larger common optimizer window below belongs only to the controlled benchmark.

All candidates used the same 163,840-event optimizer window, with accumulation when their microbatch was smaller. Each repetition scored the same 655,360 frames: 65,536 per reviewed recording, including normal traffic and positive attack frames from both vehicles. Earlier recording history was replayed continuously from physical row zero, then cloned separately per configuration. Three repetitions per case reset weights, optimizer, caches and RNG; median end-to-end events/second determined ranking. All initial weight hashes matched the prior searches.

Training used an NVIDIA A100-PCIE-40GB in Slurm allocation `7892096`, FP32, the pinned official CUDA operators, unchanged model dimensions/objectives, and seed 42. The allocator was limited to 32 GiB before correctness checks and timing, so the search could not consume the remaining physical VRAM. The graph-token capacity remains bounded at 2,097,152 per stream; scan workspaces remain bounded at 64 MiB.

| Streams | Events per full microbatch | Median events/s | Peak reserved GiB | Change from one stream |
|---:|---:|---:|---:|---:|
| 1 | 16,384 | 243,997 | 5.43 | +0.0% |
| 2 | 32,768 | 226,046 | 8.99 | -7.4% |
| 4 | 65,536 | 217,867 | 14.03 | -10.7% |
| 8 | 131,072 | 204,041 | 24.80 | -16.4% |
| 10 | 163,840 | 167,868 | 31.87 | -31.2% |

These figures compare the same GPU, source frames and optimizer-window size; they should not be treated as a direct percentage comparison with the previous 16,384-event-window run. The ten-stream case came within about 0.13 GiB of the ceiling and passed. All other cases also passed; no configuration was rejected on memory or parity. Because there are ten recordings, larger requested stream counts naturally shrink when recordings finish. Groups also split between independent streams to fill complete optimizer windows; no chronological chunk is split, padded, repeated or dropped.

| Streams | Forward + objectives, all 655,360 frames | Backward, all 655,360 frames |
|---:|---:|---:|
| 1 | 1.553 s | 1.083 s |
| 2 | 1.483 s | 1.365 s |
| 4 | 1.460 s | 1.498 s |
| 8 | 1.479 s | 1.682 s |
| 10 | 1.507 s | 2.343 s |

Forward time was roughly flat while backward time grew from about 1.08 seconds to 2.34 seconds. In this implementation, the additional batching and larger working set did not improve total training throughput. These timings identify the backward path as the principal loss in scaling; they do not by themselves isolate one hardware cause. More free memory was not sufficient to make additional streams faster.

Every candidate passed outputs, nonzero incoming caches and parameter gradients against independent serial execution. All timed repetitions asserted chronological row contiguity and complete exact scored coverage. CPU tests passed 81 cases with six CUDA cases skipped; the CUDA event gate passed 43 cases, covering all six CUDA-only cases and the newly configurable caps. Together these cover all 87 unique regression cases.

The data schema now exposes `max_microbatch_events` explicitly (default 16,384, supported up to 262,144), and requires each requested microbatch to fit that cap and `effective_batch_events`. This permits a fixed chunk length with additional streams without silently changing the cap or losing frames. Neither the input-cache identity nor the learned architecture changes. No new dependency, precision mode, compilation, target evaluation or full learning run was introduced.

Reproduce inside a one-A100 Slurm allocation:

```bash
python scripts/event_model_gate.py --mode benchmark --steps 4 --repeats 3 \
  --chunks 16384 --streams 1 2 4 8 10 \
  --control-chunk 16384 --control-streams 1 \
  --max-microbatch-events 163840 --effective-events 163840 \
  --output <directory>
```

[Complete measurements](event-fixed16k-streams-2026-10-16.json) retain every repetition, row coverage, memory peak and phase breakdown. Logs and CPU/CUDA profiles are preserved under:

`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-fixed16k-streams-2026-10-16/`
