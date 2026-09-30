# Temporal Throughput Baseline

This page records the current speed baseline for the temporal hybrid diagnostic
profile and the constraints for future throughput A/B tests.

## Current Baseline

Use the FP32 stream-lane profile with 16 lanes and 32 events per lane:

```text
configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
```

This is the current speed baseline for `set_01` rich joint
`temporal_hybrid` throughput probes.

The profile keeps:

- dataset: `set_01`
- model: rich joint `temporal_hybrid`
- data path: `batch_mode: stream_lanes`
- train events per full step: `512`
- profiler callback settings
- trainer budget: `limit_train_batches: 100`, `max_epochs: 3`

The current measured baseline from the September 30, 2026 lane-scan optimization
run is:

```text
geometry                         = 16 lanes x 32 chunk
profile/train_events_per_sec     = 2240.772
profile/train_batch_ms           = 228.493
profile/train_data_wait_ms       = 1.205
profile/train_events             = 512
resource/gpu_util_percent_mean   = 17.207
resource/gpu_util_percent_max    = 20.0
resource/gpu_mem_gib_max         = 1.094
resource/rss_gib_max             = 5.900
```

This replaced the September 29, 2026 `16 x 32` baseline of `1519.294
events/sec`. The gain came from reducing scan/state-update overhead, not from
better GPU occupancy: GPU utilization remains around 17-20%, and GPU memory
remains near 1.1 GiB.

## A/B Rules

Equal-work throughput comparisons are valid only when the profiled event count
matches. For 512-event geometry A/Bs in this diagnostic track, require:

```text
profile/train_events = 512
```

Do not compare raw `profile/train_events_per_sec` across runs with different
`profile/train_events` as a drop-in replacement for an equal-work A/B. Smaller
event counts can make batch time look better without measuring the same amount
of work.

Scaling sweeps may intentionally change `profile/train_events`, but they must
report the event count, active lanes, padding fraction, GPU utilization, GPU
memory, RSS, and elapsed time together. Treat those sweeps as capacity and
efficiency probes, not as automatic replacements for the 512-event speed
baseline.

Stream-lane profile configs enforce this invariant with
`expected_train_events: 512` and `fail_on_event_mismatch: true` on
`TemporalBatchProfilerCallback`.

The active lane diagnostics also include
`ResourceUtilizationCallback`, so offline runs record aggregate CPU/RSS and GPU
utilization metrics in `.graphids/mlflow_ingest.json`. The active 8x64 and
16x32 lane probes request `24G` because job `55186678` observed roughly
`4.6 GiB` max RSS; final train/test configs keep their existing resource
requests until the smaller diagnostic allocation is confirmed.

For lane batches, also inspect:

```text
profile/train_active_lanes
profile/train_lane_capacity_events
profile/train_padding_events
profile/train_padding_fraction
```

High padding or too few active lanes means the nominal lane geometry is not
delivering the requested amount of work per profiled step.

## Chunk Scaling Probe

The September 29, 2026 chunk-size sweep kept `stream_lanes: 16` and increased
`chunk_size` from 32 to 64, 128, and 256. The goal was to test whether larger
per-step work would improve GPU utilization on a lightly loaded 16 GiB GPU.

| Geometry | Train events | Events/sec | GPU util mean/max | GPU mem max | RSS max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `16 x 32` | 512 | 1519.294 | 17.2% / 21.0% | 1.090 GiB | 4.152 GiB |
| `16 x 64` | 1024 | 1557.144 | 19.7% / 23.0% | 1.095 GiB | 4.172 GiB |
| `16 x 128` | 2048 | 1563.292 | 20.4% / 22.0% | 1.109 GiB | 4.186 GiB |
| `16 x 256` | 4096 | 1572.428 | 20.5% / 22.0% | 1.133 GiB | 4.216 GiB |

All chunk-sweep runs completed with `profile/train_active_lanes = 16` and
`profile/train_padding_fraction = 0.0`. The larger chunks are stable under the
24 GiB memory request, but they do not materially increase throughput or GPU
occupancy. The observed improvement from `16 x 32` to `16 x 256` is only about
3.5% in events/sec, while wall-clock per step increases roughly in proportion
to the event count.

This means the current temporal hybrid diagnostic is not limited by GPU memory
capacity. It is more likely limited by per-event sequential work, small-kernel
overhead, framework scheduling overhead, or CPU/GPU launch balance.

## Throughput Levers

Prefer throughput experiments that increase useful GPU work per event or reduce
per-step overhead:

- Test a more compute-dense model shape, such as a larger temporal hybrid scale,
  a deeper SSM-lite backbone, or wider embeddings, while keeping the same
  telemetry callback. This checks whether the current diagnostic model is too
  small to occupy the GPU.
- Run a `torch.compile` A/B on the stable `16 x 32` or `16 x 64` geometry.
  Compile is worth testing only with a narrow target and a short wall-time cap.
  Compiling `_forward_with_state` with inductor did not reach the first profiled
  batch after about 10 minutes on job `55205789`, so broad compilation of the
  dynamic scan path is not currently viable.
- Look for fusion opportunities in the temporal hybrid scan/backbone path before
  adding more orchestration. If many small tensor operations run per event,
  batching more events will not fix occupancy as well as reducing kernel count.
- Consider concurrent same-GPU trials only as a separate occupancy experiment.
  It may improve node utilization for independent seeds or hyperparameter
  sweeps, but it does not automatically make a single model train faster and
  adds optimizer, reporting, and failure-isolation complexity if gradients are
  averaged across replicas.

Do not promote `16 x 128` or `16 x 256` as the default diagnostic geometry based
on this sweep. They are valid stress probes, but the small throughput gain does
not justify the longer wall-clock diagnostic cycle.

The first compute-density follow-up also tested a larger/deeper SSM-lite model
on the `16 x 32` geometry:

```text
scale: large
input.hidden: 128
input.embedding_dim: 32
memory.time_encoding_dim: 8
backbone.layers: 2
motif.embedding_dim/time_dim: 16
```

That job completed, but throughput fell to `1405.797 events/sec` and GPU
utilization remained low (`18.4%` mean, `21.0%` max). The larger model did not
make the workload compute-bound. Prioritize fusion/vectorization of the
lane-scan state updates before increasing model width again.

## Implemented Scan Optimizations

The September 30, 2026 scan optimization pass improved the active `16 x 32`
baseline from `1519.294` to `2240.772 events/sec` while preserving:

```text
profile/train_events             = 512
profile/train_active_lanes       = 16
profile/train_padding_fraction   = 0.0
```

The useful changes were:

- clone lane memory/rhythm/motif/backbone state once per chunk instead of once
  per step;
- update active lane rows with vectorized indexed assignments;
- batch rhythm row updates;
- prefetch stream-lane batches through PyTorch `DataLoader` workers with
  `num_workers: 4` and `prebuild_batches: 8`.

An intermediate sparse-overlay version completed as job `55231925`, but it
regressed to `1133.401 events/sec` because CPU dictionary/list bookkeeping ran
inside the CUDA step loop. Do not reintroduce CPU overlay bookkeeping on the
hot path.

The corrected job `55232073` completed in `00:03:45` with:

```text
profile/train_batch_ms           = 228.493
profile/train_data_wait_ms       = 1.205
profile/train_events_per_sec     = 2240.772
resource/gpu_util_percent_mean   = 17.207
resource/gpu_util_percent_max    = 20.0
resource/gpu_mem_gib_max         = 1.094
resource/rss_gib_max             = 5.900
```

This is now the equal-work speed baseline. It is not evidence of improved GPU
occupancy or higher VRAM use. The next throughput work should fuse larger
compute regions in the recurrent scan/backbone path or test a backend that can
run the recurrence as fewer, larger kernels.

## Failed Geometry Probe

The `32 lanes x 16 chunk` probe completed, but it did not preserve the 512-event
invariant on `set_01`:

```text
geometry                         = 32 lanes x 16 chunk
profile/train_events             = 288
profile/train_events_per_sec     = 1605.295
```

The profiler saw only 18 active lanes in the measured batch, so the real event
count was:

```text
18 active lanes x 16 chunk = 288 events
```

That result is invalid as an equal-event speed comparison and must not replace
the `16 lanes x 32 chunk` baseline.

## Guidance

Keep `16 lanes x 32 chunk` as the speed baseline until a new treatment both:

- completes successfully with finite train and validation losses;
- records `profile/train_events = 512`;
- clears the chosen speedup threshold against `16x32`.

Do not increase `stream_lanes` past the number of active streams available in
the profiled training batches unless the loader or dataset sharding is changed
to guarantee 512 real train events per step.

Retired diagnostic configs for slower AMP treatments and the invalid
`32 lanes x 16 chunk` geometry should stay out of the active config tree. The
September 29 empirical note keeps those results as historical provenance.
