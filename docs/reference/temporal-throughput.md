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

The measured baseline from the September 29, 2026 geometry A/B was:

```text
geometry                         = 16 lanes x 32 chunk
profile/train_events_per_sec     = 1513.168
profile/train_batch_ms           = 338.363
profile/train_data_wait_ms       = 13.929
profile/train_events             = 512
```

## A/B Rules

Throughput comparisons are valid only when the profiled event count matches.
For this diagnostic track, require:

```text
profile/train_events = 512
```

Do not compare raw `profile/train_events_per_sec` across runs with different
`profile/train_events`. Smaller event counts can make batch time look better
without measuring the same amount of work.

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
