# Temporal Throughput Baseline

> Historical event-model architecture. The current path is documented in [Graph Mamba](graph-mamba.md).

This page records the current speed baseline for the temporal hybrid diagnostic
profile and the constraints for future throughput A/B tests.

## Current Baseline

Use the FP32 strict-rich stream-lane profile with 16 lanes and 32 events per
lane:

```text
configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_strict_rich_profile.yml
```

This is the current speed baseline for `set_01` rich joint
`temporal_hybrid` throughput probes. The non-strict
`temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml`
remains the base geometry profile for historical comparisons.

The profile keeps:

- dataset: `set_01`
- model: rich joint `temporal_hybrid`
- data path: `batch_mode: stream_lanes`
- train events per full step: `512`
- profiler callback settings
- trainer budget: `limit_train_batches: 100`, `max_epochs: 3`

The current measured strict rich baseline from SLURM job `55280466` is:

```text
geometry                         = 16 lanes x 32 chunk
profile/train_events             = 512
profile/train_padding_fraction   = 0.0
final events/sec                 = 2873.71
epoch 2 median events/sec        = 2887.05
train_loss                       = 1.125
val_loss                         = 1.016
gpu util mean/max                = 16.2% / 19.0%
gpu mem max                      = 1.142 GiB
rss max                          = 31.959 GiB
```

This replaced the October 1, 2026 fused-context baseline of
`2825.70 events/sec`, improving final throughput by about `1.7%` and the epoch
2 median by about `2.0%`. Against the September 30, 2026 feature-ablation rich
strict baseline of `2568.81 events/sec`, the final gain is about `11.9%`. The
gain came from reducing scan/state-update overhead, not from lane widening. The
writer-fusion lane scan is now the single lane-batch implementation; the public
`model.scan` surface is profiling-only.

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

## Active Follow-up Diagnostics

The next diagnostic configs keep the `16 lanes x 32 chunk` and
`profile/train_events = 512` invariant:

- `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_phase_profile.yml`
  enables opt-in phase timing for encode, scan, heads, forward, loss, and
  heads+loss. Use it before changing scan/backbone code.
- `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_workers0_profile.yml`
  disables DataLoader workers and prefetching to confirm whether the current
  prebuild path is still buying measurable throughput.
- `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_persistent_profile.yml`
  keeps `num_workers: 4` and `prebuild_batches: 8`, but enables persistent
  workers to isolate worker startup/reuse effects.

Promote a data-side treatment only if it preserves finite losses and improves
equal-work `profile/train_events_per_sec` against the current `2873.71`
events/sec strict rich baseline. If phase timing shows scan time dominates while
`profile/train_data_wait_ms` stays small relative to batch time, stop increasing
prefetch depth and target recurrent scan/backbone fusion instead.

### September 30 Follow-up Results

The first submission exposed a diagnostic-data issue: multi-worker stream-lane
runs could surface a partial lane batch (`352` events) while
`fail_on_event_mismatch` expected `512`. The stream-lane loader now has an
opt-in `drop_partial_lane_batches` switch, and the active lane diagnostic
configs enable it.

Successful resubmissions:

| Config | SLURM job | Events/sec | Batch ms | Data wait ms | GPU util mean/max | GPU mem max | RSS max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lanes16_chunk32_profile` | `55234519` | `2233.770` | `229.209` | `1.168` | `15.9% / 20.0%` | `1.159 GiB` | `52.513 GiB` |
| `lanes16_chunk32_phase_profile` | `55234504` | `2175.233` | `235.377` | `1.166` | `16.4% / 20.0%` | `1.159 GiB` | `37.683 GiB` |
| `lanes16_chunk32_persistent_profile` | `55234505` | `2190.822` | `233.702` | `1.142` | `15.7% / 20.0%` | `1.159 GiB` | `52.509 GiB` |
| `lanes16_chunk32_workers0_profile` | `55234298` | `2276.320` | `224.924` | `14.344` | `16.6% / 20.0%` | `1.142 GiB` | `29.244 GiB` |

The phase profile shows the forward pass is dominated by scan time:

```text
profile/train_phase_encode_ms      = 0.389
profile/train_phase_scan_ms        = 111.984
profile/train_phase_heads_loss_ms  = 1.954
profile/train_phase_forward_ms     = 113.723
```

Conclusion: worker prefetching still reduces measured data wait, but it does
not improve final equal-work throughput versus the no-worker diagnostic on this
run. The bottleneck remains inside the recurrent scan/backbone path, not CPU
batch construction.

## Historical Scan Backend Diagnostics

This historical scan-throughput pass tested an opt-in backend matrix that has
since been retired from the active config/API surface:

- `backend: lane_fastpath` uses the common all-valid lane chunk shape to avoid
  active-lane gather/scatter and uses flattened lane-ID row addressing.
- `backend: fused_memory` adds the lane fast path plus batched TGN memory source
  and destination GRU updates.
- `backend: compiled_inner` adds the fused-memory path and applies a narrow
  `torch.compile` A/B to the batched memory update on CUDA, with CPU fallback.

Diagnostic configs:

- `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_lane_fastpath_profile.yml`
- `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_memory_profile.yml`
- `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_compiled_inner_profile.yml`

All preserved `16 lanes x 32 chunk`, `profile/train_events = 512`, and
`drop_partial_lane_batches: true`. They also enabled phase timing and scan
subphase timing. The retired configs remain benchmark history only; do not add
these backend names to new active configs.

### September 30 Scan Backend Submissions

Submitted on September 30, 2026:

| Config | SLURM job | Treatment |
| --- | ---: | --- |
| `lanes16_chunk32_lane_fastpath_profile` | `55237253` | all-valid lane fast path, flattened lane-ID row addressing, scan subphase timing |
| `lanes16_chunk32_fused_memory_profile` | `55237254` | lane fast path plus batched TGN source/destination GRU memory update |
| `lanes16_chunk32_compiled_inner_profile` | `55237252` | fused-memory path plus narrow CUDA `torch.compile` A/B for the batched memory update |

All three top-level jobs completed with exit code `0:0`:

| Config | SLURM job | State | Elapsed |
| --- | ---: | --- | ---: |
| `lanes16_chunk32_lane_fastpath_profile` | `55237253` | `COMPLETED` | `00:03:56` |
| `lanes16_chunk32_fused_memory_profile` | `55237254` | `COMPLETED` | `00:03:53` |
| `lanes16_chunk32_compiled_inner_profile` | `55237252` | `COMPLETED` | `00:04:01` |

Final logged metrics:

| Config | Events/sec | Batch ms | Data wait ms | Scan ms | GPU util mean/max | GPU mem max | RSS max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lane_fastpath` | `2218.192` | `230.819` | `1.472` | `117.792` | `13.2% / 16.0%` | `1.159 GiB` | `40.886 GiB` |
| `fused_memory` | `2157.131` | `237.352` | `1.327` | `119.900` | `14.6% / 18.0%` | `1.159 GiB` | `37.319 GiB` |
| `compiled_inner` | `2325.474` | `220.170` | `0.995` | `120.619` | `11.8% / 18.0%` | `1.194 GiB` | `48.216 GiB` |

Final scan subphase timings:

| Config | Memory ctx | Rhythm ctx | Motif ctx | Backbone | Memory update | Rhythm update | Motif update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lane_fastpath` | `19.567` | `19.208` | `14.715` | `14.068` | `13.206` | `14.827` | `6.964` |
| `fused_memory` | `17.980` | `18.529` | `13.709` | `13.785` | `20.820` | `13.961` | `6.472` |
| `compiled_inner` | `18.866` | `18.799` | `13.990` | `14.182` | `18.015` | `14.609` | `6.824` |

Status: **do not promote by default yet**. `compiled_inner` beat the
`2233.770 events/sec` baseline, but measured scan time rose from the phase
baseline (`111.984 ms`) to `120.619 ms`; the improvement may be noise, lower
data wait, timing-instrumentation effects, or compile-side behavior rather than
a confirmed recurrent-scan win. `lane_fastpath` and `fused_memory` were slower
than the current baseline. The subphase breakdown shows memory/rhythm context
and updates remain the largest fused targets.

### Context and Update Micro-optimization Submissions

The next code pass targeted those scan subphases directly:

- batched source/destination elapsed-time memory context uses one time-encoder
  call and a preallocated common-path context tensor;
- batched TGN memory update reuses the GRU input projection for source and
  destination updates;
- batched rhythm context/update helpers avoid small stack/cat allocations.

Validation:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
25 passed

uv run ruff check graphids/core/models/temporal/_hybrid/contexts.py graphids/core/models/temporal/_hybrid/scan.py tests/core/models/test_temporal_hybrid.py
All checks passed
```

Submitted on September 30, 2026. All completed at the top level with exit code
`0:0`:

| Config | SLURM job | State | Elapsed |
| --- | ---: | --- | ---: |
| `lanes16_chunk32_lane_fastpath_profile` | `55238250` | `COMPLETED` | `00:03:49` |
| `lanes16_chunk32_fused_memory_profile` | `55238251` | `COMPLETED` | `00:03:30` |
| `lanes16_chunk32_compiled_inner_profile` | `55238252` | `COMPLETED` | `00:04:23` |

Final logged metrics:

| Config | Events/sec | Batch ms | Data wait ms | Scan ms | GPU util mean/max | GPU mem max | RSS max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lane_fastpath` | `2267.660` | `225.783` | `1.188` | `114.537` | `14.2% / 18.0%` | `1.159 GiB` | `94.271 GiB` |
| `fused_memory` | `2121.101` | `241.384` | `1.344` | `123.222` | `14.6% / 18.0%` | `1.159 GiB` | `47.725 GiB` |
| `compiled_inner` | `2280.270` | `224.535` | `0.986` | `122.151` | `11.7% / 17.0%` | `1.192 GiB` | `36.766 GiB` |

Final scan subphase timings:

| Config | Memory ctx | Rhythm ctx | Motif ctx | Backbone | Memory update | Rhythm update | Motif update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lane_fastpath` | `18.421` | `19.788` | `13.421` | `13.449` | `12.190` | `16.150` | `6.610` |
| `fused_memory` | `19.295` | `19.929` | `13.751` | `13.364` | `19.052` | `16.141` | `6.713` |
| `compiled_inner` | `18.633` | `19.979` | `13.711` | `13.860` | `17.778` | `16.568` | `6.627` |

Status: **repeat before promoting**. `lane_fastpath` and `compiled_inner`
both beat the `2233.770 events/sec` baseline, but gains are modest and scan
time still does not beat the earlier phase baseline (`111.984 ms`).
`lane_fastpath` is the cleaner candidate because it avoids compile overhead and
reduced memory-update time, but its RSS max was an outlier. Repeat
`lane_fastpath` without scan subphase timing before making it the default.

## Lane-Scaling Workers0 Diagnostics

The next diagnostic isolated lane scaling with the flattened lane scan and no
phase or scan subphase timing. All three configs used `num_workers: 0`,
`prebuild_batches: 0`, `pin_memory: false`, `drop_partial_lane_batches: true`,
and `fail_on_event_mismatch: true`.

Submitted on September 30, 2026:

| Config | SLURM job | Geometry | Expected events | State | Elapsed |
| --- | ---: | --- | ---: | --- | ---: |
| `lanes32_chunk16_lane_fastpath_workers0_profile` | `55240966` | `32 x 16` | `512` | `COMPLETED` | `00:01:21` |
| `lanes64_chunk8_lane_fastpath_workers0_profile` | `55240967` | `64 x 8` | `512` | `COMPLETED` | `00:01:25` |
| `lanes32_chunk32_lane_fastpath_workers0_profile` | `55240968` | `32 x 32` | `1024` | `COMPLETED` | `00:01:21` |

Outcome: all three runs produced zero training batches. Lightning reported:

```text
Total length of `TemporalStreamLaneLoader` across ranks is zero.
`Trainer.fit` stopped: No training batches.
```

No profiler metrics or final losses were logged for these runs, so they fail
the validity rule: `profile/train_events` is absent rather than equal to the
expected event count, and `profile/train_padding_fraction` is absent rather than
`0.0`. The `64 x 8` job logged one system-metrics sample before exit
(`resource/gpu_util_percent_mean = 0.0`, `resource/gpu_mem_gib_max = 1.093`,
`resource/rss_gib_max = 28.033`), but that is startup/idle telemetry, not
training throughput.

Conclusion: under strict full-batch diagnostics, the `set_01` stream
distribution cannot support widened lane geometries at 32 or 64 lanes with the
current loader. Do not promote lane widening or add worker variants from this
matrix. Further lane-scaling tests need a loader/dataset change that can
guarantee full widened-lane batches before throughput is meaningful.

## Strict Feature Ablation Results

The September 30, 2026 strict feature-ablation matrix is the current reference
for rich feature cost. All runs reached `max_epochs=3` with `train_events=512`
and `train_padding_fraction=0.0`. Job `55246546` then promoted the cleanup
baseline for the full rich variant.

| Variant | SLURM job | Epoch 2 median eps | Final eps | Train loss | Val loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rich` | `55242488` | `2602.36` | `2568.81` | `1.746` | `1.459` |
| `minus_motif` | `55242495` | `3128.97` | `3129.65` | `1.264` | `1.114` |
| `minus_rhythm` | `55242500` | `3311.96` | `3315.82` | `1.325` | `1.216` |
| `minus_memory` | `55242508` | `4230.88` | `4230.16` | `1.554` | `1.239` |
| `memory_only` | `55242513` | `4023.18` | `4021.14` | `1.310` | `1.123` |
| `backbone_only` | `55242520` | `20275.07` | `20169.79` | `1.495` | `1.188` |
| `backbone_none` | `55242564` | `45038.00` | `45907.58` | `2.248` | `2.029` |
| `rich cleanup` | `55246546` | `2772.26` | `2755.05` | `1.147` | `1.030` |
| `rich fused_context` | `55279425` | `2829.22` | `2825.70` | `1.330` | `1.167` |
| `rich writer_fusion` | `55280466` | `2887.05` | `2873.71` | `1.125` | `1.016` |

The original rich strict run improved over the previous `2267.660 events/sec`
reference by about `13.3%` on final throughput and `14.8%` on the epoch 2
median. The phase probe job `55242574` finished at `2404.17 events/sec` with
subphase timing enabled and measured `scan = 108.25 ms` inside
`forward = 110.10 ms`.

The cleanup run improved the full rich result from `2568.81` to `2755.05`
events/sec (`+7.2%`) while preserving the 512-event, zero-padding invariant.
The fused-context run improved the cleanup result to `2825.70` events/sec
(`+2.6%`) with the same invariant.
The writer-fusion run improved the fused-context result to `2873.71` events/sec
(`+1.7%`) while preserving the same invariant.

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

Retired diagnostic configs for slower AMP treatments and the old non-strict
invalid `32 lanes x 16 chunk` probe should stay out of the active config tree.
The September 29 empirical note keeps those results as historical provenance;
the strict workers0 lane-scaling configs above remain only as unsupported
geometry diagnostics.

## Clear Scan Improvement Targets

The current rich lane implementation already does the first round of obvious
fusion: `_lane_step_context` computes shared `flat_src`/`flat_dst` row indices
once, uses flattened memory/rhythm state in the lane fast path, and carries
gathered rows forward to `_write_lane_step_updates`. The main loop also clones
lane state once per chunk rather than once per step. See
`graphids/core/models/temporal/_hybrid/scan.py`.

The remaining profile is dominated by 32 recurrent iterations of small indexed
gathers, projections, updates, and writes. The strict phase probe measured:

| Subphase | Time |
| --- | ---: |
| memory context | `18.21 ms` |
| memory update | `10.39 ms` |
| rhythm context | `19.01 ms` |
| rhythm update | `14.62 ms` |
| motif context | `12.66 ms` |
| motif update | `6.05 ms` |
| backbone | `12.78 ms` |

Prioritize targets in this order:

1. **Fused rich context path.** Combine memory, rhythm, and motif context
   construction into one helper that returns the summed context and all update
   rows. This is partly structured already, but the three modules still launch
   separate gather/projection paths each step.
2. **Fused update/write path.** Memory, last-seen, rhythm, motif, and backbone
   state writes touch different tensors after `z_t` exists. A narrower writer
   should reduce per-step assignment overhead and avoid extra motif clones.
3. **Motif rolling-state update.** The current motif update clones
   `current_ids` and `current_iats` per step before writing shifted rows. This
   is a small but clean target because it is pure lane-local state.
4. **Rhythm context/update microkernel.** Rhythm is cheap mathematically but
   still costs about `33.6 ms` combined in the phase probe, suggesting launch
   and small tensor overhead. It is a good candidate for a custom indexed
   elementwise kernel if Python/PyTorch fusion stalls.
5. **Backbone/state write separation.** Backbone math is only about
   `12.8 ms`; do not redesign it first. The better target is reducing how many
   small operations surround each `backbone.step`.

Do not prioritize lane widening on `set_01`; strict widened-lane diagnostics
produce no full training batches or invalid partial batches. Do not pursue broad
`torch.compile` of the dynamic scan as the main path; the useful compile result
was narrow and noisy, while broad compile failed to reach useful profiler output
within the short diagnostic budget. Consider Triton/CUDA only after the Python
rich context/update fusion has been measured and scan remains dominant.

### Working-Tree Cleanup Status

The first cleanup pass is implemented in
`graphids/core/models/temporal/_hybrid/scan.py` and locally verified:

- flattened source/destination lane row indices are precomputed once per chunk;
- full-chunk all-valid lane-fastpath status is computed once per chunk;
- subphase timing calls are gated out of the hot path when disabled;
- motif lane state is shifted in place instead of cloning active rows.

Verification:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
31 passed, 4 warnings

uv run ruff check graphids/core/models/temporal/_hybrid/scan.py tests/core/models/test_temporal_hybrid.py
All checks passed
```

The first submitted benchmark for this cleanup, SLURM job `55246546`, completed
successfully and became the cleanup strict-rich baseline:

```text
final events/sec          = 2755.05
epoch 2 median events/sec = 2772.26
profile/train_events      = 512
padding fraction          = 0.0
```

Next implementation work should fuse the rich context builder and post-backbone
state writer before considering custom kernels. Also use a unique Ray output
name or clear the prior output directory for repeated A/Bs, because job
`55246546` reused a directory that already contained Ray checkpoint metadata.

The first fused-context implementation pass was benchmarked with:

```text
configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_profile.yml
```

It keeps the current strict geometry and uses a distinct experiment name so the
Ray run does not reuse `55246546` checkpoint metadata. SLURM job `55279425`
completed successfully and became the fused-context strict-rich baseline.

The follow-up writer-fusion pass was benchmarked with:

```text
configs/experiments/diagnostics/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_writer_fusion_profile.yml
```

SLURM job `55280466` completed successfully with `512` train events and
zero padding, improving the final strict-rich baseline to `2873.71 events/sec`.

## Full-Coverage Training and Experimental Sections

Production lane training defaults to `coverage_policy: full` and
`stream_execution: continuous`. A recording owns its lane state until completion;
32-event chunks detach gradients but do not reset GRU memory, backbone state,
rhythm statistics, or motif history. Chunks are therefore not independent examples.
The exact `set_01` comparison uses 18 lanes × 32 events so all recordings start
concurrently. Short streams and tails are included; padding affects neither state
nor losses. Event-mode loading retains its existing behavior.

Full coverage requires `drop_partial_lane_batches: false`, unrestricted
`limit_train_batches: 1.0`, `max_steps: -1`, sequential ordered sampling, and one
training process/GPU. Distributed index sharding, overfit sampling, and fast-dev
limits are rejected. Worker prefetch only materializes ordered map-style plans;
workers never own recurrent state. Deliberately limited smoke/profile configs use
`coverage_policy: diagnostic`. Dropped diagnostic chunks count as history gaps and
force affected lanes to reset before resuming.

`stream_execution: segmented` is explicitly experimental. Cores own at most
`segment_events: 2048` unique scored events. Each section replays at most
`segment_warmup_events: 256` preceding training events from the same recording,
starts from reset state, and resets at section completion. The deterministic
seed-42 shuffled queue allows different sections of one recording in different
lanes; chunks within each section remain chronological. Replay advances state
but contributes neither loss nor scored coverage. Warmup-only batches skip backward
and optimizer calls. Mixed batches use only scored events in their loss.
Gradients detach at every chunk boundary, including replay chunks.

Finite replay approximates full-recording state: it cannot recover arbitrary GRU
memory or cumulative rhythm statistics. Segmented training stays opt-in regardless
of pilot throughput or validation results. A section end is distinct from a real
recording end. Loader metadata supplies source indices and actual recording
successor IDs/payloads, including chunk and section boundaries. Successors never
cross recording/split terminals; terminal events still train classification.
Legacy synthetic batches without metadata retain batch-local successor labels.
The existing IAT target values/definition are unchanged.

Every full lane run automatically attaches `TemporalCoverageAudit`. It checks core
and chunk ownership before training and uses CPU bitmaps to count scored source
indices during each epoch. Reports under `coverage/epoch-N.json` include expected,
observed, unique, missing, duplicate, unexpected, per-stream and per-class coverage,
replay events, successor coverage, occupancy, padding, and end-to-end scored
throughput. Incomplete production epochs fail. Exceptions persist interrupted
reports; report-writing failures warn while preserving the original exception.

Comparison configs are in `configs/experiments/comparisons/`. Run them on one GPU
with `python -m graphids.exp.temporal_comparison CONFIG --output DIR`. The runner
loads only the training/validation caches, verifies 12,090,403 training events,
18 recordings, and 57,108 attacks, seeds before constructing the model, saves
`checkpoints/final.ckpt`, then restores it for continuous lane evaluation on the
first 8,192 validation events per recording. The existing 64-event scoring warmup
remains unscored. Evaluation drops no batches and loads no test data. Reports
include initial weight digests, validation losses, class counts, AUROC availability,
and GPU telemetry. The expected recording-successor count is 12,090,385.

Choose 64 section lanes only if the short diagnostics improve end-to-end **scored**
throughput by at least 3%; otherwise choose 32. Replay is reported separately.
Full epoch jobs request separate Pitzer allocations with one GPU, eight CPUs,
32 GiB RAM, and eight hours. Compare the two full-coverage workloads directly.
The historical strict 512-event prefix probes above are diagnostic workloads and
are not equivalent training epochs. One epoch per mode is a quality pilot and
cannot establish that approximate state should replace continuous training.

### Wider Section-Lane Sweep

`python -m graphids.exp.temporal_lane_sweep OUTPUT_DIR` repeats 64 lanes as a
baseline, then measures 128, 256, and 512 lanes sequentially on the same GPU.
It tries 1,024 lanes only when 512 improves scored throughput over 256 by at least
3%. Every geometry owns the same first 2,048 sections of the seed-42 queue; chunks
remain 32 events and replay remains at most 256 events. The comparison runner's
`--section-limit` is accepted only for diagnostic segmented runs. It truncates
section ownership rather than imposing geometry-dependent batch limits.

The sweep verifies identical initial weights, queue fingerprints, planned scored
and replay events, and eligible successor coverage. Each geometry runs in a fresh
process and saves its checkpoint and continuous bounded validation result. Reports
include end-to-end scored throughput, padding, GPU utilization, NVML VRAM usage,
and PyTorch peak allocated/reserved memory. Use
`scripts/slurm/temporal_lane_sweep.sbatch` with a unique `SWEEP_ROOT` to run the
checks and sweep on Pitzer. An optional `SWEEP_CODE_ROOT` selects an immutable code
snapshot; the default is the project checkout.

The selected lane count is a performance candidate. Production geometry remains
unchanged: increasing the scored batch size reduces optimizer updates for the
same number of events. The benign-only validation prefixes cannot establish
attack generalization or AUROC, even when their validation losses improve.

### Bounded CPU/CUDA Timeline

The comparison runner's `--profile` option captures eight training batches after
16 waiting batches and four profiler warmup batches. It writes a Chrome/Perfetto
trace, a CPU operator table, and a device activity summary under `profile/`.
The device summary counts overlapping kernels and transfers once; CPU operator
totals may overlap and must not be added into a wall-time breakdown.

`scripts/slurm/temporal_timeline.sbatch` runs paired uninstrumented and instrumented
256-lane diagnostics over the same first 256 shuffled sections. Set `PROFILE_ROOT`
to a unique result directory and optionally `PROFILE_CODE_ROOT` to a code snapshot.
Use the uninstrumented scored throughput for speed comparisons: trace collection
and shape recording can materially slow small-operation workloads. This diagnostic
retains the existing model, objectives, optimizer, state handling, and Triton options.

### Fixed-Geometry Pipeline Diagnostics

`python -m graphids.exp.temporal_pipeline_ab OUTPUT_DIR` compares zero, two, and
four ordered loader workers at 256 lanes, then repeats the zero-worker baseline.
Each fresh-process trial owns the same first 1,024 sections. Worker trials use
two prefetched batches per worker and persistent workers. The new full-coverage
`temporal_set_01_segmented_lanes256_epoch.yml` uses two workers based on the measured
prefetch comparison; continuous execution remains the default training policy.

The runner's `--fusion` option compares the reference SSM pointwise calculation
with `model.scan.ssm_kernel: triton`, repeating both treatments. This optional
float32 CUDA kernel fuses sigmoid/tanh, gated state accumulation, and its first-order
analytic backward. Projections, dropout, normalization, chunk detachment, and
state ownership remain in PyTorch. CPU, strided inputs, other dtypes, and unavailable
Triton use the reference expression. It requires an SSM-lite backbone; the
context-free sequence path still uses its existing PyTorch implementation.
Higher-order gradients through the fused recurrence are unsupported.

`scripts/slurm/temporal_pipeline.sbatch` checks CUDA correctness before running
these diagnostics. Set `PIPELINE_ROOT`, optionally `PIPELINE_CODE_ROOT`, and
`PIPELINE_FUSION=1` for the fusion treatment. Both diagnostics verify queue and
initial-weight fingerprints, scored/replay ownership, and successor coverage.

### Opt-in CUDA Graph Training

`model.scan.cuda_graph: false` is the default. The opt-in path uses a module
adapter with [`torch.cuda.make_graphed_callables`](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.make_graphed_callables.html) to capture the lane scan, heads,
losses, and backward. Lightning still owns Adam, gradient accumulation, logging,
and checkpoints. Continuous training remains the default; the benchmark uses
experimental segmented execution at 256 lanes and 32 events per chunk.

Eligibility is checked on CPU before batch transfer. Only full, entirely scored
chunks with initialized state, successor metadata for every event, and no resets,
section endings, or recording terminals replay. Warmup, mixed and partial batches,
boundaries, CPU execution, validation, and other geometries use eager execution.
The IAT targets retain the existing definition. Capture uses owned inputs and
preserves RNG, buffers, gradients, and live recording state. Retained state and
metrics own their storage. Parameter gradient hooks clone replay gradients outside
capture so later replays preserve accumulated gradients; reset and checkpoint
loading remove these hooks. The adapter adds no checkpoint keys and is discarded
when loading a checkpoint or starting another fit. Setup failures disable capture;
replay errors propagate to interrupted coverage reporting without retrying a batch.

`python -m graphids.exp.temporal_cuda_graph OUTPUT_DIR` runs three sequential
pairs in fresh processes, alternating eager/graph order on one GPU. Each treatment
uses the same initial weights and first 2,048 shuffled sections, two ordered
workers, and two prefetched batches per worker. Experimental SSM fusion stays off.
`scripts/slurm/temporal_cuda_graph.sbatch` gates these comparisons on CUDA
correctness tests. Set `GRAPH_ROOT` to a unique absolute path under `$GRAPHIDS_RUN_ROOT/experimentruns/`
(for example, `/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/cuda_graph_YYYY_MM_DD`)
and optionally
`GRAPH_CODE_ROOT` to a code snapshot. The runner reports capture seconds, replay
and fallback counts, scored replay fraction, actual Adam steps, updates/sec,
end-to-end scored events/sec, telemetry, peak memory, coverage, and continuous
checkpoint validation. Profiling is a separate run and is excluded from selection.

The bounded gate requires at least 3% median end-to-end scored throughput gain
and positive gains in at least two pairs. Full eager and graph epochs then run in
separate Pitzer allocations using `scripts/slurm/temporal_cuda_graph_epoch.sbatch`
(one GPU, eight CPUs, 32 GiB, eight hours); set `GRAPH_EPOCH_CONFIG` and `GRAPH_ROOT`.
Enable the selected 256-lane configuration only after full coverage and at least
3% full-epoch improvement. Coverage must include 12,090,403 unique events, 57,108
attacks, and 12,090,385 eligible successor targets, with no ownership errors.
The existing continuous validation uses 8,192-event prefixes and 64 warmup events;
one-class prefixes report class counts and unavailable AUROCs.

The selected 256-lane epoch configuration enables CUDA Graphs after CUDA
correctness, complete epoch coverage, and a 23.8% full-epoch throughput gain
passed. The three-pair median gain was 28.9%. See the
[throughput note](../empirical-notes/2026-09-29-temporal-lane-throughput-check.md)
for timings, replay/fallback fractions, memory, coverage, and checkpoint validation.
The public default remains false.
