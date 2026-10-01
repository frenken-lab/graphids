# Temporal Lane Throughput Check

This note records the GPU benchmark submitted for the temporal stream-lane
throughput path and the final throughput comparison.

## Result

Both jobs completed successfully.

```text
baseline profile/train_events_per_sec = 237.065
lane     profile/train_events_per_sec = 935.144
speedup                               = 3.945x
required speedup                      = 1.500x
status                                = PASS
```

The lane-batched rich hybrid profile exceeded the acceptance threshold. With
the same-day baseline, the `1.5x` target was `355.597` events/sec; lane mode
reached `935.144` events/sec.

Additional profiler context:

```text
baseline profile/train_batch_ms       = 2159.749
lane     profile/train_batch_ms       = 547.509
baseline profile/train_events         = 512
lane     profile/train_events         = 512
baseline profile/train_data_wait_ms   = 1.465
lane     profile/train_data_wait_ms   = 14.030
```

## Submitted Jobs

Submitted on September 29, 2026 at about 13:27 America/New_York.

- Baseline event batching: SLURM job `55121622`
- Stream-lane batching: SLURM job `55121623`

Final scheduler state:

```text
55121622 COMPLETED 00:13:44 0:0 ray-temporal_joint_hybrid_ssm_lite_rich_set_01_profile
55121623 COMPLETED 00:04:56 0:0 ray-temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile
```

Recheck scheduler accounting:

```bash
sacct -j 55121622,55121623 --format=JobID,State,Elapsed,ExitCode,JobName%80 -P
```

## Run Directories

Baseline:

```text
/fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_profile
```

Lane-batched:

```text
/fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile
```

The baseline directory already contained an older September 28 profile result
before these jobs were submitted. The final numbers above are from the
September 29 rerun at commit `9f3ff93940e4de0f1bb29667fac13a5ba6f44718`.

## Metric Extraction

After both jobs finish, run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

runs = {
    "baseline": Path("/fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_profile"),
    "lanes": Path("/fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile"),
}

values = {}
for name, run_dir in runs.items():
    payload = json.loads((run_dir / ".graphids" / "mlflow_ingest.json").read_text())
    status = payload.get("status")
    metrics = payload.get("metrics") or {}
    eps = float(metrics["profile/train_events_per_sec"])
    batch_ms = float(metrics["profile/train_batch_ms"])
    events = float(metrics["profile/train_events"])
    print(f"{name}: status={status} events_per_sec={eps:.3f} batch_ms={batch_ms:.3f} events={events:.0f}")
    values[name] = eps

ratio = values["lanes"] / values["baseline"]
print(f"ratio={ratio:.3f}x")
print("PASS" if ratio >= 1.5 else "FAIL")
PY
```

Historical context from the older completed baseline profile, not used for the
final pass/fail decision:

```text
profile/train_events_per_sec = 196.409
1.5x threshold = 294.614 events/sec
```

## AMP Follow-up A/B

Submitted on September 29, 2026 at about 14:40 America/New_York from commit
`b2fda2a1c653b5f42dab90e0497965e435af3c47`.

Control:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile.yml
SLURM job   = 55124959
state       = COMPLETED
elapsed     = 00:06:06
exit code   = 0:0
```

Treatment:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_amp_profile.yml
SLURM job   = 55124957
state       = FAILED
elapsed     = 00:01:19
exit code   = 1:0
```

Latest same-commit control metrics:

```text
control profile/train_events_per_sec = 765.931
control profile/train_batch_ms       = 668.468
control profile/train_data_wait_ms   = 14.882
control profile/train_events         = 512
control train_loss                   = 1.644322
control val_loss                     = 1.195258
```

The AMP treatment did not produce profiler or loss metrics. It failed in the
stream-lane memory update under mixed precision:

```text
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Half for the source.
```

Status: **FAIL**. The treatment failed before throughput could be compared, so
the current FP32 stream-lane profile remains the speed baseline. The next A/B
should test lane geometry, `16 lanes x 32 chunk`, against the current
`8 lanes x 64 chunk` profile while holding 512 train events per step.

Run directories:

```text
control   /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile
treatment /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_amp_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_amp_profile
```

The control run directory contained older September 29 entries from the prior
lane-throughput check. The AMP A/B values above use the latest event block
beginning at `2026-09-29T18:41:55+00:00`, and both `.graphids/manifest.json`
files report commit `b2fda2a1c653b5f42dab90e0497965e435af3c47`.

## AMP Compatibility Fix Rerun

After the AMP failure, the temporal hybrid memory state handling was patched in
the working tree so TGN memory state follows the memory cell dtype and AMP
write-backs cast updated rows to the persistent state dtype. Local regression
coverage was added for stream-lane and validation-style temporal batches under
`torch.autocast`.

Validation:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
21 passed, 4 warnings

uv run pytest tests/exp/test_config_representation.py -q -k 'lane_profile_configs_parse or diagnostic_profile_uses_profiler'
2 passed, 12 deselected
```

Final patched A/B jobs were submitted from the dirty working tree at HEAD
`b2fda2a1c653b5f42dab90e0497965e435af3c47`.

Control:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile.yml
SLURM job   = 55125833
state       = COMPLETED
elapsed     = 00:05:26
exit code   = 0:0
```

Treatment:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_amp_profile.yml
SLURM job   = 55125834
state       = COMPLETED
elapsed     = 00:06:27
exit code   = 0:0
```

Final metrics:

```text
control profile/train_events_per_sec = 914.137
AMP     profile/train_events_per_sec = 708.655
speedup                               = 0.775x
required speedup                      = 1.150x

control profile/train_batch_ms       = 560.091
AMP     profile/train_batch_ms       = 722.496
control profile/train_data_wait_ms   = 13.704
AMP     profile/train_data_wait_ms   = 12.462
control profile/train_events         = 512
AMP     profile/train_events         = 512
control train_loss                   = 1.365504
AMP     train_loss                   = 1.474752
control val_loss                     = 1.268835
AMP     val_loss                     = 1.059126
```

Status: **FAIL** on throughput. The compatibility patch makes AMP runnable and
losses are finite, but mixed precision is slower than the FP32 stream-lane
baseline for this profile. Keep the FP32 stream-lane profile as the speed
baseline. The next speed A/B should remain lane geometry tuning,
`16 lanes x 32 chunk` versus `8 lanes x 64 chunk`, holding 512 train events per
step.

The final values above use the latest event blocks ending at
`2026-09-29T19:12:44+00:00` for control and `2026-09-29T19:13:44+00:00` for
AMP. The run directories are the same deterministic profile directories listed
above and contain older failed/intermediate events.

The AMP compatibility patch was not kept after this failed speed result. The
working tree was reverted back to the FP32 memory-state implementation before
the lane-geometry follow-up below.

## Lane Geometry Follow-up A/B

The next speed feature tested lane geometry rather than mixed precision. A new
diagnostic config was added:

```text
temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
```

It changes only the training stream-lane geometry from `8 x 64` to `16 x 32`,
holding 512 train events per step and keeping the same model, dataset,
objective, profiler, and trainer budget.

Validation:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
19 passed, 4 warnings

uv run pytest tests/exp/test_config_representation.py -q -k 'lane_profile_configs_parse or diagnostic_profile_uses_profiler'
2 passed, 12 deselected
```

Submitted on September 29, 2026 at about 15:35 America/New_York from HEAD
`b2fda2a1c653b5f42dab90e0497965e435af3c47`.

Control:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile.yml
geometry    = 8 lanes x 64 chunk
SLURM job   = 55139247
state       = COMPLETED
elapsed     = 00:05:20
exit code   = 0:0
```

Treatment:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
geometry    = 16 lanes x 32 chunk
SLURM job   = 55139248
state       = COMPLETED
elapsed     = 00:04:18
exit code   = 0:0
```

Final metrics:

```text
control   profile/train_events_per_sec = 935.875
treatment profile/train_events_per_sec = 1513.168
speedup                                  = 1.617x
required speedup                         = 1.150x

control   profile/train_batch_ms        = 547.082
treatment profile/train_batch_ms        = 338.363
control   profile/train_data_wait_ms    = 13.782
treatment profile/train_data_wait_ms    = 13.929
control   profile/train_events          = 512
treatment profile/train_events          = 512
control   train_loss                    = 1.420593
treatment train_loss                    = 1.288775
control   val_loss                      = 1.103423
treatment val_loss                      = 1.138716
```

Status: **PASS**. The `16 lanes x 32 chunk` FP32 geometry should become the
next speed baseline. The next follow-up should test `32 lanes x 16 chunk`
against this new baseline, again holding 512 train events per step.

Treatment run directory:

```text
/fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile
```

The final geometry A/B values above use the latest event blocks ending at
`2026-09-29T19:41:06+00:00` for control and `2026-09-29T19:40:04+00:00` for
treatment.

## Lane Geometry 32x16 Probe

The next geometry probe tested `32 lanes x 16 chunk` against the new `16 lanes x
32 chunk` FP32 baseline. A new diagnostic config was added:

```text
temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk16_profile.yml
```

Config validation:

```text
uv run pytest tests/exp/test_config_representation.py -q -k 'lane_profile_configs_parse or diagnostic_profile_uses_profiler'
2 passed, 12 deselected
```

Submitted on September 29, 2026 at about 16:01 America/New_York from HEAD
`b2fda2a1c653b5f42dab90e0497965e435af3c47`.

Control:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
geometry    = 16 lanes x 32 chunk
SLURM job   = 55159759
state       = COMPLETED
elapsed     = 00:04:20
exit code   = 0:0
```

Treatment:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk16_profile.yml
geometry    = 32 lanes x 16 chunk
SLURM job   = 55159760
state       = COMPLETED
elapsed     = 00:03:07
exit code   = 0:0
```

Final metrics:

```text
control   profile/train_events_per_sec = 1502.201
treatment profile/train_events_per_sec = 1605.295
raw ratio                                = 1.069x

control   profile/train_batch_ms        = 340.833
treatment profile/train_batch_ms        = 179.406
control   profile/train_data_wait_ms    = 14.662
treatment profile/train_data_wait_ms    = 14.299
control   profile/train_events          = 512
treatment profile/train_events          = 288
control   train_loss                    = 1.259983
treatment train_loss                    = 1.538363
control   val_loss                      = 1.064944
treatment val_loss                      = 1.224413
```

Status: **INVALID / FAIL** for equal-event speed comparison. The treatment job
completed with finite losses, but it did not preserve 512 train events per
profiled step. With `chunk_size=16`, only 18 lanes were active in the observed
profiled batch, producing `18 x 16 = 288` events. The apparent batch-time drop
is therefore not a valid speedup against the `16 x 32` baseline.

Keep `16 lanes x 32 chunk` as the speed baseline. The next speed feature should
not increase lane count past the number of active set_01 streams unless the
loader or dataset sharding is changed to guarantee 512 real train events per
step.

Treatment run directory:

```text
/fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk16_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk16_profile
```

The final 32x16 probe values above use the latest event blocks ending at
`2026-09-29T20:06:21+00:00` for control and `2026-09-29T20:05:08+00:00` for
treatment.

## Resource Telemetry and Chunk Scaling Sweep

The next diagnostic pass added offline resource telemetry to the active lane
profiles and reduced the active lane-profile memory request from `96G` to
`24G`. The telemetry callback writes final `resource/*` CPU/RSS/GPU aggregate
metrics into `.graphids/mlflow_ingest.json`.

Validation:

```text
uv run pytest tests/core/test_callbacks.py -q
7 passed

uv run pytest tests/exp/test_config_representation.py -q -k 'temporal_hybrid or stream_lanes or diagnostic_profile'
6 passed, 8 deselected

uv run ruff check graphids/core/callbacks.py tests/core/test_callbacks.py tests/exp/test_config_representation.py
All checks passed
```

The updated `16 lanes x 32 chunk` baseline was submitted as SLURM job
`55196954` and completed successfully:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
geometry    = 16 lanes x 32 chunk
state       = COMPLETED
elapsed     = 00:04:26
exit code   = 0:0
```

Final baseline metrics:

```text
profile/train_events             = 512
profile/train_active_lanes       = 16
profile/train_padding_fraction   = 0.0
profile/train_events_per_sec     = 1519.294
resource/gpu_util_percent_mean   = 17.222
resource/gpu_util_percent_max    = 21.0
resource/gpu_mem_gib_max         = 1.090
resource/rss_gib_max             = 4.152
```

Because GPU utilization and memory use were low, a chunk-size scaling sweep kept
`stream_lanes: 16` and increased per-lane chunk length:

```text
16 lanes x 64 chunk    SLURM job 55199474
16 lanes x 128 chunk   SLURM job 55199475
16 lanes x 256 chunk   SLURM job 55199476
```

All three jobs completed with top-level SLURM state `COMPLETED` and exit code
`0:0`.

Final sweep metrics:

```text
geometry    events  events/sec  gpu util mean/max  gpu mem max  rss max
16 x 32        512    1519.294     17.2% / 21.0%      1.090 GiB  4.152 GiB
16 x 64       1024    1557.144     19.7% / 23.0%      1.095 GiB  4.172 GiB
16 x 128      2048    1563.292     20.4% / 22.0%      1.109 GiB  4.186 GiB
16 x 256      4096    1572.428     20.5% / 22.0%      1.133 GiB  4.216 GiB
```

Status: **PASS** for stability under the `24G` request, but **FAIL** as a major
GPU-efficiency improvement. Larger chunks preserve `profile/train_active_lanes =
16` and `profile/train_padding_fraction = 0.0`, but they improve final
events/sec by only about 3.5% from `16 x 32` to `16 x 256`. GPU utilization
plateaus around 20%, and GPU memory remains near 1.1 GiB.

Conclusion: the diagnostic is not memory-capacity limited. The next throughput
work should focus on making each event more compute dense or reducing
small-kernel/framework overhead: larger model/backbone A/Bs, `torch.compile`
A/Bs, and fusion in the temporal scan/backbone path. Running multiple same-GPU
copies may improve node occupancy for independent trials, but it is not the
first choice for faster single-model learning because gradient averaging and
failure isolation add orchestration complexity.

## Throughput Lever Follow-up

Two follow-up diagnostics tested the first low-risk throughput levers on the
stable `16 lanes x 32 chunk` geometry.

The compile treatment used native `LightningModule.compile()` on the temporal
hybrid module:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_compile_profile.yml
SLURM job   = 55205789
state       = CANCELLED by 35950
elapsed     = 00:10:09
```

The run used:

```text
backend=inductor mode=reduce-overhead fullgraph=false dynamic=false
```

but the job did not reach the first `temporal_batch_profile` line or write final
metrics after about 10 minutes. It was cancelled to avoid burning the full GPU
allocation. Status: **FAIL** as a practical speed treatment. Broad inductor
compilation of the dynamic scan/state-update path is too expensive or stalls
before useful work for this diagnostic shape.

The compute-density treatment increased model width/depth while holding the
same 512-event lane geometry:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_large_profile.yml
SLURM job   = 55205790
state       = COMPLETED
elapsed     = 00:04:32
exit code   = 0:0
```

Model treatment:

```text
scale                         = large
input.hidden                  = 128
input.embedding_dim           = 32
memory.time_encoding_dim      = 8
backbone.layers               = 2
motif.embedding_dim/time_dim  = 16
```

Final metrics:

```text
profile/train_events             = 512
profile/train_active_lanes       = 16
profile/train_padding_fraction   = 0.0
profile/train_events_per_sec     = 1405.797
resource/gpu_util_percent_mean   = 18.436
resource/gpu_util_percent_max    = 21.0
resource/gpu_mem_gib_max         = 1.121
resource/rss_gib_max             = 4.163
```

Status: **FAIL** for throughput. The larger/deeper model is stable under the
`24G` request but is slower than the `16 x 32` small-model baseline and does not
raise GPU utilization. This argues against simply widening the diagnostic model
as the next efficiency fix.

Next action: inspect and optimize the lane scan/state-update implementation.
The hot path still performs per-step clones and indexed updates across temporal
state. Fusion or vectorization there is a better target than more batch size,
more model width, or broad `torch.compile`.

## Lane Scan Optimization Follow-up

The next implementation pass targeted the lane scan/state-update overhead
directly:

- added batched rhythm row updates;
- added a context-free lane sequence path that precomputes SSM-lite projections
  across `[lanes, chunk, hidden]`;
- changed rich lane state updates from per-step full-state clones to one cloned
  state per chunk with in-place indexed updates;
- moved stream-lane batch prefetching onto PyTorch `DataLoader` workers,
  controlled by `num_workers` and `prebuild_batches`; the active lane
  diagnostic configs now use `num_workers: 4` and `prebuild_batches: 8`.

Validation:

```text
uv run pytest tests/core/data/test_temporal_datamodule.py -q
6 passed

uv run pytest tests/core/models/test_temporal_hybrid.py -q
20 passed

uv run pytest tests/exp/test_config_representation.py -q
16 passed

uv run ruff check graphids/core/models/temporal/_hybrid/scan.py graphids/core/models/temporal/_hybrid/backbone.py graphids/core/models/temporal/_hybrid/contexts.py graphids/core/data/datamodule/stream_lanes.py graphids/core/data/datamodule/temporal.py graphids/primitives_data.py tests/core/data/test_temporal_datamodule.py tests/core/models/test_temporal_hybrid.py tests/exp/test_config_representation.py
All checks passed
```

An intermediate sparse-overlay implementation was submitted first:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
SLURM job   = 55231925
state       = COMPLETED
elapsed     = 00:04:59
exit code   = 0:0
```

Final overlay metrics:

```text
profile/train_events             = 512
profile/train_active_lanes       = 16
profile/train_padding_fraction   = 0.0
profile/train_batch_ms           = 451.738
profile/train_data_wait_ms       = 1.529
profile/train_events_per_sec     = 1133.401
resource/gpu_util_percent_mean   = 16.452
resource/gpu_util_percent_max    = 19.0
resource/gpu_mem_gib_max         = 1.094
resource/rss_gib_max             = 5.896
```

Status: **FAIL**. Although prebuilding reduced data wait, the sparse overlay
used CPU dictionary/list bookkeeping inside the CUDA step loop and regressed
throughput below the prior `1519.294 events/sec` baseline.

The corrected implementation removes that CPU overlay and uses one clone per
chunk plus in-place indexed updates. It was submitted as:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_profile.yml
SLURM job   = 55232073
state       = COMPLETED
elapsed     = 00:03:45
exit code   = 0:0
```

Final corrected metrics:

```text
profile/train_events             = 512
profile/train_active_lanes       = 16
profile/train_padding_fraction   = 0.0
profile/train_batch_ms           = 228.493
profile/train_data_wait_ms       = 1.205
profile/train_events_per_sec     = 2240.772
resource/gpu_util_percent_mean   = 17.207
resource/gpu_util_percent_max    = 20.0
resource/gpu_mem_gib_max         = 1.094
resource/rss_gib_max             = 5.900
```

Status: **PASS** for equal-work throughput. The corrected scan path improves
throughput by about `1.475x` over the previous `1519.294 events/sec` `16 x 32`
baseline and reduces batch time from `336.999 ms` to `228.493 ms`.

Status remains **FAIL** for materially increasing GPU utilization or VRAM use:
GPU utilization stayed near 17-20%, and GPU memory stayed near 1.1 GiB. RSS
increased from roughly `4.152 GiB` to `5.900 GiB`, consistent with prebuilt
lane batches and one cloned state per chunk.

Conclusion: eliminating avoidable clone/data-wait overhead is valuable, but the
workload still does not saturate the GPU. The next efficiency target should be
larger fused computation inside the recurrent scan/backbone path, not more CPU
prebuilding or chunk-size scaling.

## Scan Backend Vectorization Follow-up

The next implementation pass added opt-in `model.scan` backends for the rich
lane path:

- `lane_fastpath`: all-valid lane fast path, flattened lane-ID row addressing,
  and scan subphase timing;
- `fused_memory`: `lane_fastpath` plus batched TGN source/destination GRU memory
  updates;
- `compiled_inner`: `fused_memory` plus a narrow CUDA `torch.compile` A/B for
  the batched memory update, with CPU fallback.

Validation before submission:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
25 passed

uv run pytest tests/exp/test_config_representation.py -q
19 passed

uv run ruff check graphids/primitives_models.py graphids/core/models/temporal/_hybrid/model.py graphids/core/models/temporal/_hybrid/runtime.py graphids/core/models/temporal/_hybrid/contexts.py graphids/core/models/temporal/_hybrid/scan.py tests/core/models/test_temporal_hybrid.py tests/exp/test_config_representation.py
All checks passed
```

Submitted diagnostics all completed at the top level with exit code `0:0`:

| Config | SLURM job | State | Elapsed |
| --- | ---: | --- | ---: |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_lane_fastpath_profile.yml` | `55237253` | `COMPLETED` | `00:03:56` |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_memory_profile.yml` | `55237254` | `COMPLETED` | `00:03:53` |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_compiled_inner_profile.yml` | `55237252` | `COMPLETED` | `00:04:01` |

Final metrics:

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

Status: **do not promote by default yet**. `compiled_inner` improved final
events/sec over the `2233.770 events/sec` baseline, but measured scan time rose
from the phase baseline (`111.984 ms`) to `120.619 ms`. `lane_fastpath` and
`fused_memory` were slower than the baseline. The useful evidence from this pass
is the scan subphase breakdown: memory/rhythm context and updates dominate the
remaining recurrent scan time.

## Context and Update Micro-optimization Follow-up

The next implementation pass targeted the largest scan subphases without
changing public model behavior:

- source/destination elapsed-time memory context now uses one batched time
  encoder call and one preallocated context tensor in the common rich path;
- batched TGN memory updates now reuse the GRU input projection for source and
  destination rows;
- batched rhythm context/update helpers now avoid small stack/cat allocations.

Validation before submission:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
25 passed

uv run ruff check graphids/core/models/temporal/_hybrid/contexts.py graphids/core/models/temporal/_hybrid/scan.py tests/core/models/test_temporal_hybrid.py
All checks passed
```

Submitted diagnostics all completed at the top level with exit code `0:0`:

| Config | SLURM job | State | Elapsed |
| --- | ---: | --- | ---: |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_lane_fastpath_profile.yml` | `55238250` | `COMPLETED` | `00:03:49` |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_memory_profile.yml` | `55238251` | `COMPLETED` | `00:03:30` |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_compiled_inner_profile.yml` | `55238252` | `COMPLETED` | `00:04:23` |

Final metrics:

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
reduced memory-update time, but the RSS max was an outlier. The next benchmark
should repeat `lane_fastpath` without scan subphase timing to separate runtime
benefit from profiling overhead.

## Lane-Scaling Workers0 Follow-up

The next diagnostic matrix tested whether widening the lane dimension raises GPU
utilization by increasing per-timestep work. The configs used the clean
`lane_fastpath` backend, no phase/subphase timing, and the workers0 data path:
`num_workers: 0`, `prebuild_batches: 0`, `pin_memory: false`.

Submitted diagnostics:

| Config | SLURM job | Geometry | Expected events | State | Elapsed |
| --- | ---: | --- | ---: | --- | ---: |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk16_lane_fastpath_workers0_profile.yml` | `55240966` | `32 x 16` | `512` | `COMPLETED` | `00:01:21` |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes64_chunk8_lane_fastpath_workers0_profile.yml` | `55240967` | `64 x 8` | `512` | `COMPLETED` | `00:01:25` |
| `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes32_chunk32_lane_fastpath_workers0_profile.yml` | `55240968` | `32 x 32` | `1024` | `COMPLETED` | `00:01:21` |

All three jobs exited cleanly at the SLURM level, but produced no training
batches:

```text
Total length of `TemporalStreamLaneLoader` across ranks is zero.
`Trainer.fit` stopped: No training batches.
```

Because no train batches ran, no `profile/train_events`,
`profile/train_events_per_sec`, `profile/train_batch_ms`,
`profile/train_active_lanes`, `profile/train_padding_fraction`, or final loss
metrics were logged. The `64 x 8` run produced one idle system-metrics sample
before exit (`resource/gpu_util_percent_mean = 0.0`,
`resource/gpu_mem_gib_max = 1.093`, `resource/rss_gib_max = 28.033`); the other
two exited before a resource sample was emitted.

Status: **invalid geometry under strict full-batch diagnostics**. The widened
lane configs do not satisfy the acceptance criteria because
`profile/train_events` is absent rather than `512`/`1024`, and
`profile/train_padding_fraction` is absent rather than `0.0`. This confirms the
earlier non-strict `32 x 16` result: the current `set_01` stream distribution
cannot support full widened-lane batches without changing the loader or dataset
sharding. Do not add worker variants for this matrix.

## Strict Rich Feature Ablation Follow-up

The next strict matrix compared rich-state feature variants on the valid
`16 lanes x 32 chunk` geometry. All eight runs completed successfully with
`max_epochs=3`, `train_events=512`, and `train_padding_fraction=0.0`. There
were no fatal errors or tracebacks; a few stderr logs contained only Ray
metrics-exporter warnings. A later `scancel` found no queued jobs because the
runs had already finished.

Using epoch 2 profiler samples as the stable comparison:

| Variant | SLURM job | Epoch 2 median eps | Final eps | Train loss | Val loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rich` | `55242488` | `2602.36` | `2568.81` | `1.746` | `1.459` |
| `minus_motif` | `55242495` | `3128.97` | `3129.65` | `1.264` | `1.114` |
| `minus_rhythm` | `55242500` | `3311.96` | `3315.82` | `1.325` | `1.216` |
| `minus_memory` | `55242508` | `4230.88` | `4230.16` | `1.554` | `1.239` |
| `memory_only` | `55242513` | `4023.18` | `4021.14` | `1.310` | `1.123` |
| `backbone_only` | `55242520` | `20275.07` | `20169.79` | `1.495` | `1.188` |
| `backbone_none` | `55242564` | `45038.00` | `45907.58` | `2.248` | `2.029` |

Status: **PASS** for rich strict throughput. Against the previous
`2267.660 events/sec` rich strict baseline, final throughput improved by about
`13.3%`, and the epoch 2 median improved by about `14.8%`.

The ablations show that the rich feature stack is the remaining cost center.
Removing memory has the largest single-module speed effect, but
`backbone_only` and `backbone_none` are quality/composition anchors rather than
direct substitutes for the rich model.

The strict phase probe completed as job `55242574`. Final throughput was lower
(`2404.17 events/sec`) because subphase timing was enabled. Final phase timings:

| Phase | Time |
| --- | ---: |
| forward | `110.10 ms` |
| scan | `108.25 ms` |
| memory context | `18.21 ms` |
| memory update | `10.39 ms` |
| rhythm context | `19.01 ms` |
| rhythm update | `14.62 ms` |
| motif context | `12.66 ms` |
| motif update | `6.05 ms` |
| backbone | `12.78 ms` |

Conclusion: the speed target was met, but the forward pass remains almost
entirely scan-bound. The clearest next targets are reducing per-step rich
context/update kernel count and state-write overhead, not lane widening or
additional data prefetching.

## Working-Tree Scan Hot-Path Cleanup

A follow-up implementation pass applied the first low-risk scan cleanup in the
working tree:

- precompute flattened lane-ID source/destination row indices once per chunk
  for lane-fastpath backends;
- compute the all-valid full-chunk fast-path condition once per chunk instead
  of checking `active.all()` on every timestep;
- skip scan subphase timing helper calls entirely when subphase profiling is
  disabled;
- shift motif lane state in place column-by-column instead of cloning the active
  motif rows each timestep.

Local verification:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
31 passed, 4 warnings

uv run ruff check graphids/core/models/temporal/_hybrid/scan.py tests/core/models/test_temporal_hybrid.py
All checks passed
```

Strict-rich benchmark submission:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_strict_rich_profile.yml
SLURM job   = 55246546
state       = COMPLETED
exit code   = 0:0
elapsed     = 00:03:34
submit      = 2026-09-30T16:16:05
start/end   = 2026-09-30T16:50:58 / 2026-09-30T16:54:32
git sha     = 4eadb6baf0ce1e4c454ba96d704955707c03e720
run dir     = /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_strict_rich_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_strict_rich_profile
```

The run emitted a Ray snapshot warning because the same output directory already
contained checkpoints. The project code does not call `ray.train.get_checkpoint`,
so the profiler samples are treated as a fresh throughput measurement. Keep the
warning attached to the run because checkpoint/loss interpretation is otherwise
easy to confuse.

Final logged metrics:

| Metric | Value |
| --- | ---: |
| `profile/train_events` | `512.0` |
| `profile/train_active_lanes` | `16.0` |
| `profile/train_padding_fraction` | `0.0` |
| `profile/train_events_per_sec` | `2755.051` |
| `profile/train_batch_ms` | `185.840` |
| `profile/train_data_wait_ms` | `27.729` |
| `resource/gpu_util_percent_mean/max` | `15.7% / 18.0%` |
| `resource/gpu_mem_gib_max` | `1.142` |
| `resource/rss_gib_max` | `25.928` |
| `train_loss` | `1.147` |
| `val_loss` | `1.030` |

Stable profiler sample medians:

| Epoch | Median events/sec | Median batch ms | Median data wait ms |
| ---: | ---: | ---: | ---: |
| `0` | `2770.514` | `184.835` | `17.699` |
| `1` | `2769.144` | `184.895` | `14.416` |
| `2` | `2772.259` | `184.687` | `14.605` |

Status: **PASS and promote as current strict-rich baseline**. The cleanup
preserved the equal-work invariant (`512` events, no padding) and improved final
throughput from `2568.81` to `2755.05 events/sec` (`+7.2%`). The epoch 2 median
improved from `2602.36` to `2772.26 events/sec` (`+6.5%`). Against the earlier
`2267.660 events/sec` strict-rich baseline, the final gain is about `21.5%`.

## Next Throughput Implementation Analysis

The latest result validates the low-risk lane scan cleanup, but it did not
change the basic bottleneck: GPU utilization is still only about `15.7%` mean /
`18.0%` max, while data wait is low after warmup. The remaining throughput work
should continue to reduce the 32-step rich scan's small gather/projection/update
kernel count rather than increasing loader workers, prefetch depth, or lane
width.

Recommended implementation order:

1. Fuse the rich context builder for memory, rhythm, and motif into one
   per-step helper that gathers source/destination rows once, accumulates the
   context contribution in a preallocated tensor, and returns the rows needed by
   the update path. This is the highest-probability next win because the phase
   probe measured context reads as the largest combined rich cost.
2. Fuse post-backbone updates into one writer that updates memory, last-seen,
   rhythm, motif, and backbone state from the same `_LaneStepGather`/`z_t`
   bundle. The current split writer still launches separate indexed writes for
   each state family.
3. Add a clean phase-profile rerun after those two changes. If `scan` remains
   dominant and rhythm still costs roughly `33 ms` combined context/update,
   prototype a narrow custom indexed rhythm kernel. Do this after Python/PyTorch
   fusion, not before.
4. Fix the Ray output naming before more repeated A/Bs. The next submission
   should use a unique run/output suffix or clear the prior run directory so
   checkpoint snapshots cannot pollute the audit trail.

Do not spend the next pass on lane widening for `set_01`; strict widened-lane
jobs produced zero full training batches or invalid partial batches. Do not make
`compiled_inner` the main path yet; its previous win was tiny and noisier than
the plain lane-fastpath cleanup.

## Fused Context/Write Implementation Pass

The next implementation pass is now in the working tree:

- added a batched `TemporalMotifContext.read_context_batch` helper so lane scan
  no longer reconstructs motif context inline;
- changed lane scan context accumulation to clone the encoded step once and add
  memory/rhythm/motif contexts in place;
- replaced flat lane-state advanced indexing with explicit `index_select` and
  `index_copy_` calls for memory, last-seen, and rhythm state;
- added a unique strict benchmark config,
  `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_profile.yml`,
  to avoid reusing the prior Ray output/checkpoint directory.

Local verification:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
31 passed, 4 warnings

uv run pytest tests/core/models/test_temporal_hybrid.py -q -k 'lane_scan or batch_update or sparse_row_overlay'
11 passed, 20 deselected

uv run ruff check graphids/core/models/temporal/_hybrid/scan.py graphids/core/models/temporal/_hybrid/contexts.py tests/core/models/test_temporal_hybrid.py
All checks passed
```

Fused-context benchmark:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_profile.yml
SLURM job   = 55279425
state       = COMPLETED
exit code   = 0:0
elapsed     = 00:05:08
submit      = 2026-10-01T11:32:12
start/end   = 2026-10-01T11:32:33 / 2026-10-01T11:37:41
git sha     = 4eadb6baf0ce1e4c454ba96d704955707c03e720
run dir     = /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_profile
```

Final logged metrics:

| Metric | Value |
| --- | ---: |
| `profile/train_events` | `512.0` |
| `profile/train_active_lanes` | `16.0` |
| `profile/train_padding_fraction` | `0.0` |
| `profile/train_events_per_sec` | `2825.696` |
| `profile/train_batch_ms` | `181.194` |
| `profile/train_data_wait_ms` | `14.272` |
| `resource/gpu_util_percent_mean/max` | `15.8% / 19.0%` |
| `resource/gpu_mem_gib_max` | `1.142` |
| `resource/rss_gib_max` | `37.806` |
| `train_loss` | `1.330` |
| `val_loss` | `1.167` |

Stable profiler sample medians:

| Epoch | Median events/sec | Median batch ms | Median data wait ms |
| ---: | ---: | ---: | ---: |
| `0` | `2836.492` | `180.505` | `14.619` |
| `1` | `2834.533` | `180.630` | `14.195` |
| `2` | `2829.216` | `180.969` | `14.400` |

Status: **PASS and promote as current strict-rich baseline**. The fused-context
implementation preserved the strict equal-work invariant and improved final
throughput from `2755.05` to `2825.70 events/sec` (`+2.6%`). The epoch 2 median
improved from `2772.26` to `2829.22 events/sec` (`+2.1%`). Against the
pre-cleanup `2568.81 events/sec` rich baseline, the final gain is about `10.0%`.

Next target: the scan is still underutilizing the GPU (`15.8%` mean /
`19.0%` max). The next implementation pass should use a phase/subphase profile
on the fused-context code, then decide whether to fuse the post-backbone writer
more aggressively or prototype the narrow rhythm update/context kernel.

## Post-55279425 Writer-Fusion Pass

The next throughput pass is implemented in the working tree:

- added a unique fused-context phase profile,
  `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_phase_profile.yml`,
  with `phase_timing: true`, scan subphase timing, `lane_fastpath`, and the
  strict `16 lanes x 32 chunk` equal-work data settings;
- added a separate no-subphase strict A/B config,
  `temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_writer_fusion_profile.yml`,
  so the writer-fusion benchmark will not reuse the `55279425` Ray output path;
- moved feature, backbone-state, memory, last-seen, rhythm, and motif writes
  behind one post-`backbone.step` `_write_lane_step_state` helper while keeping
  the historical subphase metric names (`backbone_write`, `memory_update`,
  `rhythm_update`, `motif_update`);
- replaced motif rolling updates in the lane fastpath with row slice assignment
  while preserving the causal shift-left then append-current-destination
  semantics for partial active-lane batches.

Local verification:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
32 passed, 4 warnings

uv run pytest tests/core/models/test_temporal_hybrid.py -q -k 'lane_scan or batch_update or sparse_row_overlay'
12 passed, 20 deselected

uv run pytest tests/exp/test_config_representation.py -q -k 'fused_context or phase_profile or strict_ablation'
5 passed, 20 deselected

uv run ruff check graphids/core/models/temporal/_hybrid/scan.py graphids/core/models/temporal/_hybrid/contexts.py tests/core/models/test_temporal_hybrid.py tests/exp/test_config_representation.py
All checks passed!

uv run python scripts/lint_docs.py
docs lint OK (6 directive targets, 61 markdown files)
```

Fused-context phase-profile submission:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_phase_profile.yml
SLURM job   = 55279729
state       = COMPLETED
exit code   = 0:0
elapsed     = 00:03:43
submit      = 2026-10-01T11:48:26
start/end   = 2026-10-01T11:51:11 / 2026-10-01T11:54:54
git sha     = 4eadb6baf0ce1e4c454ba96d704955707c03e720
run dir     = /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_phase_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_fused_context_phase_profile
```

Final phase-profile metrics:

| Metric | Value |
| --- | ---: |
| `profile/train_events` | `512.0` |
| `profile/train_active_lanes` | `16.0` |
| `profile/train_padding_fraction` | `0.0` |
| `profile/train_events_per_sec` | `2524.535` |
| `profile/train_batch_ms` | `202.810` |
| `profile/train_data_wait_ms` | `14.999` |
| `profile/train_phase_scan_ms` | `102.653` |
| `profile/train_phase_scan_memory_context_ms` | `18.240` |
| `profile/train_phase_scan_rhythm_context_ms` | `18.782` |
| `profile/train_phase_scan_motif_context_ms` | `13.073` |
| `profile/train_phase_scan_backbone_ms` | `12.628` |
| `profile/train_phase_scan_backbone_write_ms` | `0.411` |
| `profile/train_phase_scan_memory_update_ms` | `9.985` |
| `profile/train_phase_scan_rhythm_update_ms` | `14.097` |
| `profile/train_phase_scan_motif_update_ms` | `4.540` |
| `resource/gpu_util_percent_mean/max` | `15.5% / 18.0%` |
| `resource/gpu_mem_gib_max` | `1.142` |
| `resource/rss_gib_max` | `28.514` |
| `train_loss` | `1.396` |
| `val_loss` | `1.184` |

The phase profile confirms that scan still dominates and that rhythm
context/update plus memory context remain the largest rich-context costs.

Writer-fusion strict benchmark:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_writer_fusion_profile.yml
SLURM job   = 55280466
state       = COMPLETED
exit code   = 0:0
elapsed     = 00:03:39
submit      = 2026-10-01T12:04:06
start/end   = 2026-10-01T12:04:17 / 2026-10-01T12:07:56
git sha     = 4eadb6baf0ce1e4c454ba96d704955707c03e720
run dir     = /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_writer_fusion_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_writer_fusion_profile
```

Final logged metrics:

| Metric | Value |
| --- | ---: |
| `profile/train_events` | `512.0` |
| `profile/train_active_lanes` | `16.0` |
| `profile/train_padding_fraction` | `0.0` |
| `profile/train_events_per_sec` | `2873.714` |
| `profile/train_batch_ms` | `178.167` |
| `profile/train_data_wait_ms` | `14.899` |
| `resource/gpu_util_percent_mean/max` | `16.2% / 19.0%` |
| `resource/gpu_mem_gib_max` | `1.142` |
| `resource/rss_gib_max` | `31.959` |
| `train_loss` | `1.125` |
| `val_loss` | `1.016` |

Stable profiler sample medians:

| Epoch | Median events/sec | Median batch ms | Median data wait ms |
| ---: | ---: | ---: | ---: |
| `0` | `2872.446` | `178.245` | `15.284` |
| `1` | `2884.122` | `177.524` | `14.984` |
| `2` | `2887.055` | `177.344` | `14.939` |

Status: **PASS and promote as current strict-rich baseline**. The writer-fusion
implementation preserved the strict equal-work invariant and improved final
throughput from `2825.70` to `2873.71 events/sec` (`+1.7%`). The epoch 2 median
improved from `2829.22` to `2887.05 events/sec` (`+2.0%`). Against the
pre-cleanup `2568.81 events/sec` rich baseline, the final gain is about `11.9%`.

## Triton Rhythm Kernel — October 1, 2026

Implemented an opt-in float32 Triton kernel for the lane scan rhythm path.
One launch gathers source/destination rhythm rows, emits their pre-update
features, applies the Welford updates, and writes state back. Self-edges apply
two updates to the same row. The trainable `rhythm.proj` remains in PyTorch.
The lane writer skips rhythm updates already performed by the kernel.

Enable with `model.scan.rhythm_kernel: triton`. The default remains `torch`;
CPU, unsupported inputs, and synchronous import/compilation/launch failures
use the PyTorch path. Only the lane scan uses this kernel. CUDA execution
errors reported asynchronously are not covered by the synchronous fallback.

Strict benchmark:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_rhythm_profile.yml
SLURM job   = 55282240
cluster     = pitzer
state       = COMPLETED
exit code   = 0:0
elapsed     = 00:03:19
start/end   = 2026-10-01T13:35:28 / 2026-10-01T13:38:47 (America/New_York)
base commit = 4560d95e55f48297b8aa3732d0bc27d9aecb207e (plus uncommitted kernel changes)
stack       = torch 2.8.0+cu128 / triton 3.4.0
run dir     = /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_rhythm_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_rhythm_profile
stderr      = /fs/ess/PAS1266/graphids/slurm_logs/ray-temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_rhythm_profile_55282240.err
```

The job ran GPU correctness checks before starting the experiment. Earlier
attempts `55282167` and `55282185` stopped on a test assertion targeting a
`Sequential` projection as a `Linear`, and a missing primitive scan-schema
field, respectively. Both were corrected before the successful benchmark.

Final reported metrics:

| Metric | Value |
| --- | ---: |
| `profile/train_events` | `512.0` |
| `profile/train_active_lanes` | `16.0` |
| `profile/train_lane_capacity_events` | `512.0` |
| `profile/train_padding_events` | `0.0` |
| `profile/train_padding_fraction` | `0.0` |
| `profile/train_events_per_sec` | `3218.282` |
| `profile/train_batch_ms` | `159.091` |
| `profile/train_data_wait_ms` | `14.385` |
| `resource/gpu_util_percent_mean/max` | `15.72% / 18.0%` |
| `resource/gpu_mem_gib_max` | `1.142` |
| `resource/rss_gib_max` | `24.438` |
| `train_loss` | `1.274` |
| `val_loss` | `1.107` |

Stable profiler sample medians:

| Epoch | Samples | Median events/sec | Median batch ms | Median data wait ms |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `9` | `3229.152` | `158.556` | `13.809` |
| `1` | `10` | `3242.437` | `157.906` | `14.319` |
| `2` | `10` | `3218.478` | `159.082` | `14.385` |

All 29 logged samples have `512` events, `16` active lanes, and `0` padding.
Their mean throughput is `3219.727 events/sec`. No rhythm kernel fallback was
reported. The final throughput improves over writer-fusion job `55280466`
(`2873.714 events/sec`) by **12.0%**; the epoch 2 median improves from
`2887.055` to `3218.478 events/sec` by **11.5%**.

Verification:

- Local model suite: `35 passed, 2 skipped` (CUDA unavailable locally).
- GPU checks: `4 passed`; covers wrapper features/state equivalence, repeated
  updates, self-edges, strided lane vectors, negative IAT, empty inputs, lane
  reference semantics, and finite projection gradients.
- Config/primitive validation: `4 passed, 23 deselected`; accepts the new
  diagnostic config and rejects unknown kernel values.
- Ruff, documentation lint, and `git diff --check`: passed.

Status: **PASS as the current opt-in strict-rich benchmark**. Keep the default
PyTorch path; this is one successful Triton benchmark, not a repeated paired
A/B experiment. Loss values are included for context, not as evidence of
training-quality equivalence.

### Next Narrow Kernel Candidates

The first three candidates were subsequently implemented and benchmarked in
the Triton Time Features and Motif Writer section below.

These are implementation candidates inferred from the current code, not
measured speedups. The earlier phase-profile job `55279729` predates the rhythm
kernel; rerun a phase profile with Triton rhythm enabled before treating its
subphase costs as the current bottleneck. Use an uninstrumented strict A/B run
for acceptance because subphase synchronization adds overhead.

| Priority | Scope | PyTorch boundary and validation |
| ---: | --- | --- |
| `1` | Memory time features: gather last-seen timestamps, mask unseen IDs, clamp elapsed time, apply `log1p`, encode sine/cosine, and pack source/destination time features in one launch. | Keep memory-row gathers, `memory.mix`, and GRU updates in PyTorch. Time-encoder frequencies are fixed buffers. Test unseen IDs, self-edges, negative elapsed times, odd encoding dimensions, partial lanes, and dtype/layout equivalence. |
| `2` | Motif time features: fuse `log1p`, frequency division, sine/cosine, and truncation for the active lanes' IAT history. | Keep embedding lookup and `motif.proj` in PyTorch. Reuse the time-encoding kernel where practical. Preserve pre-update history and feature order. The history length is only `3`, so launch savings need measurement. |
| `3` | Extend the existing rhythm launch to shift/append motif IDs and IATs, and optionally write last-seen timestamps. | Read motif context before mutating its history; preserve independent ID snapshots used by embedding backward. Preserve the current lane writer's IAT semantics and partial-lane masking. Extending an existing launch is preferable to assuming a separate tiny writer kernel will pay off. |
| `4` | SSM-lite pointwise recurrence: fuse sigmoid/tanh and `next_state = decay * state + gate * candidate` after the three linear projections. | Keep projections, output projection, dropout, and normalization in PyTorch. This candidate requires an explicit backward and output/parameter/state-gradient equivalence tests, so it is a larger step than the non-trainable time-feature kernels. |

Memory rows depend on previous GRU outputs and carry gradients. A raw Triton
memory gather/scatter cannot replace their PyTorch operations without backward
support. Trainable custom kernels need an explicit autograd registration, as
shown in the [PyTorch Triton integration tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).
A full GRU kernel or full scan fusion is outside the next narrow pass.

Recommended next implementation: **memory time-feature fusion**, followed by
motif time-feature fusion if the refreshed profile supports it. Keep each
change opt-in, preserve the `512` events / `0` padding / `16 x 32` invariant,
and compare against the Triton rhythm benchmark above.

## Triton Time Features and Motif Writer — October 1, 2026

Implemented the first three narrow candidates above, with comments/docstrings
explaining fixed time features, feature order, and pre-event history ownership:

- Memory time features gather source/destination last-seen timestamps, mask
  unseen IDs, clamp negative elapsed time, and fuse `log1p`, frequency division,
  sine/cosine, and feature packing. Memory rows, GRU updates, `memory.mix`, and
  timestamp writes remain in PyTorch.
- Motif time features fuse the same fixed sinusoidal encoding for the IAT
  history. ID embedding lookup and `motif.proj` remain in PyTorch.
- The rhythm kernel optionally shifts/appends motif IDs and IATs in its existing
  launch. The scan takes independent history snapshots first, so context sees
  the old history and embedding backward retains the correct indices. The
  writer skips the motif update when the kernel already performed it.

The time kernels preserve sine-then-cosine feature order, including odd encoding
dimensions. Their `log1p` calculation includes a rounding correction for tiny
positive IATs. All wrappers support empty active-lane inputs and preserve the
PyTorch fallback for unsupported inputs or synchronous kernel failures.
Gradient-bearing time inputs keep the PyTorch encoder; memory and embedding
gradients never pass through these non-autograd kernels.

Enable all three candidates alongside rhythm fusion:

```yaml
model:
  scan:
    rhythm_kernel: triton
    time_kernel: triton
    motif_writer: triton
```

`time_kernel` controls both memory and motif time features. `motif_writer`
requires `rhythm_kernel: triton` because it extends that launch. All three
options still default to `torch`; CPU and non-lane scans retain PyTorch behavior.
Last-seen timestamp writes were optional in the candidate and remain unchanged.

Strict benchmark:

```text
config      = temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_context_profile.yml
SLURM job   = 55282541
cluster/node = pitzer / p0225
state       = COMPLETED
exit code   = 0:0
elapsed     = 00:03:58
start/end   = 2026-10-01T13:50:06 / 2026-10-01T13:54:04 (America/New_York)
base commit = 4560d95e55f48297b8aa3732d0bc27d9aecb207e (plus uncommitted kernel changes)
run dir     = /fs/ess/PAS1266/graphids/dev/rf15/set_01/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_context_profile/temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_context_profile
stderr      = /fs/ess/PAS1266/graphids/slurm_logs/ray-temporal_joint_hybrid_ssm_lite_rich_set_01_lanes16_chunk32_triton_context_profile_55282541.err
```

The allocation ran the expanded GPU correctness suite before training.

| Final reported metric | Value |
| --- | ---: |
| `profile/train_events` | `512.0` |
| `profile/train_active_lanes` | `16.0` |
| `profile/train_lane_capacity_events` | `512.0` |
| `profile/train_padding_events` | `0.0` |
| `profile/train_padding_fraction` | `0.0` |
| `profile/train_events_per_sec` | `3317.020` |
| `profile/train_batch_ms` | `154.355` |
| `profile/train_data_wait_ms` | `15.049` |
| `resource/gpu_util_percent_mean/max` | `14.76% / 17.0%` |
| `resource/gpu_mem_gib_max` | `1.142` |
| `resource/rss_gib_max` | `27.242` |
| `train_loss` | `1.786` |
| `val_loss` | `1.493` |

| Epoch | Samples | Median events/sec | Median batch ms | Median data wait ms |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `9` | `3352.314` | `152.730` | `14.260` |
| `1` | `10` | `3346.015` | `153.018` | `14.883` |
| `2` | `10` | `3329.341` | `153.786` | `14.907` |

All 29 logged samples preserved `512` events, `16` active lanes, and `0`
padding. Mean sampled throughput was `3340.245 events/sec`; no Triton fallback
was reported. Final throughput improved by **3.1%** over rhythm-only job
`55282240` (`3218.282 events/sec`) and **15.4%** over writer-fusion job
`55280466` (`2873.714 events/sec`). The epoch 2 median improved by **3.4%** over
rhythm-only (`3218.478 events/sec`).

Verification:

- Local model suite: `41 passed, 12 skipped` (CUDA unavailable locally).
- GPU suite before training: `20 passed, 33 deselected`. Tests cover fixed time
  features for dimensions `1`, `3`, `4`, and `5`; unseen IDs, negative elapsed
  time, tiny IATs, strided/empty inputs; repeated motif writes for lengths `1`,
  `3`, and `4`; and lane output/state/parameter-gradient equivalence against
  PyTorch with partial lanes, carried state, and independent scan options.
- Config tests: `5 passed, 23 deselected`, including primitive validation and
  rejection of a Triton motif writer without a Triton rhythm kernel.
- Ruff, documentation lint, and `git diff --check`: passed.

Status: **PASS as the current opt-in strict-rich benchmark**. These are combined
results; individual candidate speedups were not measured. The rhythm-only job
ran on `p0243`, whereas this job ran on `p0225`, so the `3.1%` incremental gain
is an observed comparison rather than a repeated same-node paired estimate.
Defaults remain PyTorch. The next profiling pass should measure the remaining
trainable projection/backbone work before adding a custom backward kernel.

## SLURM Logs

Logs are written under:

```text
/fs/ess/PAS1266/graphids/slurm_logs/
```

Useful checks:

```bash
ls -lt /fs/ess/PAS1266/graphids/slurm_logs/*5512162*
tail -n 120 /fs/ess/PAS1266/graphids/slurm_logs/ray-temporal_joint_hybrid_ssm_lite_rich_set_01_profile_55121622.err
tail -n 120 /fs/ess/PAS1266/graphids/slurm_logs/ray-temporal_joint_hybrid_ssm_lite_rich_set_01_lanes_profile_55121623.err
```

<!-- temporal-full-coverage:begin -->
## Full-Coverage Recording / Section Pilot — October 1, 2026

Implemented continuous full-coverage lane training and opt-in experimental sections.
Preflight verified **12,090,403 source events across 18 streams**, including
**57,108 attacks**, and **12,090,385 recording-successor targets**. Core/chunk
ownership passes for continuous (48,877 batches) and segmented (5,913 sections;
1,509,120 replay events) plans. The historical 512-event prefix profiles remain
explicit diagnostics, not equivalent full-epoch workloads.

Local verification: 157 data/model/config/launch tests passed, 12 CUDA tests
skipped locally. Ruff, documentation lint, and whitespace checks passed. Pitzer
GPU verification in job **55283557**: **169 passed**, including all partial-lane
Triton options. Later coverage/report regression checks also passed.

Both short diagnostics used the identical seed-42 section queue, started with
weight digest `97c56a70e1ed403d242a2393cf45ca7eb2a035e1e4a747690ec53e55021b4337`,
and scored the same first 256 section cores. No duplicate, unexpected, scored
warmup/padding events, or dropped batches were observed.

| Diagnostic metric | 32 lanes × 32 events | 64 lanes × 32 events |
| --- | ---: | ---: |
| Unique scored events | 524,288 | 524,288 |
| Replay events | 65,536 | 65,536 |
| Scored events/sec including replay and training overhead | 6,068.27 | 11,374.47 |
| Padding | 0 | 0 |

Selected **64 lanes**: end-to-end scored throughput improved **87.4%**, exceeding
the required 3%. These short diagnostics deliberately do not cover the entire
cache. Their limited-epoch audits disclose the omitted events.

The bounded validation prefixes contain **146,304 benign scored events and zero
attacks** after the existing 64-event per-recording warmup. Classification and
anomaly AUROC are therefore **unavailable**. This limits the quality comparison.

Full-epoch results are **pending**. Separate Pitzer allocations, each with one
GPU, eight CPUs, 32 GiB RAM, and an eight-hour limit:

- Continuous 18-lane exact epoch: **55283636**.
- Experimental 64-lane segmented epoch: **55283637**.
- Dependent result publication: **55283638**, runs only after both epochs succeed.

Artifacts and immutable training code snapshot:
`/fs/ess/PAS1266/graphids/dev/rf15/set_01/full_coverage_20261001_144158/`.
Each epoch writes a coverage JSON and final checkpoint, then restores that
checkpoint for the same continuous first-8,192-events-per-recording validation.
The result job replaces this bounded note section with the full measured table.
No test data is loaded. One epoch per mode is a quality pilot; finite replay
cannot guarantee recovery of GRU memory or cumulative rhythm statistics.
Segmented training remains opt-in.
<!-- temporal-full-coverage:end -->
