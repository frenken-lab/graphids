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
