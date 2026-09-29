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
