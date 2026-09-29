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
