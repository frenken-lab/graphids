# Temporal Lane Throughput Check

Use this note to evaluate the GPU benchmark submitted for the temporal
stream-lane throughput path.

## Submitted Jobs

Submitted on September 29, 2026 at about 13:27 America/New_York.

- Baseline event batching: SLURM job `55121622`
- Stream-lane batching: SLURM job `55121623`

Check scheduler state:

```bash
squeue -j 55121622,55121623 -o '%i %t %M %R %j'
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
before these jobs were submitted. When evaluating, verify the manifest/events
timestamp or SLURM log to avoid comparing against stale output.

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

Historical context from the older completed baseline profile:

```text
profile/train_events_per_sec = 196.409
1.5x threshold = 294.614 events/sec
```

Prefer the same-day baseline-vs-lane ratio from the two submitted jobs over
the historical baseline when both are available.

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
