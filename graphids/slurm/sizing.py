"""Optional walltime estimation from MLflow history.

Used by ``python -m graphids submit --time-from-history`` to set a tighter
wall limit for fit jobs on ``(cluster, group, dataset)`` combinations with
≥3 prior FINISHED runs. Returns ``None`` when there's nothing to estimate
from; callers fall back to the static per-length default in
``submit_profiles.json``.
"""

from __future__ import annotations

import math
import statistics


def estimate_walltime_minutes(cluster: str, group: str, dataset: str) -> int | None:
    """``ceil(p95(elapsed_mins) * 1.5)`` clamped to ``[10, 7 days]``.

    ``None`` when MLflow is unreachable, the URI is unset, or fewer than 3
    matching FINISHED runs exist. ``slurm.slurm_cluster_name`` is preferred
    over ``graphids.cluster`` (always set by SLURM; the latter can be empty
    when the submitter shell's ``GRAPHIDS_CLUSTER`` isn't exported into the
    job env).
    """
    try:
        from graphids._mlflow import build_search_filter, ensure_tracking_uri
    except ImportError:
        return None

    uri = ensure_tracking_uri()
    if uri is None:
        return None
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    try:
        client = MlflowClient(tracking_uri=uri)
        experiments = [e.experiment_id for e in client.search_experiments()]
        if not experiments:
            return None
        runs = client.search_runs(
            experiment_ids=experiments,
            filter_string=build_search_filter(
                cluster=cluster,
                group=group,
                dataset=dataset,
                phase="fit",
                status="FINISHED",
            ),
            max_results=50,
        )
    except Exception:
        return None

    elapsed = [
        (r.info.end_time - r.info.start_time) / 60000
        for r in runs
        if r.info.end_time and r.info.start_time and r.info.end_time > r.info.start_time
    ]
    if len(elapsed) < 3:
        return None
    p95 = statistics.quantiles(elapsed, n=100, method="inclusive")[94]
    return max(10, min(int(math.ceil(p95 * 1.5)), 7 * 24 * 60))


def format_hms(minutes: float | int) -> str:
    """Minutes → ``H:MM:SS`` sbatch-compatible duration (ceils to whole minutes)."""
    total = int(math.ceil(float(minutes))) * 60
    return f"{total // 3600}:{(total % 3600) // 60:02d}:00"
