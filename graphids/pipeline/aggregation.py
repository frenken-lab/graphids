"""Seed aggregation utilities for multi-seed experiments.

Queries MLflow for all seeds in a run_group and computes mean ± std
for statistical significance reporting (TMLR submission requirement).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def aggregate_seed_metrics(
    run_group: str, experiment_ids: list[str] | None = None
) -> dict[str, Any]:
    """Query MLflow for all seeds in a run_group, compute mean ± std.

    Parameters
    ----------
    run_group : str
        The seed-independent run identity (e.g. "hcrl_sa/gat_large_curriculum_kd").
    experiment_ids : list[str] | None
        MLflow experiment IDs to search. None searches all.

    Returns
    -------
    dict with structure:
        {
            "metric_name": {"mean": float, "std": float, "n": int, "values": list[float]},
            ...
            "_seeds": [42, 123, 456],
            "_n_runs": 3,
        }
    """
    from mlflow import MlflowClient

    from graphids.config import MLFLOW_TRACKING_URI

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    filter_string = f"tags.run_group = '{run_group}' AND tags.status = 'success'"
    runs = client.search_runs(
        experiment_ids=experiment_ids or [],
        filter_string=filter_string,
        order_by=["tags.seed ASC"],
    )

    if not runs:
        log.warning("No runs found for run_group='%s'", run_group)
        return {"_seeds": [], "_n_runs": 0}

    # Collect metrics across seeds
    import numpy as np

    all_metrics: dict[str, list[float]] = {}
    seeds: list[int] = []

    for run in runs:
        seed = int(run.data.tags.get("seed", 0))
        seeds.append(seed)
        for key, value in run.data.metrics.items():
            if isinstance(value, (int, float)):
                all_metrics.setdefault(key, []).append(float(value))

    # Compute mean/std per metric
    result: dict[str, Any] = {}
    for key, values in all_metrics.items():
        arr = np.array(values)
        result[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": len(values),
            "values": values,
        }

    result["_seeds"] = seeds
    result["_n_runs"] = len(runs)

    log.info(
        "Aggregated %d seeds for %s: %s",
        len(seeds),
        run_group,
        {
            k: f"{v['mean']:.4f}±{v['std']:.4f}"
            for k, v in result.items()
            if isinstance(v, dict) and "mean" in v
        },
    )
    return result


def format_metric_table(aggregated: dict[str, Any], metrics: list[str] | None = None) -> str:
    """Format aggregated metrics as a markdown table for reporting.

    Parameters
    ----------
    aggregated : dict
        Output from aggregate_seed_metrics().
    metrics : list[str] | None
        Subset of metrics to include. None includes all.

    Returns
    -------
    Markdown table string.
    """
    lines = ["| Metric | Mean | Std | N |", "|--------|------|-----|---|"]

    for key, value in sorted(aggregated.items()):
        if key.startswith("_"):
            continue
        if not isinstance(value, dict) or "mean" not in value:
            continue
        if metrics and key not in metrics:
            continue
        lines.append(f"| {key} | {value['mean']:.4f} | {value['std']:.4f} | {value['n']} |")

    return "\n".join(lines)
