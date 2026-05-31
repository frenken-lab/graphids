"""MLflow result views for experiment audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from mlflow.tracking import MlflowClient

from graphids._mlflow import configure_tracking_uri
from graphids.paths import CONFIG_DIR

DEFAULT_VIEW_PATH = CONFIG_DIR / "result_views.yml"


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    group: str
    variant: str
    phase: str
    run_id: str
    status: str
    run_name: str
    start_time: int | None
    metrics: dict[str, float | None]

    def flat(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "group": self.group,
            "variant": self.variant,
            "phase": self.phase,
            "run_id": self.run_id,
            "status": self.status,
            "run_name": self.run_name,
            "start_time": self.start_time,
            **self.metrics,
        }


def load_result_view(name: str, *, path: Path = DEFAULT_VIEW_PATH) -> dict[str, Any]:
    views = yaml.safe_load(path.read_text()) or {}
    if name not in views:
        raise KeyError(f"unknown result view {name!r}; available: {', '.join(sorted(views))}")
    view = views[name] or {}
    return {
        "filter": dict(view.get("filter") or {}),
        "metrics": list(view.get("metrics") or []),
    }


def _filter_string(filters: dict[str, Any]) -> str:
    parts: list[str] = []
    status = filters.pop("status", None)
    if status:
        parts.append(f"attributes.status = '{status}'")
    for key, value in filters.items():
        if value is None:
            continue
        parts.append(f"tags.`graphids.{key}` = '{value}'")
    return " and ".join(parts)


def _metric_value(metrics: dict[str, float], key: str) -> float | None:
    if key in metrics:
        return float(metrics[key])
    # Historical rows sometimes logged both nested and top-level names. Keep
    # configured names stable while tolerating older runs.
    fallbacks = []
    if key.startswith("test/test/"):
        fallbacks.extend([key.removeprefix("test/test/"), "test/" + key.removeprefix("test/test/")])
    if key.startswith("test/"):
        fallbacks.append(key.removeprefix("test/"))
    for candidate in fallbacks:
        if candidate in metrics:
            return float(metrics[candidate])
    return None


def query_result_view(
    *,
    view: str,
    datasets: list[str],
    variants: list[str] | None = None,
    latest: bool = True,
    status: str = "FINISHED",
    view_path: Path = DEFAULT_VIEW_PATH,
) -> list[ResultRow]:
    """Query MLflow for a configured result view.

    ``latest=True`` returns the most recent run per ``(dataset, variant)``.
    Metric keys are taken from ``configs/result_views.yml``.
    """
    configure_tracking_uri()
    spec = load_result_view(view, path=view_path)
    base_filter = dict(spec["filter"])
    metrics = list(spec["metrics"])
    group = str(base_filter.get("group") or view)
    phase = str(base_filter.get("phase") or "test")
    client = MlflowClient()

    rows: list[ResultRow] = []
    for dataset in datasets:
        exp = client.get_experiment_by_name(f"graphids/{dataset}/{group}")
        if exp is None:
            continue
        filters = {**base_filter, "phase": phase, "status": status}
        filter_string = _filter_string(filters)
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
        seen: set[str] = set()
        for run in runs:
            tags = run.data.tags
            variant = tags.get("graphids.variant") or tags.get("graphids.row_name") or ""
            if variants is not None and variant not in variants:
                continue
            if latest and variant in seen:
                continue
            seen.add(variant)
            row_metrics = {key: _metric_value(run.data.metrics, key) for key in metrics}
            rows.append(
                ResultRow(
                    dataset=dataset,
                    group=group,
                    variant=variant,
                    phase=phase,
                    run_id=run.info.run_id,
                    status=run.info.status,
                    run_name=tags.get("mlflow.runName", ""),
                    start_time=run.info.start_time,
                    metrics=row_metrics,
                )
            )
    return rows


def result_rows_as_json(rows: list[ResultRow]) -> list[dict[str, Any]]:
    return [row.flat() for row in rows]


def sort_rows(rows: list[ResultRow], *, by: Literal["dataset", "variant"] = "dataset") -> list[ResultRow]:
    if by == "variant":
        return sorted(rows, key=lambda r: (r.variant, r.dataset, r.start_time or 0))
    return sorted(rows, key=lambda r: (r.dataset, r.variant, r.start_time or 0))
