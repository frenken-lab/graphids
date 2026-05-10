"""MLflow tracker/logger helpers for GraphIDS.

This module is intentionally thin:
- resolve the shared tracking URI
- construct an MLflow logger
- keep system-metrics enabled when requested

Everything else now belongs in Ray, Lightning, or the experiment code.
"""

from __future__ import annotations

import mlflow
from lightning.pytorch.loggers import MLFlowLogger


def configure_tracking_uri() -> None:
    """Default MLflow URI to ``sqlite:///{lake_root}/mlflow.db`` when unset."""
    if mlflow.config.is_tracking_uri_set():
        return
    from graphids.paths import lake_root

    mlflow.set_tracking_uri(f"sqlite:///{lake_root().rstrip('/')}/mlflow.db")


def make_logger(
    *,
    experiment_name: str,
    run_name: str,
    tags: dict[str, str] | None = None,
    artifact_location: str | None = None,
    system_metrics: bool = True,
) -> MLFlowLogger:
    """Create the Lightning MLflow logger used by training and evaluation."""
    configure_tracking_uri()
    if system_metrics:
        mlflow.enable_system_metrics_logging()
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags or {},
        artifact_location=artifact_location,
    )
