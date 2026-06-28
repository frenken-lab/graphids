"""MLflow tracker/logger helpers for GraphIDS.

This module is intentionally thin:
- resolve the shared tracking URI
- construct an MLflow logger
- keep system-metrics enabled when requested

Everything else belongs in Lightning or the experiment code.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
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
    run_id: str | None = None,
    system_metrics: bool = True,
) -> MLFlowLogger:
    """Create the Lightning MLflow logger used by training and evaluation."""
    configure_tracking_uri()
    tracking_uri = mlflow.get_tracking_uri()
    if system_metrics:
        mlflow.enable_system_metrics_logging()
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags or {},
        artifact_location=artifact_location,
        run_id=run_id,
    )


class MLflowSystemMetricsCallback(pl.Callback):
    """Start MLflow's system-metrics sampler for Lightning-created runs.

    ``mlflow.enable_system_metrics_logging()`` only affects fluent
    ``mlflow.start_run()``. Lightning's ``MLFlowLogger`` creates runs through
    ``MlflowClient.create_run()``, so graphids has to start the monitor from a
    callback once the logger exposes the concrete run id.
    """

    def __init__(
        self,
        *,
        sampling_interval: int = 10,
        samples_before_logging: int = 1,
    ) -> None:
        self.sampling_interval = sampling_interval
        self.samples_before_logging = samples_before_logging
        self._monitor: Any | None = None

    def _start(self, trainer: pl.Trainer) -> None:
        if self._monitor is not None:
            return
        logger = getattr(trainer, "logger", None)
        run_id = getattr(logger, "run_id", None)
        if not run_id:
            return
        try:
            from mlflow.system_metrics.system_metrics_monitor import (
                SystemMetricsMonitor,
            )
        except ImportError:
            return

        tracking_uri = mlflow.get_tracking_uri()
        self._monitor = SystemMetricsMonitor(
            run_id,
            sampling_interval=self.sampling_interval,
            samples_before_logging=self.samples_before_logging,
            tracking_uri=tracking_uri,
        )
        self._monitor.start()

    def _stop(self) -> None:
        if self._monitor is None:
            return
        try:
            self._monitor.finish()
        finally:
            self._monitor = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._start(trainer)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._stop()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._start(trainer)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._stop()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._stop()
