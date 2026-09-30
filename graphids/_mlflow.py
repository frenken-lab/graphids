"""MLflow tracker/logger helpers for GraphIDS.

This module is intentionally thin:
- resolve the shared tracking URI
- construct an MLflow logger
- provide the optional online-mode system-metrics callback

Everything else belongs in Lightning or the experiment code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient

_MIB_PER_GIB = 1024.0


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
) -> MLFlowLogger:
    """Create the Lightning MLflow logger used by training and evaluation."""
    configure_tracking_uri()
    tracking_uri = mlflow.get_tracking_uri()
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags or {},
        artifact_location=artifact_location,
        run_id=run_id,
    )


class MLflowSystemMetricsCallback(pl.Callback):
    """Start MLflow's system-metrics sampler for online Lightning-created runs.

    ``mlflow.enable_system_metrics_logging()`` only affects fluent
    ``mlflow.start_run()``. Lightning's ``MLFlowLogger`` creates runs through
    ``MlflowClient.create_run()``, so graphids has to start the monitor from a
    callback once the logger exposes the concrete run id. This callback keeps
    live MLflow system metrics flowing into online runs; offline benchmark
    summaries use :class:`MLflowSystemMetricsSummaryCallback`.
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


class MLflowSystemMetricsSummaryCallback(pl.Callback):
    """Summarize MLflow-native system metrics under GraphIDS resource keys.

    Offline benchmark runs need final scalar metrics in ``trainer.callback_metrics``
    so the durable ``.graphids/mlflow_ingest.json`` payload can replay them later.
    This callback starts a local MLflow run with native system metrics enabled,
    then reads MLflow metric histories at fit/test end and maps them onto the
    existing ``resource/*`` names. ``resource/rss_gib_*`` is retained for
    compatibility, but now reflects MLflow's system memory usage metric rather
    than process RSS.
    """

    def __init__(
        self,
        *,
        prefix: str = "resource",
        tracking_uri: str | None = None,
        tracking_dir: str | Path | None = None,
        experiment_name: str = "graphids-system-metrics",
        run_name: str = "system-metrics",
        sampling_interval: int | None = None,
        samples_before_logging: int = 1,
        sample_interval_s: float | None = None,
        include_gpu: bool = True,
    ) -> None:
        self.prefix = prefix
        self.tracking_uri = tracking_uri
        self.tracking_dir = str(tracking_dir) if tracking_dir is not None else None
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.samples_before_logging = max(1, int(samples_before_logging))
        self.include_gpu = include_gpu
        interval = sampling_interval
        if interval is None and sample_interval_s is not None:
            interval = int(max(1.0, float(sample_interval_s)))
        self.sampling_interval = max(1, int(interval if interval is not None else 10))
        self._run_id: str | None = None
        self._resolved_tracking_uri: str | None = None
        self._previous_tracking_uri: str | None = None

    def _default_tracking_dir(self, trainer: pl.Trainer) -> Path:
        root = getattr(trainer, "default_root_dir", None)
        if root is None:
            root = Path.cwd()
        return Path(root) / ".graphids" / "system_metrics_mlflow"

    def _resolve_tracking_uri(self, trainer: pl.Trainer) -> str:
        if self.tracking_uri:
            return self.tracking_uri
        tracking_dir = Path(self.tracking_dir) if self.tracking_dir else self._default_tracking_dir(trainer)
        tracking_dir.mkdir(parents=True, exist_ok=True)
        return str(tracking_dir)

    def _set_tracking_uri(self, tracking_uri: str) -> None:
        self._previous_tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

    def _restore_tracking_uri(self) -> None:
        if self._previous_tracking_uri is not None:
            mlflow.set_tracking_uri(self._previous_tracking_uri)
            self._previous_tracking_uri = None

    def _start(self, trainer: pl.Trainer) -> None:
        if self._run_id is not None:
            return
        try:
            tracking_uri = self._resolve_tracking_uri(trainer)
            self._resolved_tracking_uri = tracking_uri
            self._set_tracking_uri(tracking_uri)
            mlflow.set_system_metrics_sampling_interval(self.sampling_interval)
            mlflow.set_system_metrics_samples_before_logging(self.samples_before_logging)
            mlflow.set_experiment(self.experiment_name)
            nested = mlflow.active_run() is not None
            run = mlflow.start_run(
                run_name=self.run_name,
                nested=nested,
                tags={"graphids.telemetry": "system_metrics_summary"},
                log_system_metrics=True,
            )
            self._run_id = run.info.run_id
        except Exception:  # noqa: BLE001 - telemetry must not fail runs
            self._run_id = None
            self._resolved_tracking_uri = None
            self._restore_tracking_uri()
            return

    def _stop_run(self) -> None:
        if self._run_id is None:
            return
        try:
            if self._resolved_tracking_uri is not None:
                mlflow.set_tracking_uri(self._resolved_tracking_uri)
            active = mlflow.active_run()
            if active is not None and active.info.run_id == self._run_id:
                mlflow.end_run(status="FINISHED")
        finally:
            self._restore_tracking_uri()

    @staticmethod
    def _metric_history_values(client: Any, run_id: str, key: str) -> list[float]:
        values: list[float] = []
        try:
            history = client.get_metric_history(run_id, key)
        except Exception:  # noqa: BLE001 - missing optional metric histories are expected
            return values
        for metric in history:
            value = getattr(metric, "value", None)
            if isinstance(value, int | float):
                values.append(float(value))
        return values

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _latest_metric_keys(client: Any, run_id: str) -> set[str]:
        try:
            run = client.get_run(run_id)
        except Exception:  # noqa: BLE001 - tests may mock only history lookups
            return set()
        metrics = getattr(getattr(run, "data", None), "metrics", None)
        return {str(key) for key in dict(metrics or {})}

    def _summarize(self) -> dict[str, float]:
        if self._run_id is None or self._resolved_tracking_uri is None:
            return {}

        client = MlflowClient(tracking_uri=self._resolved_tracking_uri)
        run_id = self._run_id
        keys = self._latest_metric_keys(client, run_id)
        keys.update(
            {
                "system/cpu_utilization_percentage",
                "system/system_memory_usage_megabytes",
                "system/gpu_0_utilization_percentage",
                "system/gpu_0_memory_usage_megabytes",
                "system/gpu_0_memory_usage_percentage",
            }
        )

        metrics: dict[str, float] = {}
        sample_counts: list[int] = []

        cpu_values = self._metric_history_values(client, run_id, "system/cpu_utilization_percentage")
        if cpu_values:
            sample_counts.append(len(cpu_values))
            metrics[f"{self.prefix}/cpu_percent_mean"] = self._mean(cpu_values)
            metrics[f"{self.prefix}/cpu_percent_max"] = max(cpu_values)

        memory_values = self._metric_history_values(client, run_id, "system/system_memory_usage_megabytes")
        if memory_values:
            sample_counts.append(len(memory_values))
            gib_values = [value / _MIB_PER_GIB for value in memory_values]
            metrics[f"{self.prefix}/rss_gib_mean"] = self._mean(gib_values)
            metrics[f"{self.prefix}/rss_gib_max"] = max(gib_values)

        if self.include_gpu:
            gpu_util_values: list[float] = []
            gpu_mem_gib_values: list[float] = []
            gpu_mem_percent_values: list[float] = []
            for key in sorted(keys):
                if key.startswith("system/gpu_") and key.endswith("_utilization_percentage"):
                    values = self._metric_history_values(client, run_id, key)
                    if values:
                        sample_counts.append(len(values))
                        gpu_util_values.extend(values)
                elif key.startswith("system/gpu_") and key.endswith("_memory_usage_megabytes"):
                    values = self._metric_history_values(client, run_id, key)
                    if values:
                        sample_counts.append(len(values))
                        gpu_mem_gib_values.extend(value / _MIB_PER_GIB for value in values)
                elif key.startswith("system/gpu_") and key.endswith("_memory_usage_percentage"):
                    values = self._metric_history_values(client, run_id, key)
                    if values:
                        sample_counts.append(len(values))
                        gpu_mem_percent_values.extend(values)

            if gpu_util_values:
                metrics[f"{self.prefix}/gpu_util_percent_mean"] = self._mean(gpu_util_values)
                metrics[f"{self.prefix}/gpu_util_percent_max"] = max(gpu_util_values)
            if gpu_mem_gib_values:
                metrics[f"{self.prefix}/gpu_mem_gib_max"] = max(gpu_mem_gib_values)
            if gpu_mem_percent_values:
                metrics[f"{self.prefix}/gpu_mem_percent_max"] = max(gpu_mem_percent_values)

        if sample_counts:
            metrics[f"{self.prefix}/samples"] = float(max(sample_counts))
        return metrics

    def _publish(self, trainer: pl.Trainer) -> None:
        self._stop_run()
        trainer.callback_metrics.update(self._summarize())
        self._run_id = None
        self._resolved_tracking_uri = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._start(trainer)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._publish(trainer)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._start(trainer)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._publish(trainer)

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._stop_run()
        self._run_id = None
        self._resolved_tracking_uri = None
