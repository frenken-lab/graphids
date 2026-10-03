"""MLflow tracker/logger helpers for GraphIDS.

This module is intentionally thin:
- resolve the shared tracking URI
- construct an MLflow logger
- provide the optional online-mode system-metrics callback

Everything else belongs in Lightning or the experiment code.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
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
        phase_output: str | Path | None = None,
    ) -> None:
        self.sampling_interval = sampling_interval
        self.samples_before_logging = samples_before_logging
        self._monitor: Any | None = None
        self.phase_output = Path(phase_output) if phase_output else None
        self.phase_summaries: list[dict[str, Any]] = []
        self._phase: dict[str, Any] | None = None
        self._trainer: pl.Trainer | None = None

    def _sample_gpu(self) -> None:
        if self._phase is None:
            return
        now = time.monotonic()
        if now - self._phase.get("last_sample_clock", float("-inf")) < self.sampling_interval:
            return
        self._phase["last_sample_clock"] = now
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            for line in result.stdout.splitlines():
                util, memory = (float(part.strip()) for part in line.split(",")[:2])
                self._phase["gpu_samples"].append((util, memory / _MIB_PER_GIB))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    def _begin_phase(self, name: str) -> None:
        self._end_phase()
        self._phase = {"phase": name, "started_at": datetime.now(UTC).isoformat(),
                       "start_clock": time.monotonic(), "scored_events": 0, "gpu_samples": [],
                       "lane_capacity": 0, "valid_events": 0, "active_lanes": 0,
                       "lane_batches": 0}
        self._sample_gpu()

    def _batch_phase(self, batch: Any) -> None:
        if self._phase is None:
            return
        mask = getattr(batch, "is_scored", None)
        if mask is not None:
            self._phase["scored_events"] += int(mask.sum().item())
        valid = getattr(batch, "valid_mask", None)
        if valid is not None:
            self._phase["lane_capacity"] += valid.numel()
            self._phase["valid_events"] += int(valid.sum().item())
            self._phase["active_lanes"] += int(valid.any(dim=1).sum().item())
            self._phase["lane_batches"] += 1
        self._sample_gpu()

    def _end_phase(self) -> None:
        if self._phase is None:
            return
        self._sample_gpu()
        phase = self._phase
        elapsed = time.monotonic() - phase["start_clock"]
        samples = phase["gpu_samples"]
        row = {"phase": phase["phase"], "started_at": phase["started_at"],
               "ended_at": datetime.now(UTC).isoformat(), "wall_seconds": elapsed,
               "scored_events": phase["scored_events"],
               "events_per_second": phase["scored_events"] / elapsed if elapsed else None,
               "lane_occupancy": (phase["valid_events"] / phase["lane_capacity"]
                                  if phase["lane_capacity"] else None),
               "mean_active_lanes": (phase["active_lanes"] / phase["lane_batches"]
                                     if phase["lane_batches"] else None),
               "gpu_sample_count": len(samples),
               "gpu_utilization_mean": sum(s[0] for s in samples) / len(samples) if samples else None,
               "gpu_utilization_max": max(s[0] for s in samples) if samples else None,
               "gpu_memory_gib_peak": max(s[1] for s in samples) if samples else None}
        self.phase_summaries.append(row)
        self._phase = None
        logger = getattr(self._trainer, "logger", None)
        if logger and hasattr(logger, "log_metrics"):
            logger.log_metrics(
                {f"phase/{row['phase']}/{key}": value for key, value in row.items()
                 if key in {"wall_seconds", "scored_events", "events_per_second",
                            "lane_occupancy", "mean_active_lanes",
                            "gpu_sample_count", "gpu_utilization_mean", "gpu_utilization_max",
                            "gpu_memory_gib_peak"} and value is not None},
                step=getattr(self._trainer, "global_step", 0),
            )
        if self.phase_output:
            self.phase_output.parent.mkdir(parents=True, exist_ok=True)
            self.phase_output.write_text(json.dumps(self.phase_summaries, indent=2) + "\n")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._begin_phase(f"train_epoch_{trainer.current_epoch + 1}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._batch_phase(batch)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._end_phase()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._begin_phase(f"source_validation_epoch_{trainer.current_epoch + 1}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if not trainer.sanity_checking:
            self._batch_phase(batch)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._end_phase()

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._end_phase()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        name = list(getattr(trainer.datamodule, "_tests", {}))
        if dataloader_idx < len(name):
            desired = name[dataloader_idx]
            if self._phase is None or self._phase["phase"] != desired:
                self._begin_phase(desired)
        elif self._phase is None:
            self._begin_phase("evaluation")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_phase(batch)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._end_phase()

    def _start(self, trainer: pl.Trainer) -> None:
        self._trainer = trainer
        if self.phase_output is None:
            root = Path(os.environ.get("GRAPHIDS_RUN_DIR", trainer.default_root_dir))
            self.phase_output = root / ".graphids" / "phase_metrics.json"
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
        self._end_phase()
        self.phase_summaries.append({"phase": "fit_complete", "optimizer_steps": trainer.global_step,
                                     "ended_at": datetime.now(UTC).isoformat()})
        if self.phase_output:
            self.phase_output.parent.mkdir(parents=True, exist_ok=True)
            self.phase_output.write_text(json.dumps(self.phase_summaries, indent=2) + "\n")
        self._stop()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._start(trainer)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._end_phase()
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
