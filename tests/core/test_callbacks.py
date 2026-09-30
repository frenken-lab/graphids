from __future__ import annotations

from types import SimpleNamespace

import torch


class _DummyModule:
    def __init__(self) -> None:
        self.logged: list[tuple[str, float | int, dict]] = []

    def log(self, name: str, value: float | int, **kwargs) -> None:
        self.logged.append((name, value, kwargs))


class _DummyLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.records.append((event, kwargs))


def _run_train_batch(callback, trainer, module, batch, batch_idx: int) -> None:
    callback.on_train_batch_start(trainer, module, batch, batch_idx)
    callback.on_train_batch_end(trainer, module, None, batch, batch_idx)


def test_temporal_batch_profiler_counts_events_after_warmup_and_interval(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import TemporalBatchProfilerCallback

    times = iter([10.0, 11.0, 13.0, 14.0, 17.0, 19.0])
    struct_log = _DummyLog()
    monkeypatch.setattr(callbacks_mod.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(callbacks_mod.log, "info", struct_log.info)
    monkeypatch.setattr(callbacks_mod.torch.cuda, "is_available", lambda: False)

    callback = TemporalBatchProfilerCallback(warmup_batches=1, log_every_n_batches=2)
    trainer = SimpleNamespace(current_epoch=3)
    module = _DummyModule()
    batch = SimpleNamespace(dst=torch.arange(5))

    for batch_idx in range(3):
        _run_train_batch(callback, trainer, module, batch, batch_idx)

    assert module.logged == [
        (
            "profile/train_batch_ms",
            2000.0,
            {"on_step": True, "on_epoch": False, "logger": True, "prog_bar": False},
        ),
        (
            "profile/train_data_wait_ms",
            3000.0,
            {"on_step": True, "on_epoch": False, "logger": True, "prog_bar": False},
        ),
        (
            "profile/train_events_per_sec",
            2.5,
            {"on_step": True, "on_epoch": False, "logger": True, "prog_bar": False},
        ),
        (
            "profile/train_events",
            5,
            {"on_step": True, "on_epoch": False, "logger": True, "prog_bar": False},
        ),
    ]
    assert struct_log.records == [
        (
            "temporal_batch_profile",
            {
                "prefix": "profile",
                "train_batch": 3,
                "epoch": 3,
                "batch_idx": 2,
                "train_batch_ms": 2000.0,
                "train_data_wait_ms": 3000.0,
                "train_events_per_sec": 2.5,
                "train_events": 5,
            },
        )
    ]


def test_temporal_batch_profiler_no_cuda_path_does_not_synchronize(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import TemporalBatchProfilerCallback

    def fail_sync() -> None:
        raise AssertionError("CUDA sync should not run when CUDA is unavailable")

    times = iter([1.0, 1.25])
    monkeypatch.setattr(callbacks_mod.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(callbacks_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(callbacks_mod.torch.cuda, "synchronize", fail_sync)

    callback = TemporalBatchProfilerCallback(warmup_batches=0, log_every_n_batches=1, sync_cuda=True)
    trainer = SimpleNamespace(current_epoch=0)
    module = _DummyModule()
    batch = {"dst": torch.arange(8)}

    _run_train_batch(callback, trainer, module, batch, 0)

    assert (
        "profile/train_events",
        8,
        {"on_step": True, "on_epoch": False, "logger": True, "prog_bar": False},
    ) in module.logged


def test_temporal_batch_profiler_logs_lane_utilization(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import TemporalBatchProfilerCallback

    times = iter([1.0, 2.0])
    struct_log = _DummyLog()
    monkeypatch.setattr(callbacks_mod.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(callbacks_mod.log, "info", struct_log.info)
    monkeypatch.setattr(callbacks_mod.torch.cuda, "is_available", lambda: False)

    callback = TemporalBatchProfilerCallback(warmup_batches=0, log_every_n_batches=1)
    trainer = SimpleNamespace(current_epoch=0)
    module = _DummyModule()
    batch = SimpleNamespace(valid_mask=torch.tensor([[True, True], [True, False], [False, False]]))

    _run_train_batch(callback, trainer, module, batch, 0)

    logged = {name: value for name, value, _kwargs in module.logged}
    assert logged["profile/train_events"] == 3
    assert logged["profile/train_lane_capacity_events"] == 6
    assert logged["profile/train_active_lanes"] == 2
    assert logged["profile/train_padding_events"] == 3
    assert logged["profile/train_padding_fraction"] == 0.5
    assert struct_log.records[0][1]["train_active_lanes"] == 2


def test_temporal_batch_profiler_warns_or_fails_on_expected_event_mismatch(monkeypatch):
    import pytest

    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import TemporalBatchProfilerCallback

    times = iter([1.0, 2.0, 3.0, 4.0])
    warnings = _DummyLog()
    monkeypatch.setattr(callbacks_mod.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(callbacks_mod.log, "warning", warnings.info)
    monkeypatch.setattr(callbacks_mod.torch.cuda, "is_available", lambda: False)

    trainer = SimpleNamespace(current_epoch=0)
    module = _DummyModule()
    batch = SimpleNamespace(valid_mask=torch.tensor([[True, False]]))

    callback = TemporalBatchProfilerCallback(
        warmup_batches=0,
        log_every_n_batches=1,
        expected_train_events=2,
    )
    _run_train_batch(callback, trainer, module, batch, 0)
    assert warnings.records == [
        (
            "temporal_batch_event_mismatch",
            {
                "prefix": "profile",
                "expected_train_events": 2,
                "train_events": 1,
                "epoch": 0,
                "batch_idx": 0,
            },
        )
    ]

    callback = TemporalBatchProfilerCallback(
        warmup_batches=0,
        log_every_n_batches=1,
        expected_train_events=2,
        fail_on_event_mismatch=True,
    )
    with pytest.raises(RuntimeError, match="expected 2 but observed 1"):
        _run_train_batch(callback, trainer, module, batch, 1)


def test_mlflow_system_metrics_summary_maps_cpu_memory_and_gpu_histories(monkeypatch):
    from graphids import _mlflow
    from graphids._mlflow import MLflowSystemMetricsSummaryCallback

    histories = {
        "system/cpu_utilization_percentage": [10.0, 30.0],
        "system/system_memory_usage_megabytes": [1024.0, 2048.0],
        "system/gpu_0_utilization_percentage": [20.0, 80.0],
        "system/gpu_1_utilization_percentage": [40.0, 60.0],
        "system/gpu_0_memory_usage_megabytes": [1024.0, 3072.0],
        "system/gpu_1_memory_usage_megabytes": [2048.0, 4096.0],
        "system/gpu_0_memory_usage_percentage": [12.5, 37.5],
        "system/gpu_1_memory_usage_percentage": [25.0, 50.0],
    }

    class FakeClient:
        def __init__(self, tracking_uri=None):
            assert tracking_uri == "/tmp/mlflow"

        def get_run(self, run_id):
            assert run_id == "run-1"
            return SimpleNamespace(data=SimpleNamespace(metrics={key: values[-1] for key, values in histories.items()}))

        def get_metric_history(self, run_id, key):
            assert run_id == "run-1"
            return [SimpleNamespace(value=value) for value in histories.get(key, [])]

    monkeypatch.setattr(_mlflow, "MlflowClient", FakeClient)
    callback = MLflowSystemMetricsSummaryCallback(sample_interval_s=5)
    callback._run_id = "run-1"
    callback._resolved_tracking_uri = "/tmp/mlflow"
    monkeypatch.setattr(callback, "_stop_run", lambda: None)
    trainer = SimpleNamespace(callback_metrics={})

    callback.on_fit_end(trainer, _DummyModule())

    assert trainer.callback_metrics["resource/samples"] == 2.0
    assert trainer.callback_metrics["resource/cpu_percent_mean"] == 20.0
    assert trainer.callback_metrics["resource/cpu_percent_max"] == 30.0
    assert trainer.callback_metrics["resource/rss_gib_mean"] == 1.5
    assert trainer.callback_metrics["resource/rss_gib_max"] == 2.0
    assert trainer.callback_metrics["resource/gpu_util_percent_mean"] == 50.0
    assert trainer.callback_metrics["resource/gpu_util_percent_max"] == 80.0
    assert trainer.callback_metrics["resource/gpu_mem_gib_max"] == 4.0
    assert trainer.callback_metrics["resource/gpu_mem_percent_max"] == 50.0


def test_mlflow_system_metrics_summary_publishes_cpu_without_gpu_histories(monkeypatch):
    from graphids import _mlflow
    from graphids._mlflow import MLflowSystemMetricsSummaryCallback

    histories = {
        "system/cpu_utilization_percentage": [12.0],
        "system/system_memory_usage_megabytes": [1024.0],
    }

    class FakeClient:
        def __init__(self, tracking_uri=None):
            del tracking_uri

        def get_run(self, run_id):
            del run_id
            return SimpleNamespace(data=SimpleNamespace(metrics={key: values[-1] for key, values in histories.items()}))

        def get_metric_history(self, run_id, key):
            del run_id
            return [SimpleNamespace(value=value) for value in histories.get(key, [])]

    monkeypatch.setattr(_mlflow, "MlflowClient", FakeClient)
    callback = MLflowSystemMetricsSummaryCallback(sample_interval_s=5)
    callback._run_id = "run-1"
    callback._resolved_tracking_uri = "/tmp/mlflow"
    monkeypatch.setattr(callback, "_stop_run", lambda: None)
    trainer = SimpleNamespace(callback_metrics={})

    callback.on_test_end(trainer, _DummyModule())

    assert trainer.callback_metrics["resource/samples"] == 1.0
    assert trainer.callback_metrics["resource/cpu_percent_max"] == 12.0
    assert trainer.callback_metrics["resource/rss_gib_max"] == 1.0
    assert "resource/gpu_util_percent_mean" not in trainer.callback_metrics


def test_resource_utilization_callback_is_mlflow_summary_compatibility_shim():
    from graphids._mlflow import MLflowSystemMetricsSummaryCallback
    from graphids.core.callbacks import ResourceUtilizationCallback

    callback = ResourceUtilizationCallback(sample_interval_s=5, include_gpu=False)

    assert isinstance(callback, MLflowSystemMetricsSummaryCallback)
    assert callback.sampling_interval == 5
    assert callback.include_gpu is False
