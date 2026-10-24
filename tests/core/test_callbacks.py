from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _DummyModule:
    def __init__(self) -> None:
        self.logged: list[tuple[str, float | int, dict]] = []

    def log(self, name: str, value: float | int, **kwargs) -> None:
        self.logged.append((name, value, kwargs))


def test_mlflow_system_metrics_summary_maps_cpu_memory_and_gpu_histories(monkeypatch):
    from graphids import _mlflow
    from graphids._mlflow import MLflowSystemMetricsSummaryCallback

    histories = {
        "system/cpu_utilization_percentage": [10.0, 30.0],
        "system/system_memory_usage_megabytes": [1024.0, 2048.0],
        "system/gpu_0_utilization_percentage": [20.0, 80.0],
        "system/gpu_1_utilization_percentage": [40.0, 60.0, 100.0],
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
            return SimpleNamespace(
                data=SimpleNamespace(
                    metrics={key: values[-1] for key, values in histories.items()}
                )
            )

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

    assert trainer.callback_metrics["resource/samples"] == 3.0
    assert trainer.callback_metrics["resource/cpu_percent_mean"] == 20.0
    assert trainer.callback_metrics["resource/cpu_percent_max"] == 30.0
    assert trainer.callback_metrics["resource/rss_gib_mean"] == 1.5
    assert trainer.callback_metrics["resource/rss_gib_max"] == 2.0
    assert trainer.callback_metrics["resource/gpu_util_percent_mean"] == 60.0
    assert trainer.callback_metrics["resource/gpu_util_percent_max"] == 100.0
    assert trainer.callback_metrics["resource/gpu_mem_gib_max"] == 4.0
    assert trainer.callback_metrics["resource/gpu_mem_percent_max"] == 50.0


def test_mlflow_system_metrics_summary_publishes_cpu_without_gpu_histories(monkeypatch):
    from graphids import _mlflow
    from graphids._mlflow import MLflowSystemMetricsSummaryCallback

    histories = {
        "system/cpu_utilization_percentage": [12.0],
        "system/system_memory_usage_megabytes": [1024.0],
    }

    lookups = []

    class FakeClient:
        def __init__(self, tracking_uri=None):
            del tracking_uri

        def get_run(self, run_id):
            del run_id
            return SimpleNamespace(
                data=SimpleNamespace(
                    metrics={
                        **{key: values[-1] for key, values in histories.items()},
                        "system/gpu_3_utilization_percentage": 1.0,
                    }
                )
            )

        def get_metric_history(self, run_id, key):
            lookups.append(key)
            if key.startswith("system/gpu_"):
                raise RuntimeError("optional GPU history unavailable")
            del run_id
            return [SimpleNamespace(value=value) for value in histories.get(key, [])]

    monkeypatch.setattr(_mlflow, "MlflowClient", FakeClient)
    for include_gpu in (False, True):
        lookups.clear()
        callback = MLflowSystemMetricsSummaryCallback(
            sample_interval_s=5, include_gpu=include_gpu
        )
        callback._run_id = "run-1"
        callback._resolved_tracking_uri = "/tmp/mlflow"
        monkeypatch.setattr(callback, "_stop_run", lambda: None)
        trainer = SimpleNamespace(callback_metrics={})

        callback.on_test_end(trainer, _DummyModule())

        assert trainer.callback_metrics["resource/samples"] == 1.0
        assert trainer.callback_metrics["resource/cpu_percent_max"] == 12.0
        assert trainer.callback_metrics["resource/rss_gib_max"] == 1.0
        assert "resource/gpu_util_percent_mean" not in trainer.callback_metrics

        assert any(key.startswith("system/gpu_") for key in lookups) is include_gpu


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.slurm])]
)
def test_phase_metrics_count_windows_chunks_and_isolate_phase_lifetimes(
    tmp_path, monkeypatch, device
):
    """Deferred counters preserve masks across devices, phase changes, and failures."""
    from graphids._mlflow import MLflowSystemMetricsCallback
    from graphids.core.data.events.data import EventStreamWindows
    from tests.event_fixtures import event_record

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _, events = event_record(tmp_path)
    callback = MLflowSystemMetricsCallback()
    monkeypatch.setattr(callback, "_sample_gpu", lambda: None)
    callback._begin_phase("train")
    callback._batch_phase(EventStreamWindows([events], 2, 1, 8192)[0].to(device))
    batch = events.batch(0, 6)
    batch.scored[:2] = False
    expected_seconds = float(batch.times[:, 1][batch.scored].sum())
    batch.times.requires_grad_()
    callback._batch_phase(batch)  # CPU chunk can join a CUDA window's phase.
    callback._begin_phase("validation")  # Flush the previous phase.
    assert callback.phase_summaries[-1]["scored_events"] == 10
    assert callback.phase_summaries[-1]["scored_seconds"] == pytest.approx(
        0.02 + expected_seconds
    )
    callback._batch_phase(batch.to(device))
    batch.scored[:] = False  # Counters must not retain aliases to the source mask.
    callback._end_phase()
    assert callback.phase_summaries[-1]["scored_events"] == 4
    assert callback.phase_summaries[-1]["scored_seconds"] == pytest.approx(
        expected_seconds
    )
    batch.times.sum().backward()
    torch.testing.assert_close(batch.times.grad, torch.ones_like(batch.times))

    callback._begin_phase("failed")
    callback._batch_phase(events.batch(0, 6).to(device))
    callback.on_exception(None, None, RuntimeError("interrupted"))
    assert callback._phase is None
    callback._begin_phase("empty")
    callback._end_phase()
    assert [row["phase"] for row in callback.phase_summaries] == [
        "train",
        "validation",
        "empty",
    ]
    assert callback.phase_summaries[-1]["scored_events"] == 0
    assert callback.phase_summaries[-1]["scored_seconds"] == 0.0
    assert isinstance(callback.phase_summaries[-1]["scored_seconds"], float)
