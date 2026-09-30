from __future__ import annotations

import sys
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


def test_resource_utilization_callback_records_cpu_rss_aggregates(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import ResourceUtilizationCallback

    class MemoryInfo:
        rss = 2 * 1024**3

    class Process:
        def __init__(self) -> None:
            self.cpu_values = iter([0.0, 10.0, 30.0])

        def cpu_percent(self, interval=None):
            del interval
            return next(self.cpu_values)

        def memory_info(self):
            return MemoryInfo()

    fake_psutil = SimpleNamespace(Process=Process)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(callbacks_mod.time, "monotonic", iter([0.0, 5.0]).__next__)

    callback = ResourceUtilizationCallback(sample_interval_s=5, include_gpu=False)
    trainer = SimpleNamespace(callback_metrics={})
    module = _DummyModule()

    callback.on_fit_start(trainer, module)
    callback.on_fit_end(trainer, module)

    assert trainer.callback_metrics["resource/samples"] == 2.0
    assert trainer.callback_metrics["resource/cpu_percent_mean"] == 20.0
    assert trainer.callback_metrics["resource/cpu_percent_max"] == 30.0
    assert trainer.callback_metrics["resource/rss_gib_max"] == 2.0


def test_resource_utilization_callback_records_gpu_aggregates(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import ResourceUtilizationCallback

    class Process:
        def cpu_percent(self, interval=None):
            del interval
            return 0.0

        def memory_info(self):
            return SimpleNamespace(rss=1024**3)

    class FakeNvml:
        def __init__(self) -> None:
            self.util_values = iter([20, 80])

        def nvmlInit(self) -> None:
            return None

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlDeviceGetHandleByIndex(self, idx: int) -> int:
            return idx

        def nvmlDeviceGetUtilizationRates(self, handle):
            del handle
            return SimpleNamespace(gpu=next(self.util_values))

        def nvmlDeviceGetMemoryInfo(self, handle):
            del handle
            return SimpleNamespace(used=3 * 1024**3, free=5 * 1024**3, total=8 * 1024**3)

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=Process))
    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml())
    monkeypatch.setattr(callbacks_mod.time, "monotonic", iter([0.0, 5.0]).__next__)

    callback = ResourceUtilizationCallback(sample_interval_s=5)
    trainer = SimpleNamespace(callback_metrics={})
    module = _DummyModule()

    callback.on_fit_start(trainer, module)
    callback.on_fit_end(trainer, module)

    assert trainer.callback_metrics["resource/gpu_util_percent_mean"] == 50.0
    assert trainer.callback_metrics["resource/gpu_util_percent_max"] == 80.0
    assert trainer.callback_metrics["resource/gpu_mem_gib_max"] == 3.0
    assert trainer.callback_metrics["resource/gpu_mem_free_gib_min"] == 5.0
    assert trainer.callback_metrics["resource/gpu_mem_total_gib_max"] == 8.0
    assert trainer.callback_metrics["resource/gpu_mem_percent_max"] == 37.5


def test_resource_utilization_callback_continues_without_nvml(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import ResourceUtilizationCallback

    class Process:
        def cpu_percent(self, interval=None):
            del interval
            return 12.0

        def memory_info(self):
            return SimpleNamespace(rss=1024**3)

    warnings = _DummyLog()
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=Process))
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(callbacks_mod.time, "monotonic", iter([0.0, 5.0]).__next__)
    monkeypatch.setattr(callbacks_mod.log, "warning", warnings.info)

    callback = ResourceUtilizationCallback(sample_interval_s=5)
    trainer = SimpleNamespace(callback_metrics={})
    module = _DummyModule()

    callback.on_fit_start(trainer, module)
    callback.on_fit_end(trainer, module)

    assert trainer.callback_metrics["resource/cpu_percent_max"] == 12.0
    assert "resource/gpu_util_percent_mean" not in trainer.callback_metrics
    assert warnings.records[0][0] == "resource_utilization_nvml_unavailable"


def test_torch_compile_callback_compiles_named_method(monkeypatch):
    import graphids.core.callbacks as callbacks_mod
    from graphids.core.callbacks import TorchCompileCallback

    calls: list[tuple[object, dict]] = []

    def fake_compile(target, **kwargs):
        calls.append((target, kwargs))

        def compiled(*args, **compiled_kwargs):
            return target(*args, **compiled_kwargs)

        return compiled

    class Module:
        def hot_path(self):
            return "ok"

    monkeypatch.setattr(callbacks_mod.torch, "compile", fake_compile)
    callback = TorchCompileCallback(method_name="hot_path", backend="inductor", mode="reduce-overhead", dynamic=False)
    trainer = SimpleNamespace(callback_metrics={})
    module = Module()

    callback.on_fit_start(trainer, module)

    assert module.hot_path() == "ok"
    assert trainer.callback_metrics["compile/enabled"] == 1.0
    assert calls[0][1] == {
        "fullgraph": False,
        "backend": "inductor",
        "mode": "reduce-overhead",
        "dynamic": False,
    }


def test_torch_compile_callback_can_fail_soft_for_missing_method():
    from graphids.core.callbacks import TorchCompileCallback

    callback = TorchCompileCallback(method_name="missing", fail_on_error=False)
    trainer = SimpleNamespace(callback_metrics={})

    callback.on_fit_start(trainer, SimpleNamespace())

    assert trainer.callback_metrics == {}
