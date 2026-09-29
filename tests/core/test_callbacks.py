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
