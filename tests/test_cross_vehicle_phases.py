"""Cross-vehicle phase telemetry persists missing GPU samples explicitly."""

import json
from types import SimpleNamespace

import torch

from graphids._mlflow import MLflowSystemMetricsCallback


def test_phase_summary_without_gpu_samples(tmp_path, monkeypatch):
    output = tmp_path / "phases.json"
    callback = MLflowSystemMetricsCallback(phase_output=output)
    monkeypatch.setattr(callback, "_sample_gpu", lambda: None)
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False, global_step=7)
    callback.on_train_epoch_start(trainer, None)
    callback.on_train_batch_end(trainer, None, None,
                                SimpleNamespace(is_scored=torch.tensor([True, False, True])), 0)
    callback.on_train_epoch_end(trainer, None)
    row = json.loads(output.read_text())[0]
    assert row["phase"] == "train_epoch_1"
    assert row["scored_events"] == 2
    assert row["gpu_sample_count"] == 0
    assert row["gpu_utilization_mean"] is None
    assert row["events_per_second"] > 0
    callback.on_validation_epoch_start(trainer, None)
    callback.on_validation_batch_end(trainer, None, None,
                                     SimpleNamespace(is_scored=torch.tensor([True])), 0)
    callback.on_validation_epoch_end(trainer, None)
    rows = json.loads(output.read_text())
    assert [item["phase"] for item in rows] == ["train_epoch_1", "source_validation_epoch_1"]
    assert rows[1]["scored_events"] == 1
