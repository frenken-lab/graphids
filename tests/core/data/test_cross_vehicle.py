"""Small fixtures for cross-vehicle provenance and causal feature contracts."""

from __future__ import annotations

import csv

import polars as pl
import pytest

from graphids.core.data.datamodule.temporal import TemporalDataModule
from graphids.core.data.datasets.can_bus import declared_vehicle_id
from graphids.core.data.datasets.cross_vehicle import (
    CrossVehicleTemporalSource,
    assign_local_ids,
)
from graphids.core.data.preprocessing.representations import TemporalRepresentationCfg
from graphids.core.data.preprocessing.temporal import (
    build_temporal_event_table,
    temporal_to_pyg,
)
from graphids.core.models.temporal._hybrid.model import TemporalHybridModel
from scripts.cross_vehicle_compare import summarize
from scripts.cross_vehicle_evaluate import (
    episode_latencies,
    score_partition,
    threshold_at_fpr,
)


def _write(path, *, marker: int, attacked: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
        for i in range(6):
            writer.writerow([1000 + marker + i * 0.1, f"{marker:03X}",
                             f"{(marker + i) % 255:02X}" * 8, int(attacked and i >= 3)])


def test_published_vehicle_mapping():
    assert declared_vehicle_id("set_01", "train_01_attack_free") == "chevrolet_impala"
    assert declared_vehicle_id("set_01", "test_02_unknown_vehicle_known_attack") == "chevrolet_silverado"
    assert declared_vehicle_id("set_03", "train_02_with_attacks") == "chevrolet_silverado"


def test_local_ids_are_collision_free_and_reset_per_recording():
    rows = pl.DataFrame({
        "vehicle_id": ["v"] * 6, "source_file": ["a"] * 4 + ["b"] * 2,
        "arb_id": ["B", "A", "B", "C", "A", "B"],
    })
    assert assign_local_ids(rows)["node_id"].to_list() == [1, 2, 1, 3, 1, 2]


def test_behavioral_features_use_only_past_same_id():
    rows = pl.DataFrame({
        "timestamp": [0.0, 0.1, 0.2, 0.4], "arb_id": ["A", "B", "A", "A"],
        "node_id": [1, 2, 1, 1], "byte_0": [1.0, 2.0, 3.0, 3.0],
        "vehicle_id": ["v"] * 4, "source_dir": ["s"] * 4, "source_file": ["f"] * 4,
    })
    table = build_temporal_event_table(rows)
    assert table["id_first_seen"].to_list() == [1.0, 1.0, 0.0, 0.0]
    assert table["payload_change_fraction"].to_list() == pytest.approx([0, 0, 1 / 8, 0])
    assert table["id_period_deviation"].to_list()[2] == 0
    changed_future = rows.with_columns(pl.when(pl.int_range(pl.len()) == 3)
                                       .then(200).otherwise(pl.col("byte_0")).alias("byte_0"))
    assert build_temporal_event_table(changed_future)["payload_change_fraction"].to_list()[:3] == (
        table["payload_change_fraction"].to_list()[:3]
    )
    assert temporal_to_pyg(table, features="behavioral").msg.shape == (4, 5)


def test_cross_vehicle_source_separates_target_calibration_and_test(tmp_path):
    raw = tmp_path / "raw"
    for set_name in ("set_01", "set_02", "set_04"):
        for i in range(2):
            _write(raw / set_name / "train_01_attack_free" / f"normal-{i}.csv",
                   marker=100 + i + 10 * int(set_name[-1]))
            _write(raw / set_name / "train_02_with_attacks" / f"dos-{i}.csv",
                   marker=200 + i + 10 * int(set_name[-1]), attacked=True)
    for i in range(2):
        _write(raw / "set_03" / "train_01_attack_free" / f"normal-{i}.csv", marker=300 + i)
    _write(raw / "set_01" / "test_02_unknown_vehicle_known_attack" / "dos-1.csv",
           marker=400, attacked=True)
    _write(raw / "set_01" / "test_04_unknown_vehicle_unknown_attack" / "fuzzing-1.csv",
           marker=401, attacked=True)

    source = CrossVehicleTemporalSource(
        held_out_vehicle="chevrolet_silverado", lake_root=str(tmp_path),
        representation_cfg=TemporalRepresentationCfg(features="behavioral"),
        val_warmup_events=1, test_warmup_events=1,
    )
    state = source.build()
    assert source.cache_ready()
    assert set(state.test) == {"target_calibration", "target_normal_test",
                               "target_known_attack", "target_unknown_attack"}
    assert int(state.train.y.sum()) > 0 and int(state.val.y.sum()) > 0
    assert int(state.test["target_calibration"].y.sum()) == 0
    assert state.train.msg.shape[1] == 5
    assert source.num_ids == 8193
    assert source.build().train.event_id.tolist() == state.train.event_id.tolist()


def test_threshold_uses_only_calibration_scores():
    scores = __import__("torch").tensor([0.1, 0.2, 0.3, 0.4])
    threshold = threshold_at_fpr(scores, 0.25)
    assert int((scores >= threshold).sum()) <= 1


def test_attack_episodes_allow_benign_frames_between_injections():
    import torch

    scores = torch.tensor([0.1, 0.2, 0.9, 0.1, 0.8])
    labels = torch.tensor([1, 0, 1, 0, 1])
    times = torch.tensor([0.0, 0.1, 0.2, 1.0, 2.0])
    streams = torch.zeros(5, dtype=torch.long)
    latencies, episodes = episode_latencies(scores, labels, times, streams, 0.5)
    assert episodes == 2
    assert latencies == pytest.approx([0.2, 0.0])


def test_pilot_gate_uses_all_three_held_out_vehicles():
    reports = {}
    for fold in ("silverado", "forester", "traverse"):
        reports[fold] = {}
        for variant, ap, recall, alarms in (("raw", 0.5, 0.4, 2.0),
                                           ("behavioral", 0.53, 0.5, 2.1)):
            partition = {name: {"auprc": ap, "recall": recall}
                         for name in ("target_known_attack", "target_unknown_attack")}
            partition["target_normal_test"] = {"false_alerts_per_hour": alarms}
            reports[fold][variant] = {"zero_shot": partition, "normal_target_adaptation": partition}
    result = summarize(reports)
    assert result["zero_shot"]["pilot_rule_passed"]
    assert result["normal_target_adaptation"]["auprc_gain_folds"] == 3


def test_lane_predictions_keep_event_order_metadata(tmp_path):
    import lightning.pytorch as lightning

    raw = tmp_path / "raw"
    for set_name in ("set_01", "set_02", "set_04"):
        for i in range(2):
            _write(raw / set_name / "train_01_attack_free" / f"normal-{i}.csv",
                   marker=500 + i + 10 * int(set_name[-1]))
            _write(raw / set_name / "train_02_with_attacks" / f"dos-{i}.csv",
                   marker=600 + i + 10 * int(set_name[-1]), attacked=True)
    for i in range(2):
        _write(raw / "set_03" / "train_01_attack_free" / f"normal-{i}.csv", marker=700 + i)
    _write(raw / "set_01" / "test_02_unknown_vehicle_known_attack" / "dos-1.csv",
           marker=800, attacked=True)
    _write(raw / "set_01" / "test_04_unknown_vehicle_unknown_attack" / "fuzzing-1.csv",
           marker=801, attacked=True)
    source = CrossVehicleTemporalSource(
        held_out_vehicle="chevrolet_silverado", lake_root=str(tmp_path),
        representation_cfg=TemporalRepresentationCfg(features="behavioral"),
        val_warmup_events=1, test_warmup_events=1,
    )
    dm = TemporalDataModule(dataset=source, batch_mode="stream_lanes", stream_lanes=2,
                            chunk_size=3, eval_batch_mode="stream_lanes")
    dm.setup(None)
    dm._tests = {"source_validation": dm.val_data, **dm.test_data}
    model = TemporalHybridModel(objective="supervised", input={"use_id_embeddings": False},
                                memory={"enabled": False}, backbone={"type": "gru"})
    model.prepare_from_datamodule(dm)
    trainer = lightning.Trainer(accelerator="cpu", devices=1, logger=False,
                                enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False)
    trainer.test(model, datamodule=dm)
    for name, data in dm.test_data.items():
        scored = score_partition(data, model._test_predictions[name])
        assert scored["scores"].numel() == int(data.is_scored.sum())
