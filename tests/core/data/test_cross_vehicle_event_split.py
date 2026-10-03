"""Event-targeted whole-recording source split contracts."""

import csv

from graphids.core.data.datasets.recording_split import recording_manifest
from graphids.primitives_data import CrossVehicleCfg


def _write(path, *, events, attacked, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
        for index in range(events):
            writer.writerow([marker * 100 + index, "123", "0011223344556677", int(attacked and index >= 2)])


def test_event_target_is_deterministic_and_groups_duplicate_files(tmp_path):
    normal = tmp_path / "normal"
    attack = tmp_path / "attack"
    for index in range(8):
        _write(normal / f"n{index}.csv", events=10, attacked=False, marker=index)
    for index in range(3):
        _write(attack / f"dos-{index}.csv", events=10, attacked=True, marker=index + 10)
    (attack / "dos-copy.csv").write_bytes((attack / "dos-0.csv").read_bytes())
    entry = {"train_subdir": "normal", "train_attack_subdir": "attack"}
    args = dict(fraction=0.1, seed=42, warmup=1, train_source_mode="mixed", policy="event_target_v2")
    first = recording_manifest(tmp_path, entry, **args)
    second = recording_manifest(tmp_path, entry, **args)
    assert first == second
    assert len(first["recordings"]) == 11
    assert max(len(record["aliases"]) for record in first["recordings"]) == 2
    assert first["event_summary"]["validation_scored_benign"] > 0
    assert first["event_summary"]["validation_scored_attacks"] > 0


def test_cross_vehicle_config_uses_ten_percent_by_default():
    assert CrossVehicleCfg(held_out_vehicle="chevrolet_silverado").val_fraction == 0.1
