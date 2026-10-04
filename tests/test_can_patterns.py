from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from graphids.can_patterns import (
    Capture,
    Event,
    MinerConfig,
    Pattern,
    PatternCatalog,
    PatternMiner,
    PatternTracker,
    TrackerState,
    load_capture,
)
from scripts.mine_can_patterns import assign


def catalog() -> PatternCatalog:
    cfg = MinerConfig(max_streams=4, max_patterns=3, max_active=20)
    pattern = Pattern(["A|arrival", "B|arrival", "C|arrival", "A|arrival"],
                      [[0, 0.1], [0, 0.1], [0, 0.1]], 20, 0.9, 0.02, [10, 10], 0.1)
    return PatternCatalog(1, ["A", "B", "C"], ["arrival", "change"], [pattern], {}, asdict(cfg))


def e(t, key):
    return Event(t, key, "arrival", 8)


def test_cycle_interleaving_overlap_deadline_and_reset():
    cat = catalog()
    state = TrackerState()
    features, scores, state = PatternTracker.step(
        [e(0, "A"), e(0.01, "X"), e(0.02, "A"), e(0.04, "B"), e(0.06, "C"), e(0.08, "A")],
        cat, state, 0.09)
    assert features["relationship_completed"][0] == 2
    assert features["relationship_active"][0] >= 1
    assert scores["pattern_alert"] == 0
    features, scores, state = PatternTracker.step([], cat, state, 0.18)
    assert features["relationship_overdue"][0] >= 1
    assert scores["pattern_alert"] == 1
    with pytest.raises(ValueError, match="reset"):
        PatternTracker.step([e(0, "A")], cat, state, 0.1)
    features, _, _ = PatternTracker.step([e(0, "A")], cat, TrackerState(), 0.05)
    assert features["relationship_overdue"][0] == 0


def test_raw_width_change_and_capture_boundary(tmp_path):
    paths = []
    for n in range(2):
        path = tmp_path / f"{n}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
            writer.writerows([[1, "001", "00", 0], [1.1, "001", "0000", 0], [1.2, "001", "0000", 0]])
        paths.append(path)
    captures = [load_capture(p) for p in paths]
    assert captures[0].payload_lengths == {1: 1, 2: 2}
    assert [x.kind for x in captures[0].events] == ["arrival", "arrival", "change", "arrival"]
    assert captures[0].events[0].arb_id == "001"
    assert captures[0].rows == 3 and captures[1].rows == 3
    assert captures[0].sha256 == hashlib.sha256(paths[0].read_bytes()).hexdigest()


def test_miner_deterministic_reference_only():
    cfg = MinerConfig(max_lag=0.08, min_support=3, min_completion=0.5,
                      min_lift=1.1, max_streams=4, sample_per_stream=100)
    def capture(name, shift):
        events = []
        for i in range(10):
            t = i + shift
            events.extend([e(t, "A"), e(t + .02, "B"), e(t + .04, "C"), e(t + .06, "A")])
        return Capture(name, name, events, events[0].timestamp, events[-1].timestamp, 40, {8: 40})
    refs = [capture("ref1", 0), capture("ref2", 0.2)]
    first = PatternMiner.fit(refs, cfg)
    second = PatternMiner.fit(refs, cfg)
    assert first.to_dict() == second.to_dict()
    assert any(p.steps == ["A|arrival", "B|arrival", "C|arrival", "A|arrival"] for p in first.patterns)
    assert set(first.source_hashes) == {"ref1", "ref2"}
    assert all(len(p.steps) <= 4 and len(set(p.steps)) <= 3 for p in first.patterns)


def test_assignment_saves_only_normal_file_hashes(tmp_path):
    root = tmp_path / "set_01"
    normal = root / "train_01_attack_free"
    normal.mkdir(parents=True)
    for name in ["accessory-1", "accessory-2", "attack-free-1", "attack-free-2"]:
        (normal / f"{name}.csv").write_text(name)
    attack = root / "test_01_known_vehicle_known_attack"
    attack.mkdir()
    (attack / "attack.csv").write_text("secret")
    assignment = assign(root)
    assert len(assignment["reference"]) == 2
    assert "accessory-1" in assignment["reference"][0]["path"]
    assert "attack-free-1" in assignment["reference"][1]["path"]
    assert all("test_" not in item["path"] for values in assignment.values() for item in values)


def test_runner_freezes_reference_and_repeats_scores(tmp_path):
    from scripts.mine_can_patterns import run_set

    raw = tmp_path / "set_01"
    normal = raw / "train_01_attack_free"
    normal.mkdir(parents=True)
    for index, name in enumerate(["accessory-1", "accessory-2", "attack-free-1", "attack-free-2"]):
        with (normal / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
            writer.writerows([[0, "001", "00", 0], [.01, "002", "01", 0],
                             [.02, "001", f"{index + 2:02x}", 0]])
    attack = raw / "test_01_known_vehicle_known_attack"
    attack.mkdir()
    (attack / "bad.csv").write_text("invalid attack file that must not be read")
    cfg = MinerConfig(min_support=5)
    one = run_set(raw, tmp_path / "out1", cfg, include_attacks=False)
    run_set(raw, tmp_path / "out2", cfg, include_attacks=False)
    assert one["patterns"] == 0
    saved = json.loads((tmp_path / "out1/catalog.json").read_text())
    assert {Path(source).name for source in saved["source_hashes"]} == {"accessory-1.csv", "attack-free-1.csv"}
    assert set(one["assignment"]) == {"reference", "calibration", "normal_test"}
    assert (tmp_path / "out1/catalog.json").read_bytes() == (tmp_path / "out2/catalog.json").read_bytes()
    assert (tmp_path / "out1/tables/normal_test_attack-free-2.jsonl").read_bytes() == (
        tmp_path / "out2/tables/normal_test_attack-free-2.jsonl").read_bytes()
    assert one["results"]["normal_test"][0]["normal_exposure_hours"] > 0


def test_decreasing_raw_timestamps_are_counted_and_sorted(tmp_path):
    path = tmp_path / "out_of_order.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field"])
        writer.writerows([[2, "A", "02"], [1, "A", "01"], [3, "A", "03"]])
    capture = load_capture(path)
    assert capture.decreasing_timestamps == 1
    assert [e.timestamp for e in capture.events if e.kind == "arrival"] == [1, 2, 3]
    assert [e.timestamp for e in capture.events if e.kind == "change"] == [2, 3]


def test_malformed_payload_keeps_raw_width(tmp_path):
    path = tmp_path / "malformed.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field"])
        writer.writerows([[1, "A", "00"], [2, "A", "0ZZ"]])
    capture = load_capture(path)
    assert capture.invalid_payloads == 1
    assert capture.raw_payload_chars == {2: 1, 3: 1}
    assert capture.events[-1].raw_payload_chars == 3
