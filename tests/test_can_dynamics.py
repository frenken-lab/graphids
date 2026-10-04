from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict

from graphids.core.data.discovery.behavior.dynamics import (
    DynamicsConfig,
    DynamicsProfile,
    PairRule,
    TimingRule,
    fit_dynamics,
    score_events,
)
from graphids.core.data.discovery.behavior.patterns import Capture, Event
from scripts.compare_can_dynamics import run_set


def event(t, key, kind="arrival"):
    return Event(t, key, kind, 1)


def test_three_scores_and_recording_reset():
    cfg = DynamicsConfig(bin_width=0.1)
    profile = DynamicsProfile(1, asdict(cfg), {}, 10,
                              [TimingRule("A", 0.15, 0.15, None, [100, 100], [.1, .1])],
                              [PairRule("A", "B", "arrival", .02, .04, [10, 10], [1, 1], [0, 0], [.03, .03])],
                              [PairRule("A", "B", "change", .02, .04, [10, 10], [1, 1], [0, 0], [.03, .03])])
    normal = [event(0, "A"), event(.01, "A", "change"), event(.03, "B"),
              event(.04, "B", "change")]
    rows = score_events(normal, 0, .3, profile)
    assert sum(x["phase"] for x in rows) == 0
    assert sum(x["change_coupling"] for x in rows) == 0
    assert sum(x["timing"] for x in rows) == 1
    changed = normal + [event(.25, "B"), event(.26, "B", "change")]
    rows = score_events(changed, 0, .3, profile)
    assert sum(x["phase"] for x in rows) == 1
    assert sum(x["change_coupling"] for x in rows) == 1
    assert score_events(normal, 0, .3, profile) == score_events(normal, 0, .3, profile)


def test_fit_retains_stable_phase_and_change_coupling():
    cfg = DynamicsConfig(min_id_samples=3, min_pair_samples=3, max_candidate_ids=3,
                         max_pair_lag=.08, sample_per_stream=100, min_control_gain=.1)
    def capture(name, offset):
        events = []
        for i in range(12):
            t = i + offset
            events.extend([event(t, "A"), event(t + .01, "A", "change"),
                           event(t + .03, "B"), event(t + .04, "B", "change")])
        return Capture(name, name, events, events[0].timestamp, events[-1].timestamp, 24, {1: 24})
    profile = fit_dynamics([capture("ref1", 0), capture("ref2", .2)], cfg)
    assert any(rule.trigger == "A" and rule.target == "B" for rule in profile.phase)
    assert any(rule.trigger == "A" and rule.target == "B" for rule in profile.change_coupling)
    assert profile.source_hashes == {"ref1": "ref1", "ref2": "ref2"}


def test_runner_uses_frozen_references_and_labels_deletion(tmp_path):
    raw = tmp_path / "raw" / "set_01"
    normal = raw / "train_01_attack_free"
    normal.mkdir(parents=True)
    names = ["accessory-1", "accessory-2", "attack-free-1", "attack-free-2"]
    for index, name in enumerate(names):
        with (normal / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
            for i in range(30):
                writer.writerow([i * .1, "A", f"{(i + index) % 256:02x}", 0])
                writer.writerow([i * .1 + .03, "B", f"{(i + index + 1) % 256:02x}", 0])
    assigned = {"reference": [names[0], names[2]], "calibration": [names[1]],
                "normal_test": [names[3]]}
    frozen = tmp_path / "pilot" / "set_01"
    frozen.mkdir(parents=True)
    assignment = {role: [{"path": f"train_01_attack_free/{name}.csv",
                          "sha256": hashlib.sha256((normal / f"{name}.csv").read_bytes()).hexdigest()}
                         for name in values] for role, values in assigned.items()}
    (frozen / "assignment.json").write_text(json.dumps(assignment))
    config = DynamicsConfig(min_id_samples=3, min_pair_samples=3, max_candidate_ids=2,
                            max_pair_lag=.08, sample_per_stream=100, synthetic_min_seconds=.3)
    report = run_set(raw, frozen, tmp_path / "out", config)
    run_set(raw, frozen, tmp_path / "out_repeat", config)
    assert (tmp_path / "out/profile.json").read_bytes() == (tmp_path / "out_repeat/profile.json").read_bytes()
    assert (tmp_path / "out/normal_test.jsonl").read_bytes() == (tmp_path / "out_repeat/normal_test.jsonl").read_bytes()
    assert set(report["assignment"]["reference"][i]["path"] for i in range(2)) == {
        "train_01_attack_free/accessory-1.csv", "train_01_attack_free/attack-free-1.csv"}
    assert report["results"]["synthetic_deletion"]["synthetic"] is True
    assert report["results"]["synthetic_deletion"]["removed_frames"] > 0
    assert report["results"]["normal_test"]["normal_exposure_hours"] > 0
    assert not (tmp_path / "out" / "attacks.jsonl").exists()


def test_deletion_rederives_payload_changes(tmp_path):
    from graphids.core.data.discovery.behavior.patterns import load_capture

    path = tmp_path / "capture.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field"])
        writer.writerows([[0, "A", "00"], [1, "A", "01"], [2, "A", "01"], [3, "A", "00"]])
    original = load_capture(path)
    deleted = load_capture(path, drop_interval=("A", 1, 3))
    assert [event.timestamp for event in original.events if event.kind == "change"] == [1, 3]
    assert [event.timestamp for event in deleted.events if event.kind == "change"] == []


def test_bus_load_selects_mode_specific_deadline():
    cfg = DynamicsConfig(bin_width=.1)
    profile = DynamicsProfile(1, asdict(cfg), {}, 2,
                              [TimingRule("A", .3, .3, .05, [100, 100], [.1, .1])], [], [])
    low = score_events([event(.02, "A")], 0, .2, profile)
    high = score_events([event(0, "X"), event(.01, "Y"), event(.02, "A")], 0, .2, profile)
    assert sum(row["timing"] for row in low) == 0
    assert sum(row["timing"] for row in high) == 1


def test_pair_candidates_must_appear_in_both_references():
    cfg = DynamicsConfig(min_id_samples=3, min_pair_samples=3,
                         max_candidate_ids=2, max_pair_lag=.02,
                         sample_per_stream=100, min_control_gain=.1)
    captures = []
    for name, offset, with_extra in [('first', 0, False), ('second', .1, True)]:
        events = []
        for i in range(30):
            t = i + offset
            events.extend([event(t, 'A'), event(t + .01, 'B')])
            if with_extra:
                events.extend(event(t + .02 + j * .001, 'X') for j in range(20))
        events.sort(key=lambda item: item.timestamp)
        captures.append(Capture(name, name, events, events[0].timestamp,
                                events[-1].timestamp, len(events), {1: len(events)}))
    profile = fit_dynamics(captures, cfg)
    assert any(rule.trigger == 'A' and rule.target == 'B' for rule in profile.phase)
