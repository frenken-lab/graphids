import csv
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.can_multiattack_manifest import Components, check_groups, repair_clock


def test_bounded_clock_repair_preserves_order_and_frame_fields(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "timestamp,arbitration_id,data_field,attack\n1.0002,A,01,0\n1.0001,B,02,1\n1.0003,C,03,0\n"
    )
    before = raw.read_bytes()
    derived = tmp_path / "derived.csv"
    result = repair_clock(raw, derived)
    rows = list(csv.DictReader(derived.open()))
    assert raw.read_bytes() == before
    assert [r["arbitration_id"] for r in rows] == ["A", "B", "C"]
    assert [r["timestamp"] for r in rows] == ["1.0002", "1.0002", "1.0003"]
    assert [r["data_field"] for r in rows] == ["01", "02", "03"]
    assert [r["attack"] for r in rows] == ["0", "1", "0"]
    assert result["changed_rows"] == 1
    assert Decimal(result["maximum_correction_seconds"]) == Decimal(".0001")
    with pytest.raises(ValueError, match="exceeds"):
        repair_clock(raw, tmp_path / "rejected.csv", Decimal(".00001"))
    assert not (tmp_path / "rejected.csv").exists()


def test_reuse_components_include_unselected_intermediates():
    groups = Components()
    groups.union("a", "unselected")
    groups.union("unselected", "b")
    canonical = {"a": "a", "b": "b"}
    check_groups(
        {"a": {"role": "val_query"}, "b": {"role": "val_query"}}, canonical, groups
    )
    with pytest.raises(ValueError, match="crosses roles"):
        check_groups(
            {"a": {"role": "train_query"}, "b": {"role": "val_query"}},
            canonical,
            groups,
        )


def test_replay_provenance_checks_attack_labels_against_interfaces(tmp_path):
    from scripts.can_pilot_manifest import compare_log

    raw = tmp_path / "raw.csv"
    log = tmp_path / "replay.log"
    raw.write_text(
        "timestamp,arbitration_id,data_field,attack\n100.0,A,01,0\n100.01,B,02,1\n"
    )
    log.write_text("(1.0) vcan0 A#01\n(1.01) vcan1 B#02\n")
    assert compare_log(raw, log, interface_labels=True)[
        "labels_from_replay_interfaces_verified"
    ]
    raw.write_text(raw.read_text().replace("B,02,1", "B,02,0"))
    with pytest.raises(ValueError, match="timed frame differs"):
        compare_log(raw, log, interface_labels=True)


def test_replay_merge_preserves_interface_order_and_audits_movements(tmp_path):
    from scripts.can_multiattack_manifest import repair_replay_clock

    raw = tmp_path / "raw.csv"
    log = tmp_path / "replay.log"
    derived = tmp_path / "derived.csv"
    raw.write_text(
        "timestamp,arbitration_id,data_field,attack\n100.0,A,01,0\n100.02,B,02,1\n100.01,C,03,0\n100.0099,D,04,0\n"
    )
    log.write_text(
        "(1.0) vcan0 A#01\n(1.02) vcan1 B#02\n(1.01) vcan0 C#03\n(1.0099) vcan0 D#04\n"
    )
    before = raw.read_bytes()
    report = repair_replay_clock(raw, log, derived)
    rows = list(csv.DictReader(derived.open()))
    assert [r["arbitration_id"] for r in rows] == ["A", "C", "D", "B"]
    assert [r["timestamp"] for r in rows] == ["100.0", "100.01", "100.01", "100.02"]
    assert [r["attack"] for r in rows] == ["0", "0", "0", "1"]
    assert report["moved_rows"] == 3
    assert report["changed_rows"] == 1
    assert len(Path(report["audit"]).read_text().splitlines()) == 3
    assert raw.read_bytes() == before
    with pytest.raises(ValueError, match="per-interface clock correction exceeds"):
        repair_replay_clock(raw, log, tmp_path / "rejected.csv", Decimal(".00001"))


def test_capture_overlap_respects_replay_bus_domains():
    from scripts.can_multiattack_manifest import acquisition_overlap

    left = dict(
        vehicle="v",
        acquisition_start=0,
        acquisition_end=10,
        interfaces=["vcan4", "vcan5"],
        clock_domain="virtual_replay",
    )
    right = dict(
        left, acquisition_start=5, acquisition_end=15, interfaces=["vcan6", "vcan7"]
    )
    assert not acquisition_overlap(left, right)
    assert acquisition_overlap(left, dict(right, interfaces=["vcan5"]))
    assert acquisition_overlap(
        dict(left, clock_domain="physical"), dict(right, clock_domain="physical")
    )
    assert not acquisition_overlap(left, dict(right, clock_domain="physical"))
