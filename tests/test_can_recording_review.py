"""Ensure reuse leads survive file offsets and injected traffic."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "can_review",
    Path(__file__).resolve().parents[1] / "scripts/can_recording_review.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

pilot_spec = importlib.util.spec_from_file_location(
    "can_pilot", Path(__file__).resolve().parents[1] / "scripts/can_pilot_manifest.py"
)
pilot = importlib.util.module_from_spec(pilot_spec)
pilot_spec.loader.exec_module(pilot)


def write_capture(root, name, rows):
    path = root / name
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
        writer.writerows(rows)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_reuse_survives_offset_and_injection(tmp_path):
    original = [
        (i / 1000, "123", hashlib.sha256(str(i).encode()).hexdigest()[:16], 0)
        for i in range(4096)
    ]
    shifted = [(100, "321", "00", 0)]
    for timestamp, arb, data, label in original:
        shifted.append((100 + timestamp, arb, data, label))
        shifted.append((100 + timestamp, "000", "01", 1))
    a = write_capture(tmp_path, "original.csv", original)
    b = write_capture(tmp_path, "shifted.csv", shifted)
    _, first = review.inspect_recording((tmp_path, a))
    rec, second = review.inspect_recording((tmp_path, b))
    assert first
    assert first.keys() <= second.keys()
    assert rec["positive_rows"] == len(original)
    assert rec["equal_timestamp_rows"] == len(original) + 1


def test_integrity_and_hash_changes_are_visible(tmp_path):
    item = write_capture(
        tmp_path,
        "bad.csv",
        [(0, "123", "00", 0), (2, "123", "0" * 18, 0), (1, "123", "00", 2)],
    )
    rec, _ = review.inspect_recording((tmp_path, item))
    assert rec["bad_clock_line_examples"] == [4]
    assert rec["bad_payload_line_examples"] == [3]
    assert rec["bad_labels"] == 1
    with (tmp_path / "bad.csv").open("a") as handle:
        handle.write("3,123,00,0\n")
    with pytest.raises(ValueError, match="raw hash changed"):
        review.inspect_recording((tmp_path, item))


def test_upstream_clock_ties_preserve_frames_without_rewriting(tmp_path):
    capture = write_capture(
        tmp_path,
        "capture.csv",
        [(0, "123", "00", 0), (0, "000", "01", 1), (1, "123", "02", 0)],
    )
    source = tmp_path / "original.log"
    source.write_text("(10) can0 000#01\n(10) can0 123#00\n(11) can0 123#02\n")
    local = tmp_path / capture["path"]
    result = pilot.compare_log(local, source)
    assert result["rows"] == 3
    assert result["original_decreasing_clock_rows"] == 0
    assert result["acquisition_start"] == 10
    assert result["acquisition_end"] == 11
    assert hashlib.sha256(local.read_bytes()).hexdigest() == capture["sha256"]
    source.write_text("(10) can0 000#01\n(10) can0 123#00\n(12) can0 123#02\n")
    with pytest.raises(ValueError, match="timed frame differs"):
        pilot.compare_log(local, source)
    source.write_text("(10) can0 000#01\n(10) can0 123#FF\n(11) can0 123#02\n")
    with pytest.raises(ValueError, match="timed frame differs"):
        pilot.compare_log(local, source)
    source.write_text("(10) can0 000#01\n(11) can0 123#02\n(10) can0 123#00\n")
    result = pilot.compare_log(local, source)
    assert result["original_decreasing_clock_rows"] == 1
    assert result["original_largest_backstep_seconds"] == 1
