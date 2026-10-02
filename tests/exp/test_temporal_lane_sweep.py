from copy import deepcopy

import pytest

from graphids.exp.temporal_lane_sweep import publish, should_try_1024, validate_trial


def trial(rate=100):
    return {
        "initial_weight_sha256": "weights", "section_queue_sha256": "sections", "cuda_device": "gpu",
        "planned_scored_events": 1000, "planned_warmup_events": 256, "planned_successor_targets": 999,
        "coverage": {"unique": 1000, "warmup_events": 256, "scored_events_per_second": rate,
                     "observed_successor_targets": 999, "duplicate": 0, "unexpected": 0,
                     "incorrectly_scored_warmup_or_padding": 0, "duplicate_successor_targets": 0,
                     "unexpected_successor_targets": 0, "padding_fraction": 0},
        "metrics": {"val_loss": 1}, "cuda_peak_memory": {"allocated_mib": 10, "reserved_mib": 20},
        "fit_wall_seconds": 2,
    }


@pytest.mark.parametrize("rate512,expected", [(102.9, False), (103, True), (200, True)])
def test_1024_gate(rate512, expected):
    assert should_try_1024(100, rate512) is expected


@pytest.mark.parametrize("key", ["initial_weight_sha256", "section_queue_sha256", "planned_scored_events", "cuda_device"])
def test_equal_work_and_initialization_required(key):
    reference = trial()
    changed = deepcopy(reference)
    changed[key] = "different"
    with pytest.raises(ValueError):
        validate_trial(changed, reference)


@pytest.mark.parametrize("key", ["unique", "warmup_events", "observed_successor_targets", "duplicate", "unexpected", "incorrectly_scored_warmup_or_padding"])
def test_invalid_coverage_rejected(key):
    changed = trial()
    changed["coverage"][key] = -1
    with pytest.raises(ValueError):
        validate_trial(changed)


def test_report_keeps_full_epoch_section(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("existing full epoch report\n")
    result = publish(tmp_path, {64: trial(), 128: trial(200), 256: trial(201), 512: trial(202)}, note, skipped_1024=True)
    assert result["selected_performance_lanes"] == 128
    assert "existing full epoch report" in note.read_text()
    assert "1,024 lanes skipped" in note.read_text()
    assert "AUROCs are unavailable" in note.read_text()
