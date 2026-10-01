import json

import pytest

from graphids.exp.temporal_results import choose, render_report, select_lanes


@pytest.mark.parametrize("a,b,lanes", [(100, 102.9, 32), (100, 103, 64), (100, 140, 64)])
def test_select_scored_geometry(a, b, lanes):
    assert select_lanes(a, b) == lanes


def test_reject_invalid_throughput():
    with pytest.raises(ValueError):
        select_lanes(0, 1)


def test_results_require_identical_weights_and_complete_coverage(tmp_path):
    def write(mode, *, digest="same", complete=True):
        root = tmp_path / mode
        root.mkdir(exist_ok=True)
        result = {"initial_weight_sha256": digest, "fit_wall_seconds": 1, "cuda_device": "test",
                  "metrics": {"val_class_0_events": 146304, "val_class_1_events": 0},
                  "coverage": {"complete": complete, "unique": 12090403, "observed_successor_targets": 12090385,
                               "wall_seconds": 1, "scored_events_per_second": 100,
                               "warmup_events": 0, "warmup_overhead": 0, "mean_active_lanes": 18,
                               "lane_occupancy": 1, "padding_events": 0, "padding_fraction": 0}}
        (root / "comparison.json").write_text(json.dumps(result))
    write("diagnostic32")
    write("diagnostic64", digest="different")
    with pytest.raises(ValueError, match="identical weights"):
        choose(tmp_path)
    write("diagnostic64")
    write("continuous")
    write("segmented", complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        render_report(tmp_path)
    write("segmented")
    report = render_report(tmp_path)
    assert "unavailable" in report
    assert "146304" in report
    assert "One epoch per mode is a quality pilot" in report


def test_note_update_preserves_following_user_content(tmp_path):
    from graphids.exp.temporal_results import update_note

    path = tmp_path / "note.md"
    path.write_text("existing text\n")
    update_note(path, "first report\n")
    path.write_text(path.read_text() + "\nuser addition\n")
    update_note(path, "updated report\n")
    assert "existing text" in path.read_text()
    assert "user addition" in path.read_text()
    assert "first report" not in path.read_text()
