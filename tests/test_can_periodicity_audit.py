from __future__ import annotations

import csv

from scripts.audit_can_periodicity import audit_set, summarize_capture


def test_periods_and_phase_locks_across_four_files(tmp_path):
    raw = tmp_path / "set_01"
    normal = raw / "train_01_attack_free"
    normal.mkdir(parents=True)
    names = ["ref1", "ref2", "calibration", "normal_test"]
    for index, name in enumerate(names):
        with (normal / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "arbitration_id", "data_field"])
            for i in range(100):
                t = index * 10 + i * .02
                writer.writerow([t, "A", "00"])
                writer.writerow([t + .004, "B", "01"])
    assignment = {"reference": [{"path": f"train_01_attack_free/{name}.csv"} for name in names[:2]],
                  "calibration": [{"path": "train_01_attack_free/calibration.csv"}],
                  "normal_test": [{"path": "train_01_attack_free/normal_test.csv"}]}
    result = audit_set(raw, assignment)
    assert result["stable_id_count_all_four_files"] == 2
    assert result["stable_test_frame_fraction_all_four_files"] == 1
    assert "0.02s" in result["period_bands_all_four_files"]
    assert result["same_period_phase_locks"]["reference_locked_pairs"] >= 1
    assert result["same_period_phase_locks"]["normal_test_stable_pairs"] >= 1
    single = summarize_capture(normal / "ref1.csv")
    assert single["stable_ids"] == 2
