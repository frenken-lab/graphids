"""Independent vehicle-scope, causal-history and complete-cohort control checks."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from graphids.data.context import previous_rows
from scripts.event_causal_controls import (
    control_scores,
    evaluate,
    fit_controls,
    markdown,
)
from tests.test_event_failure_report import events, report_fixture


def test_profiles_use_only_verified_training_and_keep_numeric_ids_vehicle_local():
    a = events("a", [10] * 5, [0] * 5, normal=True)
    b = events("b", [10, 11, 10, 11, 10], [0] * 5, normal=True, vehicle="b")
    b.arrays["times"] *= 10
    forbidden = [
        SimpleNamespace(recording=SimpleNamespace(role=role, verified_normal=flag))
        for role, flag in (
            ("val_query", True),
            ("target_reference", True),
            ("train_query", False),
        )
    ]
    profiles = fit_controls([a, b, *forbidden], min_support=2)
    assert profiles["v"]["ids"] == [10]
    assert profiles["b"]["ids"] == [10, 11]
    assert profiles["v"]["timing_statistics"][2][0] == 1
    assert profiles["b"]["timing_statistics"][2][0] == 20
    query = events("q", [11, 10, 10], [0] * 3, role="val_query")
    query.arrays["times"] = np.array([0.0, 1.0, 1.25])
    result = control_scores(query, profiles)
    assert result["id_novelty"].tolist() == [1, 0, 0]
    assert result["gap_supported"].tolist() == [False, False, True]
    # Independent serial log-gap calculation against the frozen vehicle table.
    scale, center, spread, clip = profiles["v"]["timing_statistics"][2]
    assert result["gap"][-1] == abs(
        (np.log1p(min(0.25, clip) / scale) - center) / spread
    )
    query.recording.vehicle = "unprofiled"
    missing = control_scores(query, profiles)
    for name in (
        "id_novelty",
        "gap",
        "id_novelty_supported",
        "gap_supported",
        "known_id",
    ):
        assert not missing[name].any()


def test_history_is_preceding_only_unknown_labels_still_advance_and_resets():
    train = events("train", [10] * 5, [0] * 5, normal=True)
    profiles = fit_controls([train], min_support=2)
    query = events(
        "query", [10, 11, 10, 10], [0] * 4, role="val_query", labels=[-1, 0, 1, 0]
    )
    original = control_scores(query, profiles)
    assert original["gap_supported"].tolist() == [False, False, True, True]
    modified = deepcopy(query)
    modified.arrays["ids"][3] = 255
    modified.arrays["times"][3] += 10
    modified.arrays["labels"][:] = 1
    modified.arrays["payload"][:] = 255
    modified.previous_rows = previous_rows(modified.arrays["ids"])
    for name, value in original.items():
        np.testing.assert_array_equal(
            control_scores(modified, profiles)[name][:3], value[:3]
        )
    # Independent physical-row oracle, including reset and a tied timestamp.
    query.arrays["times"][3] = query.arrays["times"][2]
    result = control_scores(query, profiles)
    prior = {}
    for row, key in enumerate(query.arrays["ids"]):
        assert result["first_seen"][row] == (key not in prior)
        if key in prior and key == 10:
            scale, center, spread, clip = profiles["v"]["timing_statistics"][2]
            gap = query.arrays["times"][row] - prior[key]
            assert result["gap"][row] == abs(
                (np.log1p(min(gap, clip) / scale) - center) / spread
            )
        prior[key] = query.arrays["times"][row]
    assert control_scores(deepcopy(query), profiles)["first_seen"][0]


def test_low_support_ids_abstain_even_when_present_in_normal_training():
    train = events("train", [10, 11, 10, 10], [0] * 4, normal=True)
    profiles = fit_controls([train], min_support=3)
    assert profiles["v"]["timing_ids"] == []
    query = events("q", [10, 11, 10, 12], [0] * 4, role="val_query")
    result = control_scores(query, profiles)
    assert result["known_id"].tolist() == [True, True, True, False]
    assert not result["gap_supported"].any()
    assert not result["gap"].any()
    with pytest.raises(ValueError, match="min_support"):
        fit_controls([train], min_support=0)
    train.arrays["labels"][0] = 1
    with pytest.raises(ValueError, match="nonnormal"):
        fit_controls([train])


def test_full_control_report_retains_unknown_missing_payload_and_first_occurrences(
    tmp_path,
):
    dm = report_fixture(tmp_path)
    output = tmp_path / "control-scores"
    output.mkdir()
    report = evaluate(dm, tmp_path, min_support=2, output=output)
    assert report["coverage"] == dict(normal=4, weak=6)
    assert report["profiles"]["v"]["recordings"] == ["train"]
    weak = report["by_recording"]["weak"]
    assert (weak["rows"], weak["positives"], weak["unknown"]) == (6, 4, 1)
    gap = weak["detectors"]["gap"]
    assert gap["supported_metrics"]["frames"] == 3
    assert weak["detectors"]["payload"]["supported_metrics"]["frames"] == 4
    assert gap["at_normal_budgets"][0]["positives"] == 4
    assert gap["at_normal_budgets"][0]["unsupported_positives"] == 1
    assert (
        weak["strata"]["gap_supported"]["detectors"]["original"]["supported_metrics"][
            "frames"
        ]
        == 3
    )
    assert weak["strata"]["first_seen"]["positives"] == 1
    assert report["aggregate"]["original"]["supported_metrics"]["frames"] == 9
    assert report["aggregate"]["gap"]["supported_metrics"]["frames"] == 6
    assert (
        report["aggregate"]["gap"]["macro_recordings"] == 0
    )  # support in weak is all positive
    assert report["aggregate"]["original"]["macro_recordings"] == 1
    for points in report["normal_budget_points"].values():
        for point in points:
            assert point["achieved_fp"] <= point["allowed_fp"]
            assert point["normal_frames"] == 4
    for key, artifact in report["control_artifacts"].items():
        arrays = np.load(output / artifact["file"])
        assert len(arrays["gap"]) == dm.event_recordings[key].count
    assert "Fixed ID-novelty alarm" in markdown(report)
    assert report["saved_metric_parity"]["passed"]
    # Validation labels/scores cannot change the fitted training controls.
    assert (
        fit_controls(dm.event_recordings.values(), min_support=2) == report["profiles"]
    )


def test_binary_novelty_ties_can_exhaust_budget_and_missing_vehicles_abstain(tmp_path):
    dm = report_fixture(tmp_path)
    # Change vehicle metadata only; the saved physical-row alignment stays valid.
    dm.event_recordings["normal"].recording.vehicle = "b"
    dm.event_recordings["weak"].recording.vehicle = "unprofiled"
    dm.event_recordings["b-train"] = events(
        "b-train", [12, 12, 12, 12], [0] * 4, normal=True, vehicle="b"
    )
    # Retain one normal capture with gap support to permit the timing budget.
    dm.event_recordings["train"].recording.vehicle = "b"
    report = evaluate(dm, tmp_path, min_support=2)
    # Here ID 10 is in the union, so the fixed novelty alarm remains quiet.
    assert report["by_recording"]["normal"]["fixed_id_novelty"]["fp"] == 0
    assert (
        report["by_recording"]["weak"]["detectors"]["id_novelty"]["at_normal_budgets"][
            0
        ]["unsupported_positives"]
        == 4
    )
    from scripts.event_failure_report import budget_point, decisions

    normal_scores = np.array([1.0, 1.0, 0.0, 0.0])
    point = budget_point(normal_scores, np.ones(4, bool), 0.25)
    assert point["achieved_fp"] == 0
    assert decisions(np.array([1.0]), np.ones(1, bool), np.array([1]), point)["tp"] == 0


def test_evaluation_rejects_target_queries_and_invalid_normal_metadata(tmp_path):
    dm = report_fixture(tmp_path)
    dm.event_recordings["target"] = SimpleNamespace(
        recording=SimpleNamespace(role="target_attack_test")
    )
    with pytest.raises(ValueError, match="source query"):
        evaluate(dm, tmp_path)
    del dm.event_recordings["target"]
    dm.event_recordings["normal"].arrays["labels"][0] = -1
    with pytest.raises(ValueError, match="nonnormal"):
        evaluate(dm, tmp_path)
