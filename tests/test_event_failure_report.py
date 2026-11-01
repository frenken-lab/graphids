"""Independent denominator, chronology, source-only and artifact-alignment checks."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from graphids.data.context import previous_rows
from graphids.evaluation import EventScores
from scripts.event_failure_report import (
    ARMS,
    budget_point,
    byte_keys,
    decisions,
    diagnose,
    fit_profiles,
    markdown,
    payload_features,
    read_scores,
    timing_features,
)


def events(
    key,
    ids,
    values,
    *,
    role="train_query",
    normal=False,
    vehicle="v",
    labels=None,
    lengths=None,
):
    n = len(ids)
    payload = np.zeros((n, 8), dtype=np.uint8)
    payload[:, 0] = values
    return SimpleNamespace(
        recording=SimpleNamespace(
            key=key,
            role=role,
            vehicle=vehicle,
            verified_normal=normal,
            payload_length_verified=True,
            labels_verified=True,
            sha256="a" * 64,
        ),
        count=n,
        digest=f"digest-{key}",
        previous_rows=previous_rows(np.array(ids)),
        arrays=dict(
            ids=np.array(ids),
            times=np.arange(n, dtype=float),
            payload=payload,
            lengths=np.array(lengths if lengths is not None else [1] * n),
            labels=np.array(labels if labels is not None else [0] * n),
        ),
    )


def test_budget_respects_ties_and_keeps_unsupported_normal_exposure():
    scores = np.array([0.9, 0.9, 0.5, 0.1, 100])
    support = np.array([True, True, True, True, False])
    for rate in (0, 0.2, 0.4, 0.8):
        point = budget_point(scores, support, rate)
        allowed = int(rate * len(scores))
        candidates = [np.nextafter(v, np.inf) for v in np.unique(scores[support])]
        candidates.append(scores[support].min())
        possible = [int((scores[support] >= t).sum()) for t in candidates]
        assert point["achieved_fp"] == max(n for n in possible if n <= allowed)
        assert point["normal_frames"] == 5
        assert point["supported_normal_frames"] == 4
    assert budget_point(scores, support, 0.2)["achieved_fp"] == 0
    labels = np.array([1, 0, -1, 1])
    result = decisions(
        np.array([1.0, 0.9, 10, 10]),
        np.array([True, True, True, False]),
        labels,
        dict(threshold=0.95),
    )
    assert result == dict(
        tp=1, fp=0, positives=2, negatives=1, unsupported_positives=1, recall=0.5, fpr=0
    )


def test_training_profiles_exclude_validation_attack_and_reserved_arrays():
    normal = events("normal", [10, 11, 10, 10], [0, 9, 1, 1], normal=True)
    other = events("other", [10, 10], [7, 8], normal=True, vehicle="other")
    excluded = [
        SimpleNamespace(recording=SimpleNamespace(role=role, verified_normal=flag))
        for role, flag in (
            ("val_query", True),
            ("train_query", False),
            ("target_reference", True),
        )
    ]
    profiles = fit_profiles([normal, other, *excluded])
    query = events(
        "query", [10, 11, 10, 10, 10, 10], [0, 9, 1, 0, 2, 7], role="val_query"
    )
    features = payload_features(query, profiles["v"])
    assert features["novel_value"].tolist() == [False, False, False, False, True, True]
    assert features["novel_transition_known_values"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert not payload_features(query, profiles["pooled"])["novel_value"][-1]
    assert features["transition_available"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    # A current payload mutation cannot alter any preceding transition feature.
    changed = deepcopy(query)
    changed.arrays["payload"][-1, 0] = 255
    for name, value in features.items():
        np.testing.assert_array_equal(
            payload_features(changed, profiles["v"])[name][:-1], value[:-1]
        )


def test_same_id_transition_crosses_chunks_and_respects_lengths_and_provenance():
    record = events("record", [10, 11, 10, 10], [0, 255, 1, 0], lengths=[1, 1, 1, 0])
    whole, valid = byte_keys(record, 0, 4, transition=True)
    part, part_valid = byte_keys(record, 2, 4, transition=True)
    np.testing.assert_array_equal(part, whole[2:])
    np.testing.assert_array_equal(part_valid, valid[2:])
    assert whole[2, 0] == (10 * 8 * 256 + 0) * 256 + 1
    assert valid.sum() == 1
    record.recording.payload_length_verified = False
    assert not byte_keys(record, 0, 4)[1].any()
    assert not byte_keys(record, 0, 4, transition=True)[1].any()


def test_timing_support_first_seen_and_fallback_follow_checkpoint_tables():
    record = events("timing", [10, 11, 10, 11], [0, 0, 0, 0])
    statistics = np.array([[1, 0, 1, 10], [2, 0, 1, 10], [1, 0, 1, 1]])
    result = timing_features(record, np.array([10]), statistics)
    assert result["first_seen"].tolist() == [True, True, False, False]
    assert result["timing_supported"].tolist() == [False, False, True, False]
    assert result["timing_fallback"].tolist() == [False, False, False, True]
    assert result["timing_clipped"].tolist() == [False, False, True, False]
    np.testing.assert_allclose(result["normalized_gap"], [0, 0, np.log(2), np.log(2)])


def report_fixture(tmp_path):
    train = events("train", [10, 11, 10, 10], [0, 9, 1, 1], normal=True)
    attack = events("attack", [10, 10, 10], [8, 9, 8], labels=[1, 0, 1])
    normal = events("normal", [10] * 4, [0, 1, 0, 1], role="val_query", normal=True)
    weak = events(
        "weak",
        [10, 11, 10, 10, 10, 10],
        [0, 9, 1, 0, 2, 7],
        role="val_query",
        labels=[1, 0, 1, 1, -1, 1],
        lengths=[1, 1, 0, 1, 1, 1],
    )
    dm = SimpleNamespace(
        event_recordings={r.recording.key: r for r in (train, attack, normal, weak)}
    )
    for arm in ARMS:
        writer = EventScores(tmp_path / arm)
        for record in (normal, weak):
            score = (
                np.array([0.1, 0.2, 0.3, 0.4])
                if record is normal
                else np.array([0.05, 0.2, 0.15, 0.3, 0.99, 0.8])
            )
            payload = score if record is normal else np.array([0.9, 1, 100, 2, 0.99, 4])
            values = np.column_stack(
                (
                    record.arrays["times"],
                    record.arrays["ids"],
                    record.arrays["labels"],
                    score * (0.9 if arm == "balanced" else 1),
                    payload,
                    record.arrays["lengths"] > 0,
                )
            )
            for lo, hi in ((0, 2), (2, record.count)):
                writer.write(
                    values[lo:hi],
                    recording=record.recording.key,
                    source_digest=record.digest,
                    row_start=lo,
                )
        writer.finish(dict(normal=4, weak=6))
    return dm


def test_full_report_counts_unknown_and_unsupported_and_does_not_fit_on_focus(tmp_path):
    dm = report_fixture(tmp_path)
    statistics = np.array([[1, 0, 1, 10]] * 3)
    report = diagnose(dm, tmp_path, dict(weak=["attack"]), np.array([10]), statistics)
    assert report["coverage"] == dict(normal=4, weak=6)
    group = report["focus"]["weak"]
    assert group["matched_training_positives"] == 2
    assert group["audit"]["unknown"] == 1
    assert group["metrics"]["original"]["frames"] == 5
    assert group["metrics"]["payload"]["frames"] == 4
    assert group["metrics"]["original_on_payload_support"]["frames"] == 4
    assert group["at_normal_budgets"]["payload"][0]["recall"] == 0.75
    assert group["at_normal_budgets"]["payload"][0]["unsupported_positives"] == 1
    assert group["strata"]["positive"]["rows"] == 4
    assert group["strata"]["unknown"]["rows"] == 1
    for strata in group["disagreement"].values():
        for counts in strata.values():
            assert (
                sum(
                    counts[k]
                    for k in ("both", "classifier_only", "payload_only", "neither")
                )
                == counts["rows"]
            )
    assert "Recall at empirical normal-frame budgets" in markdown(report)
    dm.event_recordings["weak"].arrays["labels"][0] = 0
    with pytest.raises(ValueError, match="timestamps/IDs/labels"):
        diagnose(dm, tmp_path, dict(weak=[]), np.array([10]), statistics)


@pytest.mark.parametrize("column", [3, 4])
def test_report_rejects_nonfinite_or_changed_paired_payload_scores(tmp_path, column):
    dm = report_fixture(tmp_path)
    values = np.memmap(
        tmp_path / "balanced/scores.bin", dtype="<f8", mode="r+", shape=(10, 6)
    )
    values[0, column] = float("nan") if column == 3 else 999
    values.flush()
    with pytest.raises(ValueError, match="nonfinite|payload changed"):
        read_scores(tmp_path, dm.event_recordings, {"weak", "normal"})
