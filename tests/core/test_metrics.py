import pytest

from graphids.core.metrics import frame_metrics, raw_average_precision


def test_frame_metrics_match_independent_threshold_and_pairwise_definitions():
    """Ranking refactors must preserve AP endpoints and half-credit AUC ties."""
    import random

    rng = random.Random(42)
    cases = [
        [],
        [(1.0, -1)],
        [(0.0, 1)],
        [(0.5, 0), (0.5, 1)],
        [(2.0, 1), (2.0, 0), (1.0, 1), (20.0, -1)],
        [(1.0, 0), (2.0, -1)],
    ]
    cases += [
        [(rng.choice([-2.0, 0.0, 0.5, 3.0]), rng.choice([-1, 0, 1])) for _ in range(20)]
        for _ in range(40)
    ]
    for rows in cases:
        scores = [score for score, _ in rows]
        labels = [label for _, label in rows]
        known = [(score, label) for score, label in rows if label >= 0]
        positive = [score for score, label in known if label == 1]
        negative = [score for score, label in known if label == 0]
        actual = frame_metrics(scores, labels)
        assert actual["frames"] == len(known)
        if not positive or not negative:
            assert "frame_ap" not in actual and "frame_auroc" not in actual
            assert raw_average_precision(scores, labels) is None
            continue
        # Precision at each positive's score threshold gives tied AP endpoints.
        ap = sum(
            sum(label for value, label in known if value >= score)
            / sum(value >= score for value, _ in known)
            for score in positive
        ) / len(positive)
        auc = sum(float(p > n) + 0.5 * (p == n) for p in positive for n in negative) / (
            len(positive) * len(negative)
        )
        assert actual["frame_ap"] == pytest.approx(ap)
        assert actual["frame_auroc"] == pytest.approx(auc)
        assert raw_average_precision(scores, labels) == actual["frame_ap"]


def test_unknown_labels_and_no_predictions_do_not_invent_precision():
    metrics = frame_metrics([0.1, 0.2, 1.0], [0, 1, -1], threshold=0.5)
    assert metrics["frame_auroc"] == 1
    assert metrics["recall"] == metrics["f1"] == 0
    assert metrics["precision"] is None
    assert metrics["frames"] == 2
