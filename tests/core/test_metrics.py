import pytest

from graphids.core.metrics import raw_average_precision


def test_exact_frame_ap_preserves_raw_score_ties_and_unknown_labels():
    """Raw anomaly scores must retain ties and exclude unreviewed labels."""
    assert raw_average_precision([2.0, 2.0, 1.0, 20.0], [1, 0, 1, -1]) == pytest.approx(
        (0.5 + 2 / 3) / 2
    )
    assert raw_average_precision([1.0, 1.0], [0, 1]) == 0.5
    assert raw_average_precision([1.0, 2.0], [0, -1]) is None
