from __future__ import annotations

import pytest

from can_discovery.relationships.shared.time_join import following_lags, preceding_lags


def test_strict_time_joins_and_horizon():
    assert preceding_lags([0, .04, .1, .2], [0, .02, .1], .08) == pytest.approx([.02, .08])
    assert following_lags([0, .1], [0, .02, .04, .11, .2], .02, .1) == pytest.approx([.04, .1])
    assert preceding_lags([], [0], .1) == []
    assert following_lags([0], [], 0, .1) == []
