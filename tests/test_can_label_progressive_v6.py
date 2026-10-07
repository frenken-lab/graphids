"""Progressive value classes, contrast, and gated relational evidence."""
from __future__ import annotations

import numpy as np

from can_discovery.label_completion.progressive_v6 import (
    candidate_features,
    fit_contrast,
    relation_net,
    trajectory_signature,
)


def test_affine_value_class_retains_distinct_layout_members():
    times = np.arange(128, dtype=float) * .1
    values = np.arange(128, dtype=float)
    first = [(times, values), (times, values * 2 + 10)]
    second = [(times, -3 * values + 8), (times, -6 * values + 4)]
    assert trajectory_signature(first) == trajectory_signature(second)
    shifted = [(times, values**2), (times, values**2)]
    assert trajectory_signature(first) != trajectory_signature(shifted)


def test_contrast_fits_source_positives_against_competitors():
    positive = [np.array([.9, .8, .5, .01, .2, .25, 0.]),
                np.array([.8, .7, .6, .02, .3, .25, 0.])]
    negative = [np.array([.7, .2, .1, .5, .9, .1, 0.]),
                np.array([.6, .3, .1, .4, .8, .1, 0.]),
                np.array([.8, .2, .2, .6, .7, .1, 1.]),
                np.array([.75, .25, .15, .55, .75, .1, 1.])]
    model = fit_contrast(positive, negative)
    assert model is not None
    assert min(model.adjustment(x) for x in positive) > max(
        model.adjustment(x) for x in negative)


def test_relation_requires_synchronized_change_beyond_shifted_control():
    rng = np.random.default_rng(4)
    steps = rng.normal(size=128)
    values = np.cumsum(steps)
    times = np.arange(128, dtype=float) * .1
    aligned = [(times, values)]
    anchor = [(times, values * 2)]
    assert relation_net(aligned, anchor) is not None
    assert relation_net(aligned, anchor) > .5
    assert relation_net(aligned, [(times, np.ones(128))]) is None


def test_features_are_scale_free_except_declared_unary_score():
    times = np.arange(128, dtype=float)
    values = np.arange(128, dtype=float)
    row = {'width': 16, 'signed': False}
    first = candidate_features(row, [(times, values)], .8)
    second = candidate_features(row, [(times, values * 10 + 3)], .8)
    assert first.shape == (7,)
    assert np.allclose(first[:4], second[:4])
