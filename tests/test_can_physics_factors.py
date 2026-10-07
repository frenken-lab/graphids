"""Controlled masked trips for physical relationship evidence."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from can_discovery.relationships.physical import physics_factors as factors


def _evaluate(monkeypatch, name, data):
    valid = np.ones_like(data, dtype=bool)
    monkeypatch.setattr(factors, 'align', lambda *args, **kwargs:
                        (data, valid, np.zeros_like(data), .1))
    return factors.evaluate_factor([SimpleNamespace()],
                                   tuple(SimpleNamespace() for _ in factors.FACTOR_ROLES[name]), name)


def test_gear_conditioned_response_beats_shifted_states(monkeypatch):
    n = 480
    t = np.arange(n)
    speed = 20 + 5 * np.sin(t / 20)
    gear = np.where((t // 80) % 2 == 0, 2., 3.)
    rpm = speed * np.where(gear == 2, 80., 50.) + 500
    found = _evaluate(monkeypatch, 'speed_rpm_gear', np.column_stack((speed, rpm, gear)))
    assert found.status == 'supported' and found.effect > .2


def test_lagged_steering_response(monkeypatch):
    t = np.arange(480)
    speed = 20 + 3 * np.sin(t / 37)
    steering = np.sin(t / 12)
    yaw = np.roll(speed * steering, 2)
    found = _evaluate(monkeypatch, 'steering_yaw',
                      np.column_stack((speed, steering, yaw)))
    assert found.status == 'supported' and found.lag_bins in (1, 2, 3, 5)


def test_redundant_speed_is_affine_equivalent(monkeypatch):
    t = np.arange(300)
    speed = 20 + 8 * np.sin(t / 15) + t / 70
    found = _evaluate(monkeypatch, 'wheel_speed',
                      np.column_stack((speed, 3 * speed + 7)))
    assert found.status == 'supported' and found.residual_quality > .99


def test_independent_trip_cannot_choose_a_new_lag(monkeypatch):
    rng = np.random.default_rng(7)
    t = np.arange(600)
    speed = 20 + 2 * np.sin(t / 23)
    steering = rng.normal(size=len(t))
    traces = {lag: np.column_stack((speed, steering,
                                   np.roll(speed * steering, lag))) for lag in (2, 5)}
    monkeypatch.setattr(factors, 'align', lambda capture, *args, **kwargs:
                        (traces[capture.source], np.ones_like(traces[capture.source], dtype=bool),
                         np.zeros_like(traces[capture.source]), .1))
    fields = (SimpleNamespace(),) * 3
    first = factors.evaluate_factor([SimpleNamespace(source=2)], fields, 'steering_yaw')
    second = factors.evaluate_factor([SimpleNamespace(source=5)], fields, 'steering_yaw')
    together = factors.evaluate_factor([SimpleNamespace(source=2), SimpleNamespace(source=5)],
                                       fields, 'steering_yaw')
    assert first.status == second.status == 'supported'
    assert first.lag_bins == 2 and second.lag_bins == 5
    assert together.status == 'failed'
    assert together.lag_bins == first.lag_bins
