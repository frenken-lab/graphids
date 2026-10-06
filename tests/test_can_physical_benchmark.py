from __future__ import annotations

from types import SimpleNamespace

from can_discovery.preprocessing.capture import Capture, Frame
from scripts.experiment_can_physical import _value_metrics


def _capture(name: str, multiplier: int) -> Capture:
    frames = tuple(Frame(float(i), i, '123', bytes((i, multiplier * i + 3)))
                   for i in range(40))
    return Capture(name, name, frames, 0, 39, 0, 0, {2: 40}, {4: 40})


def test_affine_value_equivalence_is_fitted_before_held_out_trip():
    candidate = SimpleNamespace(arb_id='123', payload_length=2, start=0,
                                width=8, endian='little', signed=False,
                                mux_start=None, mux_value=None)
    truth = SimpleNamespace(arb_id='123', payload_length=2, start=8,
                            width=8, endian='little', signed=False,
                            mux_start=None, mux_value=None)
    references = [_capture('a', 2), _capture('b', 2)]
    same = _value_metrics(candidate, truth, references, _capture('held', 2))
    changed = _value_metrics(candidate, truth, references, _capture('held', 3))
    assert same is not None and same['value_equivalent']
    assert same['held_out_mean_absolute_raw_error'] < 1e-10
    assert changed is not None and not changed['value_equivalent']
