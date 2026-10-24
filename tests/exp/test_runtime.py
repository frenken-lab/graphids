"""Construction errors and metric conversion at the Lightning boundary."""

import pytest

from graphids.exp.runtime import _scalar_metrics, build_component, resolve_spec


def test_component_builder_errors_propagate_without_retry(monkeypatch):
    """A TypeError inside a builder must keep its original identity and traceback."""
    from graphids import primitives

    error = TypeError("bug inside builder")
    calls = []

    class BrokenConfig:
        def build(self):
            calls.append("build")
            raise error

    monkeypatch.setattr(primitives, "can_event_dm", lambda **kwargs: BrokenConfig())
    with pytest.raises(TypeError) as caught:
        build_component({"type": "can_event_dm"})
    assert caught.value is error
    assert calls == ["build"]
    with pytest.raises(ValueError, match="unknown primitive"):
        resolve_spec({"type": "missing_primitive"})


def test_scalar_metrics_preserve_scalars_and_ignore_vectors():
    """Offline ingestion receives numeric scalars, without hiding conversion errors."""
    import numpy as np
    import torch

    metrics = {
        "loss": torch.tensor(0.25, requires_grad=True),
        "count": 3,
        "numpy": np.float32(0.5),
        "vector": torch.ones(2),
        "empty": torch.empty(0),
        "metadata": "unscored",
    }
    assert _scalar_metrics(metrics) == {"loss": 0.25, "count": 3.0, "numpy": 0.5}
