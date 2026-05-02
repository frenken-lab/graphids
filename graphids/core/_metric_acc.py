"""Dict-based batch-weighted metric accumulator.

Lives in :mod:`graphids.core` (not :mod:`graphids.core.trainer`) because
:mod:`graphids.core.models.base` imports it — keeping it here avoids a
``models → trainer`` import cycle.

Why not ``torchmetrics.MeanMetric`` per key: every ``MeanMetric`` is an
``nn.Module`` with state on a CUDA device. Computing ``.compute().item()``
per metric per batch (which the trainer does at log-throttle intervals)
forces a host/device sync per metric per call. The dict-based version
accumulates Python floats — zero sync, zero device traffic.
"""

from __future__ import annotations

import math


class MetricAccumulator:
    """Dynamic-keyed batch-weighted mean.

    Plain ``dict[str, (sum, count)]`` — NOT an ``nn.Module``. Storing it
    in a ``ModuleDict`` would pollute the parent's ``state_dict`` and
    reject keys with ``"."`` (``add_module`` attribute-name check),
    breaking metric names like ``"test/precision@0.95recall"``.

    NaN detection hard-fails the run — under ``precision: 16-mixed`` a
    silent NaN in ``callback_metrics`` fools ``EarlyStopping``
    (``NaN < inf`` is False) and wastes the full patience window.
    """

    def __init__(self, nan_strategy: str = "error") -> None:
        self._nan_strategy = nan_strategy
        self._sums: dict[str, float] = {}
        self._counts: dict[str, float] = {}

    def update(self, name: str, value: float, batch_size: int = 1) -> None:
        v = float(value)
        if math.isnan(v):
            if self._nan_strategy == "error":
                raise ValueError(f"NaN encountered in metric {name!r}")
            return
        self._sums[name] = self._sums.get(name, 0.0) + v * batch_size
        self._counts[name] = self._counts.get(name, 0.0) + batch_size

    def compute(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts.get(k)}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
