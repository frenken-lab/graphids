"""Tests for ``graphids._mlflow`` metric-name sanitization.

MLflow's name validator accepts only ``[A-Za-z0-9_\\-. :/]``. Operating-point
metric keys emitted by ``core/models/base.py::_log_operating_points``
(``test/precision@0.95recall`` etc.) embed ``@`` and would otherwise trip
``log_batch``, killing the whole test-phase row.
"""

from __future__ import annotations

from graphids._mlflow import _scalar_metrics


# REGRESSION: test-phase MLflow row failed with
# "Invalid value 'test/precision@0.95recall' for parameter 'metrics[10].name'"
# because `@` is outside MLflow's allowed metric-name alphabet.
def test_scalar_metrics_replaces_at_sign():
    out = _scalar_metrics({"test/precision@0.95recall": 0.9})
    assert out == {"test/precision_at_0.95recall": 0.9}


# INVARIANT: sanitized keys never contain characters outside MLflow's
# allowed set. Covers the mapping without mirroring the replacement formula.
def test_scalar_metrics_keys_have_no_at_sign():
    raw = {
        "precision@0.95recall": 0.5,
        "test/recall@0.99precision": 0.7,
        "threshold@0.95recall": 0.2,
    }
    for key in _scalar_metrics(raw):
        assert "@" not in key


# CONTRACT: non-``@`` keys pass through unchanged so existing dashboards
# and search_runs filters keep working.
def test_scalar_metrics_passthrough_for_plain_keys():
    raw = {"train_loss": 1.0, "val/auroc": 0.8, "f1": 0.6}
    assert _scalar_metrics(raw) == raw
