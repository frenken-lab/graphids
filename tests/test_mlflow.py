"""Tests for ``graphids._mlflow`` metric-name sanitization + idempotency lookup.

MLflow's name validator accepts only ``[A-Za-z0-9_\\-. :/]``. Operating-point
metric keys emitted by ``core/models/base.py::_log_operating_points``
(``test/precision@0.95recall`` etc.) embed ``@`` and would otherwise trip
``mlflow.log_metrics``, killing the whole test-phase row.
"""

from __future__ import annotations

import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from graphids import _mlflow
from graphids._mlflow import _scalar_metrics, is_finished


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
        "nested": {"threshold@0.95recall": 0.2},
    }
    for key in _scalar_metrics(raw):
        assert "@" not in key


# CONTRACT: non-``@`` keys pass through unchanged so existing dashboards
# / search_runs filters keep working.
def test_scalar_metrics_passthrough_for_plain_keys():
    raw = {"train_loss": 1.0, "val/auroc": 0.8, "nested": {"f1": 0.6}}
    assert _scalar_metrics(raw) == {
        "train_loss": 1.0,
        "val/auroc": 0.8,
        "nested/f1": 0.6,
    }


# REGRESSION risk: a transient MLflow lookup error (DB lock, stale
# tracking-URI, schema migration race) must not block ``submit``. The
# manifest workflow re-renders + re-bashes after fixes, so a flaky
# ``--skip-if-finished`` check that *raised* would strand the script
# mid-loop. Soft-fail to ``False`` (== "submit anyway") is the contract.
def test_is_finished_soft_fails_on_mlflow_error(monkeypatch):
    monkeypatch.setattr(_mlflow, "ensure_tracking_uri", lambda: "sqlite:///x.db")
    monkeypatch.setattr(_mlflow.mlflow, "set_tracking_uri", lambda _u: None)

    def boom(**_kw):
        raise MlflowException("simulated DB lock")

    monkeypatch.setattr(_mlflow.mlflow, "search_runs", boom)
    assert is_finished(dataset="ds", group="g", variant="v", seed=1) is False


# CONTRACT: ``is_finished`` must call ``search_runs`` with
# ``order_by=start_time DESC`` + ``max_results=1`` so the *latest* attempt
# decides skip-vs-submit. A stale FINISHED row from a prior code version
# would otherwise silently mask today's FAILED.
def test_is_finished_queries_latest_only(monkeypatch):
    monkeypatch.setattr(_mlflow, "ensure_tracking_uri", lambda: "sqlite:///x.db")
    monkeypatch.setattr(_mlflow.mlflow, "set_tracking_uri", lambda _u: None)
    captured: dict = {}

    def fake_search(**kw):
        captured.update(kw)
        return pd.DataFrame([{"status": "FINISHED"}])

    monkeypatch.setattr(_mlflow.mlflow, "search_runs", fake_search)
    assert is_finished(dataset="ds", group="g", variant="v", seed=1) is True
    assert captured["order_by"] == ["attributes.start_time DESC"]
    assert captured["max_results"] == 1


# CONTRACT: only ``FINISHED`` is a skip. ``RUNNING`` (live job) and
# ``FAILED`` (resubmit) and ``KILLED`` all return False so a re-render +
# re-bash re-submits them.
@pytest.mark.parametrize("status", ["RUNNING", "FAILED", "KILLED", "TERMINATED"])
def test_is_finished_only_FINISHED_returns_true(monkeypatch, status):
    monkeypatch.setattr(_mlflow, "ensure_tracking_uri", lambda: "sqlite:///x.db")
    monkeypatch.setattr(_mlflow.mlflow, "set_tracking_uri", lambda _u: None)
    monkeypatch.setattr(
        _mlflow.mlflow,
        "search_runs",
        lambda **_kw: pd.DataFrame([{"status": status}]),
    )
    assert is_finished(dataset="ds", group="g", variant="v", seed=1) is False
