"""Invariant budget-selection tests."""

from __future__ import annotations

from graphids.core.budget import BudgetResult, node_budget


def test_budget_fallback_returns_node_and_edge_limits():
    """Without a model, node_budget still returns a usable heuristic budget."""
    result = node_budget("set_01", model=None)
    assert result.binding == "heuristic"
    assert result.budget > 0
    assert result.edge_budget is not None and result.edge_budget > 0
    assert result.target_bytes > 0


def test_auto_budget_uses_probe_when_prereqs_exist(monkeypatch):
    """Default auto mode preserves the measured budget path for real training."""
    calls = {}

    class _HParams:
        conv_type = "gatv2"

    class _Model:
        hparams = _HParams()

    def fake_probe(model, train_dataset, *, quadratic, min_steps):
        calls["model"] = model
        calls["train_dataset"] = train_dataset
        calls["quadratic"] = quadratic
        calls["min_steps"] = min_steps
        return BudgetResult(budget=123, edge_budget=456, binding="measured")

    train_dataset = [object()]
    monkeypatch.delenv("GRAPHIDS_BUDGET_MODE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("graphids.core.budget.probe", fake_probe)

    result = node_budget("set_01", model=_Model(), train_dataset=train_dataset, min_steps=9)

    assert result.binding == "measured"
    assert result.budget == 123
    assert calls["train_dataset"] is train_dataset
    assert calls["quadratic"] is False
    assert calls["min_steps"] == 9
