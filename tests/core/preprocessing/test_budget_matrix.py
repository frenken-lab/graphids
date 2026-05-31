"""Property-based tests for node_budget() budget-selection paths.

Covers auto selection and the non-probe path:

- GPS conv: closed-form sqrt scaling from free VRAM.
- No-model path: emits binding='heuristic' with node and edge budgets.
- With CUDA + model + train dataset, auto mode routes to the empirical probe.

The empirical probe path itself isn't exercised here — it needs CUDA and a
live model; live probe correctness is checked by GPU smoke tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graphids.core.budget import BudgetResult, node_budget

DATASETS = ["hcrl_ch", "hcrl_sa", "set_01"]

# Free VRAM per GPU model, bytes. Covers the cluster inventory we deploy against
# (V100 on Pitzer, A100 on Ascend, H100 on Cardinal). Monotonicity fixture only —
# the test asserts ordering, not specific budget numbers.
GPU_TYPES: dict[str, int] = {
    "v100_16gb": 14 * 1024**3,
    "a100_40gb": 36 * 1024**3,
    "a100_80gb": 76 * 1024**3,
    "h100_94gb": 90 * 1024**3,
}


def test_gps_budget_scales_monotonically_with_vram():
    """Ranking of GPU sizes is preserved in the GPS quadratic budget."""
    budgets = {}
    for gpu, free in sorted(GPU_TYPES.items(), key=lambda kv: kv[1]):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free, free)),
        ):
            budgets[gpu] = node_budget("set_01", conv_type="gps", heads=4, model=None).budget

    sizes = sorted(GPU_TYPES.items(), key=lambda kv: kv[1])
    for (gpu_a, _), (gpu_b, _) in zip(sizes, sizes[1:]):
        assert budgets[gpu_b] >= budgets[gpu_a], (
            f"GPS budget not monotonic in VRAM: {gpu_a}={budgets[gpu_a]}, {gpu_b}={budgets[gpu_b]}"
        )


@pytest.mark.parametrize("dataset", DATASETS)
def test_fallback_binding_when_no_model(dataset):
    """Without a model, node_budget still returns a usable heuristic budget."""
    result = node_budget(dataset, model=None)
    assert result.binding == "heuristic"
    assert result.budget > 0
    # Dual-budget invariant: edge_budget must be set even on the fallback path,
    # or pack_offline raises downstream when edge_sizes is passed.
    assert result.edge_budget is not None and result.edge_budget > 0
    # target_bytes must be non-zero so post-hoc utilization analysis has a
    # denominator for tracking and Ray-side telemetry.
    assert result.target_bytes > 0


def test_strict_probe_mode_raises_without_probe_prereqs(monkeypatch):
    """Strict probe mode is still available for calibration/debugging."""
    monkeypatch.setenv("GRAPHIDS_BUDGET_MODE", "probe")
    monkeypatch.setenv("GRAPHIDS_BUDGET_STRICT_PROBE", "1")
    with pytest.raises(RuntimeError, match="budget probe requires CUDA"):
        node_budget("set_01", model=None)


def test_auto_mode_prefers_probe_when_prereqs_exist(monkeypatch):
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
