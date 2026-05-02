"""Property-based tests for node_budget() — formula-free paths only.

Covers the two non-probe paths, both of which require
``GRAPHIDS_ALLOW_FALLBACK_BUDGET=1`` to opt into a conservative
hardcoded budget instead of a RuntimeError:

- GPS conv: closed-form sqrt scaling from free VRAM.
- No-model fallback: emits binding='fallback' with a default bpn.

The probe path itself isn't exercised here — it needs CUDA and a live
model; live probe correctness is checked by the end-to-end fit on set_01
plus the ``VRAMDriftCallback`` guard at epoch 1.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graphids.core.data.budget import node_budget

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


@pytest.fixture
def allow_fallback(monkeypatch):
    monkeypatch.setenv("GRAPHIDS_ALLOW_FALLBACK_BUDGET", "1")


def test_gps_budget_scales_monotonically_with_vram(allow_fallback):
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
def test_fallback_binding_when_no_model(dataset, allow_fallback):
    """Without a model, node_budget reports binding='fallback' and budget > 0."""
    result = node_budget(dataset, model=None)
    assert result.binding == "opted_in_fallback"
    assert result.budget > 0
    # Dual-budget invariant: edge_budget must be set even on the fallback path,
    # or pack_offline raises downstream when edge_sizes is passed.
    assert result.edge_budget is not None and result.edge_budget > 0
    # target_bytes must be non-zero so post-hoc utilization analysis has a
    # denominator (logged as param ``graphids.budget_target_bytes`` at epoch 0
    # by :class:`graphids._mlflow.MLflowTrainingCallback._stamp_run_config`,
    # paired with metric ``graphids.peak_vram_mb`` at fit-end).
    assert result.target_bytes > 0


def test_fallback_raises_without_env_opt_in():
    """Prereq missing + no env opt-in → loud RuntimeError. Silent fallbacks
    produce under-utilized runs that look successful but waste GPU memory.
    """
    with pytest.raises(RuntimeError, match="GRAPHIDS_ALLOW_FALLBACK_BUDGET"):
        node_budget("set_01", model=None)
