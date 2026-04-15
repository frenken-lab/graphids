"""Property-based tests for node_budget() — formula-free paths only.

Covers the two non-probe paths:
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
    """Without a model, node_budget reports binding='fallback' and budget > 0."""
    result = node_budget(dataset, model=None)
    assert result.binding == "fallback"
    assert result.budget > 0
