"""Property-based tests for node_budget() — formula-free paths only.

The 40 tests that patched ``budget._probe_vram`` were deleted 2026-04-13
after ``_probe_vram`` ceased to exist in the production module. The
remaining tests exercise the GPS quadratic path and the fallback-when-
no-model path — both of which work without the vanished private probe.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from graphids.core.data.budget import node_budget

# Real dataset statistics (from cache_metadata.json). Only used to vary the
# *input* to the budget function; tests never assert on the specific numbers.
DATASETS = {
    "hcrl_ch": (21.9, 24),
    "hcrl_sa": (36.5, 63),
    "set_01": (28.2, 35),
}

# Free VRAM per GPU model, bytes. Covers the cluster inventory we deploy against
# (V100 on Pitzer, A100 on Ascend, H100 on Cardinal). Monotonicity fixture only —
# the test asserts ordering, not specific budget numbers.
GPU_TYPES: dict[str, int] = {
    "v100_16gb": 14 * 1024**3,
    "a100_40gb": 36 * 1024**3,
    "a100_80gb": 76 * 1024**3,
    "h100_94gb": 90 * 1024**3,
}


# ---------------------------------------------------------------------------
# GPS quadratic path (no probe — uses closed-form sqrt scaling)
# ---------------------------------------------------------------------------


def _write_metadata(path, mean_nodes: float) -> None:
    """Minimal v2-shape metadata — only fields node_budget reads."""
    path.write_text(
        json.dumps(
            {
                "metadata_schema_version": 2,
                "splits": {
                    "train": {
                        "graph_stats": {
                            "node_count": {"mean": mean_nodes},
                        },
                    },
                },
            }
        )
    )


def test_gps_budget_scales_monotonically_with_vram(tmp_path):
    """Ranking of GPU sizes is preserved in the GPS quadratic budget."""
    _write_metadata(tmp_path / "cache_metadata.json", 28.2)

    budgets = {}
    for gpu, free in sorted(GPU_TYPES.items(), key=lambda kv: kv[1]):
        with (
            patch("graphids.core.data.budget.cache_dir", return_value=tmp_path),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free, free)),
        ):
            budgets[gpu] = node_budget(
                "set_01",
                str(tmp_path),
                conv_type="gps",
                heads=4,
                model=None,
            ).budget

    sizes = sorted(GPU_TYPES.items(), key=lambda kv: kv[1])
    for (gpu_a, _), (gpu_b, _) in zip(sizes, sizes[1:]):
        assert budgets[gpu_b] >= budgets[gpu_a], (
            f"GPS budget not monotonic in VRAM: {gpu_a}={budgets[gpu_a]}, {gpu_b}={budgets[gpu_b]}"
        )


# ---------------------------------------------------------------------------
# Fallback path (no model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_fallback_binding_when_no_model(tmp_path, dataset):
    """Without a model, node_budget reports binding='fallback' and budget > 0."""
    mean_nodes, _ = DATASETS[dataset]
    _write_metadata(tmp_path / "cache_metadata.json", mean_nodes)

    with patch("graphids.core.data.budget.cache_dir", return_value=tmp_path):
        result = node_budget(dataset, str(tmp_path), model=None)

    assert result.binding == "fallback"
    assert result.budget > 0
