"""Property-based tests for node_budget() with mocked VRAM probes.

These tests exercise *invariants* (monotonicity, memory-boundedness, sqrt
scaling for GPS) without re-implementing the formula from budget.py. Any
refactor to the budget math that preserves these properties leaves the
tests green.

The per-model VRAM probe values that used to live here were deliberately
removed — they were hardware measurements from a specific date and served
only as static fixtures for property tests, not as behavioral guarantees.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from graphids.config.constants import PROJECT_ROOT
from graphids.core.data.budget import node_budget

# Real dataset statistics (from cache_metadata.json). Only used to vary the
# *input* to the budget function; tests never assert on the specific numbers.
DATASETS = {
    "hcrl_ch": (21.9, 24),
    "hcrl_sa": (36.5, 63),
    "set_01":  (28.2, 35),
}

# Generic probe archetypes — NOT measured values. Small / medium / large
# represent scale bands so monotonicity tests have >=2 probe magnitudes.
PROBE_ARCHETYPES = {
    "small":  (20_000, 1.5),
    "medium": (60_000, 1.4),
    "large":  (200_000, 1.5),
}

_clusters = json.loads(
    (PROJECT_ROOT / "configs" / "resources" / "clusters.json").read_text()
)
GPU_TYPES = {
    name: int(spec["free_gb"] * 1024**3)
    for name, spec in _clusters["gpu_vram"].items()
}


def _run(tmp_path, *, dataset, probe, gpu, conv_type="gatv2"):
    """Run node_budget with a mocked VRAM probe."""
    mean_nodes, p95_nodes = DATASETS[dataset]
    bpn, bwd_mult = PROBE_ARCHETYPES[probe]
    free = GPU_TYPES[gpu]

    edge_mean = mean_nodes * 4.5
    edge_p95 = p95_nodes * 4.5
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(
        f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes},"p95":{p95_nodes}}},'
        f'"edge_count":{{"mean":{edge_mean},"p95":{edge_p95}}}}}}}'
    )

    def mock_probe_vram(model, ds, step_fn=None):
        return bpn, bwd_mult

    with (
        patch("graphids.core.data.budget.cache_dir", return_value=tmp_path),
        patch("graphids.core.data.budget._probe_vram", mock_probe_vram),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(free, free)),
    ):
        return node_budget(
            dataset, str(tmp_path), conv_type=conv_type,
            model=True, train_dataset=True,
        )


# ---------------------------------------------------------------------------
# Invariants (no formula-mirroring)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", list(DATASETS))
@pytest.mark.parametrize("probe", list(PROBE_ARCHETYPES))
@pytest.mark.parametrize("gpu", list(GPU_TYPES))
def test_budget_is_positive_and_memory_bound(tmp_path, dataset, probe, gpu):
    """Budget is >=1, binding is 'memory', and budget equals the memory ceiling."""
    result = _run(tmp_path, dataset=dataset, probe=probe, gpu=gpu)
    assert result.budget >= 1
    assert result.binding == "memory"
    assert result.budget == result.mem_budget


# ---------------------------------------------------------------------------
# Monotonicity properties — independent of the specific formula
# ---------------------------------------------------------------------------


def test_larger_gpu_gives_larger_or_equal_budget(tmp_path):
    """More VRAM ⇒ budget does not decrease."""
    budgets = {
        gpu: _run(tmp_path, dataset="set_01", probe="medium", gpu=gpu).budget
        for gpu in ("v100_16gb", "a100_40gb", "a100_80gb")
    }
    assert budgets["a100_40gb"] >= budgets["v100_16gb"]
    assert budgets["a100_80gb"] >= budgets["a100_40gb"]


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_larger_probe_gives_smaller_or_equal_mem_budget(tmp_path, dataset):
    """More bytes/node ⇒ mem_budget does not increase."""
    small = _run(tmp_path, dataset=dataset, probe="small", gpu="v100_16gb").mem_budget
    medium = _run(tmp_path, dataset=dataset, probe="medium", gpu="v100_16gb").mem_budget
    large = _run(tmp_path, dataset=dataset, probe="large", gpu="v100_16gb").mem_budget
    assert medium <= small
    assert large <= medium


# ---------------------------------------------------------------------------
# GPS quadratic path (no probe — uses closed-form sqrt scaling)
# ---------------------------------------------------------------------------


def test_gps_budget_scales_monotonically_with_vram(tmp_path):
    """Ranking of GPU sizes is preserved in the GPS quadratic budget."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":28.2}}}')

    budgets = {}
    for gpu, free in sorted(GPU_TYPES.items(), key=lambda kv: kv[1]):
        with (
            patch("graphids.core.data.budget.cache_dir", return_value=tmp_path),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free, free)),
        ):
            budgets[gpu] = node_budget(
                "set_01", str(tmp_path), conv_type="gps", heads=4, model=None,
            ).budget

    sizes = sorted(GPU_TYPES.items(), key=lambda kv: kv[1])
    for (gpu_a, _), (gpu_b, _) in zip(sizes, sizes[1:]):
        assert budgets[gpu_b] >= budgets[gpu_a], (
            f"GPS budget not monotonic in VRAM: {gpu_a}={budgets[gpu_a]}, "
            f"{gpu_b}={budgets[gpu_b]}"
        )


# ---------------------------------------------------------------------------
# Fallback path (no model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_fallback_binding_when_no_model(tmp_path, dataset):
    """Without a model, node_budget reports binding='fallback' and budget > 0."""
    mean_nodes, _ = DATASETS[dataset]
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes}}}}}}}')

    with patch("graphids.core.data.budget.cache_dir", return_value=tmp_path):
        result = node_budget(dataset, str(tmp_path), model=None)

    assert result.binding == "fallback"
    assert result.budget > 0
