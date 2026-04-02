"""Tests for throughput-aware node budget (budget.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from graphids.core.preprocessing.budget import (
    BudgetResult,
    CostCoefficients,
    _FALLBACK_BYTES_PER_NODE,
    _SAFETY_MARGIN,
    _throughput_budget_nodes,
    collation_time,
    gpu_time,
    node_budget,
    probe,
    regime,
)


def _make_dataset(n_graphs: int = 20, nodes_per_graph: int = 100) -> list[Data]:
    """Tiny dataset of uniform graphs for testing."""
    graphs = []
    for _ in range(n_graphs):
        n = nodes_per_graph
        graphs.append(Data(
            x=torch.rand(n, 35),
            edge_index=torch.stack([
                torch.randint(0, n, (n * 2,)),
                torch.randint(0, n, (n * 2,)),
            ]),
            edge_attr=torch.rand(n * 2, 11),
            node_id=torch.randint(0, 10, (n,)),
            y=torch.tensor([0]),
        ))
    return graphs


class _DummyModel(torch.nn.Module):
    """Minimal model that processes graph data for probe testing."""

    def __init__(self, in_channels: int = 35, hidden: int = 16):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, hidden)

    def forward(self, data):
        return self.lin(data.x).mean()


# --- probe tests -------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_returns_positive_bytes_and_coefficients():
    model = _DummyModel().cuda()
    dataset = _make_dataset()
    bpn, coeffs = probe(model, dataset, n_target=500, n_small=100)
    assert isinstance(bpn, int)
    assert bpn > 0
    assert isinstance(coeffs, CostCoefficients)
    assert coeffs.collate_per_graph_s > 0
    assert coeffs.gpu_per_node_s >= 0
    assert coeffs.gpu_overhead_s >= 0
    assert coeffs.probe_n_graphs > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_restores_training_state():
    model = _DummyModel().cuda()
    model.train()
    probe(model, dataset=_make_dataset(), n_target=500, n_small=100)
    assert model.training, "probe must restore model.training state"


# --- node_budget fallback tests (CPU, no model) ------------------------------


def test_fallback_when_no_model(tmp_path):
    """Without model, uses _FALLBACK_BYTES_PER_NODE constant."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    with patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path):
        result = node_budget("test", str(tmp_path), model=None)

    assert isinstance(result, BudgetResult)
    assert result.mean_nodes == 30.0
    assert result.binding == "fallback"
    assert result.throughput_budget is None
    assert result.regime == "unknown"
    expected = int(12 * 1024**3 * _SAFETY_MARGIN / _FALLBACK_BYTES_PER_NODE)
    assert result.mem_budget == expected
    assert result.budget == expected


def test_budget_scales_with_free_vram(tmp_path):
    """Doubling free VRAM should roughly double the budget."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    def run_with_free(free_bytes):
        with (
            patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free_bytes, free_bytes)),
        ):
            result = node_budget("test", str(tmp_path), model=None)
        return result.budget

    b1 = run_with_free(8 * 1024**3)
    b2 = run_with_free(16 * 1024**3)
    assert 1.8 < b2 / b1 < 2.2, f"expected ~2x ratio, got {b2/b1:.2f}"


def test_quadratic_path_for_gps(tmp_path):
    """GPS conv type should use the quadratic formula, not probe."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    with patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path):
        result = node_budget(
            "test", str(tmp_path), conv_type="gps", heads=4, model=None,
        )

    assert result.budget > 0
    assert result.binding == "memory"
    assert result.regime == "compute-dominated"


# --- cost model unit tests ---------------------------------------------------


def test_collation_time_linear():
    coeffs = CostCoefficients(
        collate_per_graph_s=0.001, gpu_per_node_s=1e-6,
        gpu_overhead_s=0.003, probe_n_graphs=100, probe_n_nodes=2000,
    )
    assert collation_time(100, coeffs) == pytest.approx(0.1)
    assert collation_time(200, coeffs) == pytest.approx(0.2)


def test_gpu_time_affine():
    coeffs = CostCoefficients(
        collate_per_graph_s=0.001, gpu_per_node_s=1e-6,
        gpu_overhead_s=0.005, probe_n_graphs=100, probe_n_nodes=2000,
    )
    # T_gpu = 0.005 + 1e-6 * N
    assert gpu_time(0, coeffs) == pytest.approx(0.005)
    assert gpu_time(10000, coeffs) == pytest.approx(0.015)
    assert gpu_time(20000, coeffs) == pytest.approx(0.025)


def test_regime_classification():
    # Collation-dominated: high collate rate, low GPU rate
    collation_heavy = CostCoefficients(
        collate_per_graph_s=73e-6, gpu_per_node_s=0.15e-6,
        gpu_overhead_s=0.003, probe_n_graphs=70, probe_n_nodes=2000,
    )
    assert regime(collation_heavy, num_workers=2, mean_nodes=28.2) == "collation-dominated"

    # Compute-dominated: many workers overwhelm collation cost
    assert regime(collation_heavy, num_workers=50, mean_nodes=28.2) == "compute-dominated"


def test_throughput_budget_exists_when_collation_dominated():
    """In collation-dominated regime with GPU overhead, a finite optimum exists."""
    coeffs = CostCoefficients(
        collate_per_graph_s=73e-6, gpu_per_node_s=0.15e-6,
        gpu_overhead_s=0.003, probe_n_graphs=70, probe_n_nodes=2000,
    )
    tb = _throughput_budget_nodes(coeffs, num_workers=2, mean_nodes=28.2)
    assert tb is not None
    assert tb > 0


def test_throughput_budget_none_when_compute_dominated():
    """In compute-dominated regime, no finite throughput budget — memory binds."""
    coeffs = CostCoefficients(
        collate_per_graph_s=73e-6, gpu_per_node_s=5e-6,  # GPU is slow
        gpu_overhead_s=0.003, probe_n_graphs=70, probe_n_nodes=2000,
    )
    tb = _throughput_budget_nodes(coeffs, num_workers=6, mean_nodes=28.2)
    assert tb is None


def test_throughput_budget_scales_with_workers():
    """More workers → larger throughput budget (pipeline delivers faster)."""
    coeffs = CostCoefficients(
        collate_per_graph_s=73e-6, gpu_per_node_s=0.15e-6,
        gpu_overhead_s=0.003, probe_n_graphs=70, probe_n_nodes=2000,
    )
    tb2 = _throughput_budget_nodes(coeffs, num_workers=2, mean_nodes=28.2)
    tb6 = _throughput_budget_nodes(coeffs, num_workers=6, mean_nodes=28.2)
    assert tb2 is not None and tb6 is not None
    assert tb6 > tb2


def test_throughput_budget_none_when_no_overhead():
    """With α=0 (pure linear model), ratio is constant — no finite optimum."""
    coeffs = CostCoefficients(
        collate_per_graph_s=73e-6, gpu_per_node_s=0.15e-6,
        gpu_overhead_s=0.0, probe_n_graphs=70, probe_n_nodes=2000,
    )
    tb = _throughput_budget_nodes(coeffs, num_workers=2, mean_nodes=28.2)
    assert tb is None
