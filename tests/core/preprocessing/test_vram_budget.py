"""Tests for node budget (budget.py).

Tests the public API: node_budget() and _probe().
Cost model math is tested by injecting known probe values and checking
the resulting budget, not by calling deleted intermediate functions.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from graphids.core.preprocessing.budget import (
    _FALLBACK_BYTES_PER_NODE,
    _SAFETY_MARGIN,
    BudgetResult,
    _probe,
    node_budget,
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


# --- _probe tests (GPU only) ------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_returns_positive_values():
    model = _DummyModel().cuda()
    dataset = _make_dataset()
    bpn, bpe, bwd_mult, gamma, alpha, beta = _probe(model, dataset)
    assert isinstance(bpn, int) and bpn > 0
    assert isinstance(bpe, int) and bpe >= 0
    assert bwd_mult >= 1.0, "backward multiplier must be >= 1"
    assert gamma > 0, "collation rate must be positive"
    assert beta >= 0, "per-node GPU cost must be non-negative"
    assert alpha >= 0, "GPU overhead must be non-negative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_restores_training_state():
    model = _DummyModel().cuda()
    model.train()
    _probe(model, dataset=_make_dataset())
    assert model.training, "_probe must restore model.training state"


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
    assert result.cg_ratio is None
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
    assert result.cg_ratio is None


# --- throughput budget tests (mock _probe to inject coefficients) ------------

def _mock_probe_factory(bytes_per_node, gamma, alpha, beta,
                        bytes_per_edge=0, backward_multiplier=2.0):
    """Return a mock _probe that returns fixed values (6-tuple)."""
    def _mock_probe(model, dataset, step_fn=None):
        return bytes_per_node, bytes_per_edge, backward_multiplier, gamma, alpha, beta
    return _mock_probe


def _run_with_probe(tmp_path, gamma, alpha, beta, num_workers, mean_nodes=28.2,
                    bytes_per_node=2000, free_gb=16):
    """Run node_budget with mocked probe returning specific coefficients."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes}}}}}}}')

    mock = _mock_probe_factory(bytes_per_node, gamma, alpha, beta)
    free = int(free_gb * 1024**3)

    with (
        patch("graphids.core.preprocessing.budget.cache_dir", return_value=tmp_path),
        patch("graphids.core.preprocessing.budget._probe", mock),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(free, free)),
    ):
        # Pass a truthy model and dataset to trigger the probe path
        return node_budget("test", str(tmp_path), model=True, train_dataset=True,
                           num_workers=num_workers)


def test_throughput_budget_exists_when_collation_dominated(tmp_path):
    """When γ/W > β·m̄ and α > 0, a finite throughput budget exists."""
    # γ=73μs/graph, β=0.15μs/node, α=3ms, W=2, m̄=28.2
    # gap = 73e-6/2 - 0.15e-6*28.2 = 36.5e-6 - 4.23e-6 = 32.27e-6 > 0 ✓
    # B = 0.003 / 32.27e-6 ≈ 93 graphs → 93 * 28.2 ≈ 2622 nodes
    result = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.003, beta=0.15e-6,
                             num_workers=2)
    assert result.throughput_budget is not None
    assert result.throughput_budget > 0
    assert result.cg_ratio is not None and result.cg_ratio > 1.0


def test_throughput_budget_none_when_compute_dominated(tmp_path):
    """When GPU is slower per graph (β·m̄ > γ/W), no throughput ceiling."""
    # γ=73μs/graph, β=5μs/node, W=6, m̄=28.2
    # gap = 73e-6/6 - 5e-6*28.2 = 12.2e-6 - 141e-6 < 0 → None
    result = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.003, beta=5e-6,
                             num_workers=6)
    assert result.throughput_budget is None
    assert result.binding == "memory"


def test_throughput_budget_scales_with_workers(tmp_path):
    """More workers → larger throughput budget (pipeline delivers faster)."""
    r2 = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.003, beta=0.15e-6,
                          num_workers=2)
    r6 = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.003, beta=0.15e-6,
                          num_workers=6)
    assert r2.throughput_budget is not None and r6.throughput_budget is not None
    assert r6.throughput_budget > r2.throughput_budget


def test_throughput_budget_none_when_no_overhead(tmp_path):
    """With α=0, T_gpu is pure linear — no finite optimum."""
    result = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.0, beta=0.15e-6,
                             num_workers=2)
    assert result.throughput_budget is None


def test_cg_ratio_reflects_worker_count(tmp_path):
    """More workers reduces cg_ratio (collation gets faster per worker)."""
    r2 = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.003, beta=0.15e-6,
                          num_workers=2)
    r8 = _run_with_probe(tmp_path, gamma=73e-6, alpha=0.003, beta=0.15e-6,
                          num_workers=8)
    assert r2.cg_ratio > r8.cg_ratio
