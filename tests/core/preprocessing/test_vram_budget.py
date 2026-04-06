"""Tests for node budget (budget.py).

Tests the public API: node_budget() and _probe_vram().
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from graphids.core.data.budget import (
    _FALLBACK_BYTES_PER_NODE,
    _SAFETY_MARGIN,
    BudgetResult,
    _probe_vram,
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


# --- _probe_vram tests (GPU only) -------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_vram_returns_positive_values():
    model = _DummyModel().cuda()
    dataset = _make_dataset()
    bpn, bwd_mult = _probe_vram(model, dataset)
    assert isinstance(bpn, int) and bpn > 0
    assert bwd_mult >= 1.0, "backward multiplier must be >= 1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_vram_restores_training_state():
    model = _DummyModel().cuda()
    model.train()
    _probe_vram(model, dataset=_make_dataset())
    assert model.training, "_probe_vram must restore model.training state"


# --- node_budget fallback tests (CPU, no model) ------------------------------


def test_fallback_when_no_model(tmp_path):
    """Without model, uses _FALLBACK_BYTES_PER_NODE constant."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    with patch("graphids.core.data.budget.cache_dir", return_value=tmp_path):
        result = node_budget("test", str(tmp_path), model=None)

    assert isinstance(result, BudgetResult)
    assert result.mean_nodes == 30.0
    assert result.binding == "fallback"
    expected = int(12 * 1024**3 * _SAFETY_MARGIN / _FALLBACK_BYTES_PER_NODE)
    assert result.mem_budget == expected
    assert result.budget == expected


def test_budget_scales_with_free_vram(tmp_path):
    """Doubling free VRAM should roughly double the budget."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    def run_with_free(free_bytes):
        with (
            patch("graphids.core.data.budget.cache_dir", return_value=tmp_path),
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

    with patch("graphids.core.data.budget.cache_dir", return_value=tmp_path):
        result = node_budget(
            "test", str(tmp_path), conv_type="gps", heads=4, model=None,
        )

    assert result.budget > 0
    assert result.binding == "memory"


# --- node_budget with mocked VRAM probe --------------------------------------


def _mock_probe_vram_factory(bytes_per_node, backward_multiplier=2.0):
    """Return a mock _probe_vram that returns fixed values (2-tuple)."""
    def _mock(model, dataset, step_fn=None):
        return bytes_per_node, backward_multiplier
    return _mock


def _run_with_probe(tmp_path, bytes_per_node=2000, free_gb=16, mean_nodes=28.2):
    """Run node_budget with mocked probe returning specific VRAM values."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text(f'{{"graph_stats":{{"node_count":{{"mean":{mean_nodes}}}}}}}')

    mock = _mock_probe_vram_factory(bytes_per_node)
    free = int(free_gb * 1024**3)

    with (
        patch("graphids.core.data.budget.cache_dir", return_value=tmp_path),
        patch("graphids.core.data.budget._probe_vram", mock),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(free, free)),
    ):
        return node_budget("test", str(tmp_path), model=True, train_dataset=True)


def test_budget_equals_vram_ceiling(tmp_path):
    """Budget should equal mem_budget (VRAM ceiling is the only constraint)."""
    result = _run_with_probe(tmp_path, bytes_per_node=2000, free_gb=16)
    assert result.budget == result.mem_budget
    assert result.binding == "memory"


def test_edge_aware_margin_reduces_budget_vs_balanced(tmp_path):
    """Edge-dense p95 produces a strictly smaller budget than edge-balanced p95.

    Differential test — no formula mirroring. Runs node_budget twice with the
    same probe and VRAM, varying only the p95 edge/node ratio in metadata.
    """
    mock = _mock_probe_vram_factory(2000)
    free = int(16 * 1024**3)

    def _run(edge_p95: float) -> int:
        meta = tmp_path / "cache_metadata.json"
        meta.write_text(
            '{"graph_stats":{"node_count":{"mean":28.2,"p95":35},'
            f'"edge_count":{{"mean":126.9,"p95":{edge_p95}}}}}}}'
        )
        with (
            patch("graphids.core.data.budget.cache_dir", return_value=tmp_path),
            patch("graphids.core.data.budget._probe_vram", mock),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free, free)),
        ):
            return node_budget("test", str(tmp_path), model=True, train_dataset=True).budget

    # Balanced: p95_edge/p95_node == mean_edge/mean_node == 4.5
    balanced = _run(edge_p95=35 * 4.5)
    # Dense: p95 edge/node ratio is larger → stricter margin → smaller budget
    dense = _run(edge_p95=210.0)
    assert dense < balanced, f"edge-aware margin not applied: balanced={balanced}, dense={dense}"
