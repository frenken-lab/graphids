"""Tests for probe-based VRAM node budget."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from graphids.core.preprocessing.datamodule import (
    _BYTES_PER_NODE,
    _SAFETY_MARGIN,
    _probe_bytes_per_node,
    vram_node_budget,
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


# --- _probe_bytes_per_node tests -------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_returns_positive_int():
    model = _DummyModel().cuda()
    dataset = _make_dataset()
    bpn = _probe_bytes_per_node(model, dataset, n_target=500)
    assert isinstance(bpn, int)
    assert bpn > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_probe_restores_training_state():
    model = _DummyModel().cuda()
    model.train()
    _probe_bytes_per_node(model, dataset=_make_dataset(), n_target=500)
    assert model.training, "probe must restore model.training state"


# --- vram_node_budget fallback tests (CPU, no model) -----------------------


def test_fallback_when_no_model(tmp_path):
    """Without model, uses _BYTES_PER_NODE constant."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    with patch("graphids.core.preprocessing.datamodule.cache_dir", return_value=tmp_path):
        budget, mean = vram_node_budget("test", str(tmp_path), model=None)

    assert mean == 30.0
    # CPU fallback: 12 GiB free
    expected = int(12 * 1024**3 * _SAFETY_MARGIN / _BYTES_PER_NODE)
    assert budget == expected


def test_budget_scales_with_free_vram(tmp_path):
    """Doubling free VRAM should roughly double the budget."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    def run_with_free(free_bytes):
        with (
            patch("graphids.core.preprocessing.datamodule.cache_dir", return_value=tmp_path),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free_bytes, free_bytes)),
        ):
            budget, _ = vram_node_budget("test", str(tmp_path), model=None)
        return budget

    b1 = run_with_free(8 * 1024**3)
    b2 = run_with_free(16 * 1024**3)
    assert 1.8 < b2 / b1 < 2.2, f"expected ~2x ratio, got {b2/b1:.2f}"


def test_quadratic_path_for_gps(tmp_path):
    """GPS conv type should use the quadratic formula, not probe."""
    metadata = tmp_path / "cache_metadata.json"
    metadata.write_text('{"graph_stats":{"node_count":{"mean":30.0}}}')

    with patch("graphids.core.preprocessing.datamodule.cache_dir", return_value=tmp_path):
        budget, _ = vram_node_budget(
            "test", str(tmp_path), conv_type="gps", heads=4, model=None,
        )

    assert budget > 0
