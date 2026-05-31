"""GAT sequence pooling for snapshot_sequence representation."""

from __future__ import annotations

import pytest
import torch
from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS
from torch_geometric.data import Batch, Data


def _sequence_graph(sequence_id: int, *, steps: int = 3, nodes_per_step: int = 4) -> Data:
    num_nodes = steps * nodes_per_step
    x = torch.rand(num_nodes, IN_CHANNELS)
    node_id = torch.randint(0, NUM_IDS, (num_nodes,))
    node_sequence_step = torch.arange(steps).repeat_interleave(nodes_per_step)
    edge_parts: list[torch.Tensor] = []
    for step in range(steps):
        start = step * nodes_per_step
        src = torch.arange(start, start + nodes_per_step - 1)
        dst = torch.arange(start + 1, start + nodes_per_step)
        edge_parts.append(torch.stack([src, dst]))
    edge_index = torch.cat(edge_parts, dim=1)
    edge_attr = torch.rand(edge_index.shape[1], EDGE_DIM)
    edge_sequence_step = node_sequence_step[edge_index[0]]
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_id=node_id,
        y=torch.tensor([sequence_id % 2]),
        node_sequence_step=node_sequence_step,
        node_sequence_length=torch.full((num_nodes,), steps),
        node_sequence_stride=torch.ones(num_nodes, dtype=torch.long),
        node_snapshot_wid=node_sequence_step,
        edge_sequence_step=edge_sequence_step,
        edge_sequence_length=torch.full((edge_index.shape[1],), steps),
        edge_sequence_stride=torch.ones(edge_index.shape[1], dtype=torch.long),
        edge_snapshot_wid=edge_sequence_step,
        sequence_id=torch.tensor([sequence_id]),
        sequence_length=torch.tensor([steps]),
        sequence_stride=torch.tensor([1]),
    )


def _batch() -> Batch:
    return Batch.from_data_list([_sequence_graph(0), _sequence_graph(1)])


def _model(sequence_pool: str):
    from graphids.core.models.supervised.gat import GAT

    return GAT(
        sequence_pool=sequence_pool,
        num_ids=NUM_IDS,
        in_channels=IN_CHANNELS,
        num_classes=2,
        hidden=8,
        layers=1,
        heads=2,
        fc_layers=1,
        embedding_dim=4,
        dropout=0.0,
        edge_dim=EDGE_DIM,
        gradient_checkpointing=False,
    )


@pytest.mark.parametrize("sequence_pool", ["auto", "flat", "mean", "attention", "gru"])
def test_gat_sequence_pooling_forward_shape(sequence_pool):
    model = _model(sequence_pool)
    out = model(_batch())
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_gat_sequence_pooling_return_embedding_shape():
    model = _model("gru")
    logits, emb = model(_batch(), return_embedding=True)
    assert logits.shape == (2, 2)
    assert emb.shape[0] == 2
