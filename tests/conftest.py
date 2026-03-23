"""Shared test fixtures. All graph/config construction lives here — no duplication."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Batch, Data

NUM_IDS = 10
IN_CHANNELS = 35
EDGE_DIM = 11
N_NODES = 8  # fixed default — tests must not assume random sizes


def make_graph(num_nodes: int = N_NODES, num_edges: int = 12) -> Data:
    """Synthetic CAN-bus-like graph: all continuous features + separate node_id."""
    x = torch.rand(num_nodes, IN_CHANNELS)
    node_id = torch.randint(0, NUM_IDS, (num_nodes,))
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    edge_attr = torch.rand(num_edges, EDGE_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_id=node_id, y=torch.tensor([1]))


def make_batch(n_graphs: int = 4) -> Batch:
    """Batch of fixed-size graphs. Deterministic — no random node counts."""
    return Batch.from_data_list([make_graph() for _ in range(n_graphs)])


def make_variable_batch(sizes: list[int]) -> Batch:
    """Batch of explicitly-sized graphs for variable-size tests."""
    return Batch.from_data_list([make_graph(num_nodes=n, num_edges=n * 2) for n in sizes])


@pytest.fixture(scope="session")
def base_cfg():
    """Session-scoped resolved config (vgae small, CPU). Clone before mutating."""
    from graphids.config import resolve

    cfg = resolve("model_type=vgae", "scale=small", "lake_root=/tmp", "device=cpu")
    with open_dict(cfg):
        cfg.num_ids = NUM_IDS
        cfg.in_channels = IN_CHANNELS
        cfg.num_classes = 2
        cfg.num_workers = 0
        cfg.training.max_epochs = 2
        cfg.training.precision = 32
        cfg.training.gradient_checkpointing = False
        cfg.training.compile_model = False
        cfg.training.dynamic_batching = False
        cfg.training.log_every_n_steps = 1
        cfg.training.patience = 100
    return cfg


@pytest.fixture()
def vgae_cfg(base_cfg):
    """VGAE config (deep copy of base)."""
    return OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))


@pytest.fixture()
def gat_cfg(base_cfg):
    """GAT config derived from base."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    with open_dict(cfg):
        cfg.model_type = "gat"
        cfg.stage = "curriculum"
    return cfg
