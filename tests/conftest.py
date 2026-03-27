"""Shared test fixtures. All graph/config construction lives here — no duplication."""

from __future__ import annotations

import copy

import pytest
import torch
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
    """Session-scoped config namespace for tests. Clone before mutating."""
    import types
    from graphids.config.defaults.schema import (
        VGAEConfig, GATConfig, DGIConfig, TrainingConfig,
        FusionConfig, DQNConfig, BanditConfig, EvaluationConfig,
        PreprocessingConfig, TemporalConfig,
    )
    cfg = types.SimpleNamespace(
        model_type="vgae", scale="small", stage="autoencoder",
        lake_root="/tmp", dataset="test", seed=42,
        device="cpu", num_workers=0,
        num_ids=NUM_IDS, in_channels=IN_CHANNELS, num_classes=2,
        gat_stage="curriculum", auxiliaries=[],
        vgae=VGAEConfig(hidden_dims=[32, 16], latent_dim=16, heads=2, embedding_dim=4),
        gat=GATConfig(hidden=16, layers=2, heads=2, fc_layers=2, embedding_dim=4),
        dgi=DGIConfig(hidden_dims=[32, 16], latent_dim=16, heads=2, embedding_dim=4),
        training=TrainingConfig(
            max_epochs=2, precision="32", gradient_checkpointing=False,
            compile_model=False, dynamic_batching=False, log_every_n_steps=1,
            patience=100, batch_size=32,
        ),
        fusion=FusionConfig(), dqn=DQNConfig(), bandit=BanditConfig(),
        evaluation=EvaluationConfig(), temporal=TemporalConfig(),
        preprocessing=PreprocessingConfig(),
        checkpoints={},
    )
    return cfg


@pytest.fixture()
def vgae_cfg(base_cfg):
    """VGAE config (deep copy of base)."""
    return copy.deepcopy(base_cfg)


@pytest.fixture()
def gat_cfg(base_cfg):
    """GAT config derived from base."""
    cfg = copy.deepcopy(base_cfg)
    cfg.model_type = "gat"
    cfg.stage = "curriculum"
    return cfg
