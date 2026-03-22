"""Smoke test: synthetic data → forward pass → output shape check.

Catches import breakage and basic shape mismatches on CPU in <5s.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Batch, Data

from graphids.config import resolve


def _make_graphs(n_graphs: int = 3, n_nodes: int = 5, in_channels: int = 31, edge_dim: int = 12):
    """Create synthetic PyG Data objects mimicking CANBusDataset output."""
    graphs = []
    for i in range(n_graphs):
        num_ids = 10
        x = torch.rand(n_nodes, in_channels)
        x[:, 0] = torch.randint(0, num_ids, (n_nodes,))  # CAN ID column
        # Simple chain graph: 0→1→2→3→4 (bidirectional)
        src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.rand(len(src), edge_dim)
        y = torch.tensor(i % 2, dtype=torch.long)  # binary label
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs


def test_gat_forward_pass():
    """GAT small: synthetic batch → forward → [n_graphs, 2] logits."""
    graphs = _make_graphs(edge_dim=12)  # match config default edge_dim
    batch = Batch.from_data_list(graphs)

    cfg = resolve("model_type=gat", "scale=small", "lake_root=/tmp", "device=cpu")
    with open_dict(cfg):
        cfg.num_ids = 10
        cfg.in_channels = 31
        cfg.training.gradient_checkpointing = False

    from graphids.pipeline.stages.modules import GATModule

    module = GATModule(cfg)
    module.eval()
    with torch.no_grad():
        logits = module(batch)

    assert logits.shape == (3, 2), f"Expected (3, 2), got {logits.shape}"


def test_vgae_forward_pass():
    """VGAE small: synthetic batch → forward → reconstructions + latent."""
    graphs = _make_graphs(edge_dim=12)
    batch = Batch.from_data_list(graphs)

    cfg = resolve("model_type=vgae", "scale=small", "lake_root=/tmp", "device=cpu")
    with open_dict(cfg):
        cfg.num_ids = 10
        cfg.in_channels = 31
        cfg.training.gradient_checkpointing = False

    from graphids.pipeline.stages.modules import VGAEModule

    module = VGAEModule(cfg)
    module.eval()
    with torch.no_grad():
        cont, canid, nbr, z, kl, mask = module(batch)

    n_nodes = batch.x.size(0)  # 3 graphs × 5 nodes = 15
    latent_dim = cfg.vgae.latent_dim
    assert cont.shape == (n_nodes, 30), f"Cont shape {cont.shape}"  # in_channels - 1
    assert canid.shape == (n_nodes, 10), f"CAN ID shape {canid.shape}"  # num_ids
    assert z.shape == (n_nodes, latent_dim), f"Latent shape {z.shape}"
    assert mask is None  # eval mode → no masking
