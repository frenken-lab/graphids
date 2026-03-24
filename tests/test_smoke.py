"""Smoke test: imports + forward pass. Catches breakage fast."""

from __future__ import annotations

import torch
from conftest import N_NODES, make_batch


def test_gat_forward(gat_cfg):
    """GAT: synthetic batch → [n_graphs, num_classes] logits."""
    from graphids.core.models.gat import GATModule

    module = GATModule(gat_cfg)
    module.eval()
    with torch.no_grad():
        logits = module(make_batch(3))
    assert logits.shape == (3, 2)


def test_vgae_forward(vgae_cfg):
    """VGAE: synthetic batch → (cont, canid, nbr, z, kl, mask)."""
    from graphids.core.models.vgae import VGAEModule

    module = VGAEModule(vgae_cfg)
    module.eval()
    with torch.no_grad():
        out = module(make_batch(3))
    assert len(out) == 6
    assert out[0].shape[0] == 3 * N_NODES  # cont
    assert out[3].shape[0] == 3 * N_NODES  # z
