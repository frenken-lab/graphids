"""Reusable input-masking transforms for masked-input training regimes.

A masker is an :class:`nn.Module` that takes ``(x, node_id)`` and returns
``(x_masked, node_id_masked, mask)``. State (the mask token, the reserved
vocab slot) is owned by the masker, not by the model — any model that
wants masked-input training (autoencoder masked-recon, MAE-style
classifier, masked-attribute distillation) composes a masker as a
submodule.

Decoupling the masker from the model:
- removes the ``self.mask_token`` / ``self.mask_id`` cluster from the
  model class — fewer attributes the trainer-bridge has to know about,
- makes mask-rate / mask-token-init a research knob without touching
  model code,
- lets a future masked-GAT or masked-DGI reuse the exact same
  augmentation without copy-paste.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RandomNodeMasker(nn.Module):
    """Random-node masking transform.

    Replaces a random subset of node features with a learned (frozen)
    mask token and routes their IDs to a reserved mask-vocabulary slot.
    The encoder is responsible for sizing its ID-embedding table to
    ``num_ids + 1`` so ``mask_id == num_ids`` indexes the reserved slot.

    Forward returns the (possibly cloned) ``x`` and ``node_id`` plus a
    boolean ``mask`` aligned with ``x.shape[0]`` so downstream loss /
    score components can weight masked nodes specifically.
    """

    def __init__(self, *, in_channels: int, mask_id: int, mask_rate: float = 0.15):
        super().__init__()
        # Stored as a buffer (not Parameter): zero-initialized, never optimized,
        # but shows up in state_dict for ckpt round-trip. The original VGAE
        # used ``nn.Parameter(..., requires_grad=False)`` for the same effect;
        # buffer is the honest spelling.
        self.register_buffer("mask_token", torch.zeros(in_channels))
        self.mask_id = int(mask_id)
        self.mask_rate = float(mask_rate)

    def forward(
        self, x: torch.Tensor, node_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = x.size(0)
        mask = torch.rand(n, device=x.device) < self.mask_rate
        x = x.clone()
        node_id = node_id.clone()
        x[mask] = self.mask_token
        node_id[mask] = self.mask_id
        return x, node_id, mask
