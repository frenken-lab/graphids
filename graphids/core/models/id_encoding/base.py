"""Base class for pluggable identity encoders.

Contract (duck-typed, matching the rest of the codebase):

- ``forward(node_id: LongTensor) -> Tensor`` of shape ``(N, out_dim)``.
- ``out_dim: int`` attribute set in ``__init__``.
- All stateful policy (vocab size, hash seeds, UNK-drop rate) lives on
  the encoder instance — ``InputEncoder`` holds one and does not branch
  on its type.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class IdEncoder(nn.Module):
    """Maps per-node identities to per-node embedding vectors.

    Planned subclasses:
    - ``LookupIdEncoder`` — dense ``nn.Embedding`` over a shared vocab,
      with optional stochastic UNK-drop (Stage 3 ablation).
    - ``HashIdEncoder`` (Stage 2 primary, not yet implemented) — k-probe
      hash embedding per Yan et al. 2021 (CIKM).
    """

    out_dim: int

    def forward(self, node_id: Tensor) -> Tensor:  # pragma: no cover - interface only
        raise NotImplementedError
