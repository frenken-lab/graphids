"""Dense lookup embedding with optional stochastic UNK-drop.

Default (``p_unk_drop=0.0``) reproduces the pre-refactor ``nn.Embedding``
behavior byte-for-byte so existing single-vocab runs are a no-op change.

``p_unk_drop > 0.0`` implements the Stage 3 ablation arm from
``~/plans/oov-embedding-handling.md``: during training, each node_id is
remapped to ``UNK_INDEX`` with probability ``p``, so the OOV row
receives gradient and attack-introduced IDs at inference land in a
trained slot instead of init noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import IdEncoder

# Row reserved for out-of-vocabulary IDs. The shared-vocab helper
# (Stage 1) routes unknown arb_ids here; Stage 3 trains this row via
# stochastic UNK-drop. Keeping it at 0 matches the existing
# ``replace_strict(vocab, default=0)`` in ``can_bus.py``.
UNK_INDEX = 0


class LookupIdEncoder(IdEncoder):
    def __init__(
        self,
        num_ids: int,
        embedding_dim: int,
        *,
        p_unk_drop: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= p_unk_drop <= 1.0:
            raise ValueError(f"p_unk_drop must be in [0, 1], got {p_unk_drop}")
        self.embedding = nn.Embedding(num_ids, embedding_dim)
        self.out_dim = embedding_dim
        self.num_ids = num_ids
        self.p_unk_drop = p_unk_drop

    def forward(self, node_id: Tensor) -> Tensor:
        if self.training and self.p_unk_drop > 0.0:
            mask = torch.rand_like(node_id, dtype=torch.float32) < self.p_unk_drop
            node_id = node_id.masked_fill(mask, UNK_INDEX)
        return self.embedding(node_id)

    @classmethod
    def from_vocab_size(
        cls,
        num_ids: int,
        *,
        embedding_dim: int,
        p_unk_drop: float = 0.0,
    ) -> LookupIdEncoder:
        return cls(num_ids=num_ids, embedding_dim=embedding_dim, p_unk_drop=p_unk_drop)
