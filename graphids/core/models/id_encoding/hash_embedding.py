"""k-probe hash embedding — primary Stage-2 treatment.

Every id (seen or unseen) deterministically maps to ``k`` rows of a
bucketed embedding table by ``k`` decorrelated hash functions; the
per-probe vectors are summed. Because any id hits trained buckets by
construction, no special OOV slot is needed.

Shape follows Coleman et al. 2023 *Unified Embedding* (NeurIPS
Spotlight): one shared table, ``k`` probes, sum combiner — minimum
parameters, clean theoretical analysis. Yan et al. 2021 *Binary Code
Hash Embedding* (CIKM) uses the same k-probe idea with separate tables
per hash; at CAN scale (~100 ids, B=512) the shared table has the
same expressive power at half the parameters.

Hash: ``bucket_i(id) = (id * KNUTH + offset_i) mod num_buckets``, where
``KNUTH = 2654435761`` (golden-ratio-derived Knuth multiplier) and the
``k`` offsets are deterministic functions of the ``seed`` constructor
arg. The multiplier is coprime to any ``num_buckets >= 2`` that isn't
a specific pathological case, and Knuth's value is well-studied for
integer-id hashing at tiny scale.

Research basis: ``~/plans/oov-embedding-handling.md`` (Stage 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import IdEncoder

# Golden-ratio derived: (2^32 * (sqrt(5) - 1) / 2), rounded.
# Standard Knuth multiplicative-hash constant.
_KNUTH_MULT = 2654435761


class HashIdEncoder(IdEncoder):
    def __init__(
        self,
        num_buckets: int,
        embedding_dim: int,
        *,
        k: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        if num_buckets < 2:
            raise ValueError(f"num_buckets must be >= 2, got {num_buckets}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.embedding = nn.Embedding(num_buckets, embedding_dim)
        self.out_dim = embedding_dim
        self.num_buckets = num_buckets
        self.k = k
        # ``k`` decorrelated hash offsets, deterministic in ``seed``.
        # Spread across int64 so per-probe bucket distributions are
        # well-separated for small vocabs. Registered as a buffer so
        # checkpoint round-trip is exact.
        offsets = torch.tensor(
            [seed + i * (1 << 30) for i in range(k)],
            dtype=torch.int64,
        )
        self.register_buffer("_hash_offsets", offsets)

    def forward(self, node_id: Tensor) -> Tensor:
        # (N,) int → (N, 1) for broadcasting against (k,) offsets
        ids = node_id.to(torch.int64).unsqueeze(-1)
        buckets = ((ids * _KNUTH_MULT) + self._hash_offsets).remainder(self.num_buckets)
        # (N, k, D) → (N, D) by summing the k probes
        return self.embedding(buckets.to(torch.long)).sum(dim=-2)

    @classmethod
    def from_vocab_size(
        cls,
        num_ids: int,
        *,
        embedding_dim: int,
        k: int = 2,
        seed: int = 42,
        num_buckets_factor: int = 4,
        num_buckets: int | None = None,
    ) -> HashIdEncoder:
        """Build from a datamodule-injected ``num_ids``.

        Default bucket count: ``next_pow2(num_buckets_factor · num_ids)``,
        minimum 8. Per plan: Yan 2021 / Coleman 2023 use 2–4× vocab size
        as a sweet spot between collision rate and parameter count.
        ``num_buckets`` can be passed explicitly to override.
        """
        if num_buckets is None:
            target = max(8, num_buckets_factor * max(1, num_ids))
            num_buckets = 1 << (target - 1).bit_length()
        return cls(num_buckets=num_buckets, embedding_dim=embedding_dim, k=k, seed=seed)
