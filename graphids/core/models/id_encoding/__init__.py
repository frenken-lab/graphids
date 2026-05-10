"""Pluggable identity-encoding strategies for graph nodes.

An ``IdEncoder`` maps a ``node_id`` LongTensor to per-node embedding
vectors. Subclasses implement different strategies (lookup table,
k-probe hash, ...) behind a uniform interface so VGAE / GAT / DGI do
not know which strategy is in use.

Research basis: ``~/plans/oov-embedding-handling.md``.
"""

from .base import IdEncoder, build_encoder
from .config import (
    HashEncodingCfg,
    IdEncodingCfg,
    IdEncodingPlan,
    LookupEncodingCfg,
    build_id_encoder,
    encoding_kind,
    encoding_plan,
    hash_encoding,
    lookup_encoding,
)
from .hash_embedding import HashIdEncoder
from .lookup import UNK_INDEX, LookupIdEncoder

__all__ = [
    "HashIdEncoder",
    "HashEncodingCfg",
    "IdEncoder",
    "IdEncodingCfg",
    "IdEncodingPlan",
    "LookupEncodingCfg",
    "LookupIdEncoder",
    "UNK_INDEX",
    "build_encoder",
    "build_id_encoder",
    "encoding_kind",
    "encoding_plan",
    "hash_encoding",
    "lookup_encoding",
]
