"""Explicit ID-encoding configs and factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .hash_embedding import HashIdEncoder
from .lookup import LookupIdEncoder


@dataclass(frozen=True)
class LookupEncodingCfg:
    kind: Literal["lookup"] = "lookup"
    embedding_dim: int = 16
    p_unk_drop: float = 0.0


@dataclass(frozen=True)
class HashEncodingCfg:
    kind: Literal["hash"] = "hash"
    embedding_dim: int = 16
    k: int = 2
    seed: int = 42
    num_buckets_factor: int = 4
    num_buckets: int | None = None


IdEncodingCfg = LookupEncodingCfg | HashEncodingCfg


@dataclass(frozen=True)
class IdEncodingPlan:
    kind: Literal["lookup", "hash"]
    cfg: IdEncodingCfg


def encoding_kind(cfg: IdEncodingCfg) -> str:
    if isinstance(cfg, LookupEncodingCfg):
        return "lookup"
    if isinstance(cfg, HashEncodingCfg):
        return "hash"
    raise TypeError(f"unsupported id-encoding config: {type(cfg)!r}")


def encoding_plan(cfg: IdEncodingCfg) -> IdEncodingPlan:
    return IdEncodingPlan(kind=encoding_kind(cfg), cfg=cfg)


def lookup_encoding(*, embedding_dim: int = 16, p_unk_drop: float = 0.0) -> LookupEncodingCfg:
    return LookupEncodingCfg(embedding_dim=embedding_dim, p_unk_drop=p_unk_drop)


def hash_encoding(
    *,
    embedding_dim: int = 16,
    k: int = 2,
    seed: int = 42,
    num_buckets_factor: int = 4,
    num_buckets: int | None = None,
) -> HashEncodingCfg:
    return HashEncodingCfg(
        embedding_dim=embedding_dim,
        k=k,
        seed=seed,
        num_buckets_factor=num_buckets_factor,
        num_buckets=num_buckets,
    )


def build_id_encoder(cfg: IdEncodingCfg, *, num_ids: int):
    if isinstance(cfg, LookupEncodingCfg):
        return LookupIdEncoder(
            num_ids=num_ids,
            embedding_dim=cfg.embedding_dim,
            p_unk_drop=cfg.p_unk_drop,
        )
    if isinstance(cfg, HashEncodingCfg):
        return HashIdEncoder.from_vocab_size(
            num_ids=num_ids,
            embedding_dim=cfg.embedding_dim,
            k=cfg.k,
            seed=cfg.seed,
            num_buckets_factor=cfg.num_buckets_factor,
            num_buckets=cfg.num_buckets,
        )
    raise TypeError(f"unsupported id-encoding config: {type(cfg)!r}")
