"""Vocabulary scan, digest, persist, and load primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from graphids._fs import atomic_write_text

UNK_INDEX = 0


def scan_arb_ids(raw_dir: Path, source_dirs: list[str]) -> list[Any]:
    """Sorted unique ``arb_id`` across every CSV under ``source_dirs``."""
    if not source_dirs:
        raise ValueError("source_dirs is empty; cannot scan for arb_ids")
    frames: list[pl.LazyFrame] = []
    for sub in source_dirs:
        sub_path = raw_dir / sub
        if not sub_path.is_dir():
            raise FileNotFoundError(f"Source dir missing: {sub_path}")
        for csv_path in sorted(sub_path.rglob("*.csv")):
            lf = pl.scan_csv(csv_path)
            cols = lf.collect_schema().names()
            col = "arbitration_id" if "arbitration_id" in cols else "arb_id"
            if col not in cols:
                raise ValueError(
                    f"{csv_path} has neither arbitration_id nor arb_id; got {cols!r}"
                )
            frames.append(lf.select(pl.col(col).alias("arb_id")))
    if not frames:
        raise ValueError(f"No CSVs under {source_dirs!r} in {raw_dir}")
    return pl.concat(frames).collect()["arb_id"].unique().sort().to_list()


def vocab_digest(vocab: dict[Any, int]) -> str:
    """SHA256 over ``(id, index)`` pairs sorted by index."""
    canon = json.dumps(
        sorted(((str(k), v) for k, v in vocab.items()), key=lambda kv: kv[1]),
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode()).hexdigest()


def persist_vocab(vocab: dict[Any, int], path: Path) -> str:
    """Atomic write and return the digest."""
    digest = vocab_digest(vocab)
    payload = {
        "digest": digest,
        "unk_index": UNK_INDEX,
        "entries": {str(k): v for k, v in vocab.items()},
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return digest


def load_vocab(path: Path) -> tuple[dict[str, int], str]:
    """Return ``(entries, digest)`` from a persisted vocab file."""
    payload = json.loads(path.read_text())
    return payload["entries"], payload["digest"]
