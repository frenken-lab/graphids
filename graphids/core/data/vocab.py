"""Shared-vocabulary construction for datasets with categorical node IDs.

Every GraphIDS dataset that uses ``nn.Embedding(num_ids, ...)`` over a
per-node categorical identity (CAN arbitration IDs, sensor names, etc.)
MUST build its vocab once across all splits (train + val + every test
subdir) and pass the result to every split at construction time.

Rationale: a per-split vocab drifts the index → physical-id mapping
across splits and leaves the model's embedding table under-sized at
test time, because test subdirs can contain attack-injected IDs absent
from train. Index 0 is reserved for UNK (out-of-vocabulary); real IDs
start at 1.

Research basis: ``~/plans/oov-embedding-handling.md`` (Stage 1).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

UNK_INDEX = 0


def scan_arb_ids(raw_dir: Path | str, source_dirs: list[str]) -> list[Any]:
    """Return sorted unique ``arb_id`` values across every CSV under every source_dir.

    Tolerates both the HCRL ``arbitration_id`` and the in-schema
    ``arb_id`` column names. Only the id column is materialized.
    """
    raw_dir = Path(raw_dir)
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
            if "arbitration_id" in cols:
                col = "arbitration_id"
            elif "arb_id" in cols:
                col = "arb_id"
            else:
                raise ValueError(
                    f"{csv_path} has neither 'arbitration_id' nor 'arb_id' column; got {cols!r}"
                )
            frames.append(lf.select(pl.col(col).alias("arb_id")))
    if not frames:
        raise ValueError(f"No CSVs found under any of {source_dirs!r} in {raw_dir}")
    combined = pl.concat(frames).collect()
    return combined["arb_id"].unique().sort().to_list()


def vocab_digest(vocab: dict[Any, int]) -> str:
    """Stable SHA256 digest over the vocab's (id, index) pairs.

    Used as a cache invariant — any vocab change forces rebuild. Sorted
    by index so the digest is insensitive to dict iteration order but
    sensitive to any (id, index) change.
    """
    canon = json.dumps(
        sorted(((str(k), v) for k, v in vocab.items()), key=lambda kv: kv[1]),
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode()).hexdigest()


def persist_vocab(vocab: dict[Any, int], path: Path | str) -> str:
    """Atomically write vocab as JSON under ``path``; return its digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = vocab_digest(vocab)
    payload = {
        "digest": digest,
        "unk_index": UNK_INDEX,
        "entries": {str(k): v for k, v in vocab.items()},
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return digest


def load_vocab(path: Path | str) -> tuple[dict[str, int], str]:
    """Read a persisted vocab; return ``(entries, digest)``.

    Keys are stringified at persist time (JSON constraint), so reloaded
    ``entries`` is always ``str → int`` even if the original in-memory
    vocab was ``int → int``. Callers that pipe the result into polars
    ``replace_strict`` against a numeric column must cast keys first,
    otherwise the match silently fails and every row routes to UNK.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    return payload["entries"], payload["digest"]
