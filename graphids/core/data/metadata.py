"""Cache metadata — v2 schema, per-split merge writer, and validator.

One ``cache_metadata.json`` per dataset cache directory. Per-split
accounting lives under ``splits[<name>]``. Every writer holds an
exclusive flock on ``{cache_dir}/.metadata_lock`` so train + multiple
test-subdir processes can contribute without torn writes. See plan
``~/plans/graphids-preprocessing-metadata.md`` §3 for field semantics.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METADATA_SCHEMA_VERSION = 2

INVARIANT_KEYS = ("preprocessing_version", "window_size", "stride", "val_fraction", "seed")


def load_metadata(cache_dir: Path | str) -> dict[str, Any]:
    """Read + version-gate a v2 cache_metadata.json.

    Raises ``FileNotFoundError`` if missing; ``ValueError`` if the schema
    is not v2 (old caches must be rebuilt, not migrated).
    """
    path = Path(cache_dir) / "cache_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"cache_metadata.json missing at {path}; run rebuild-caches")
    meta = json.loads(path.read_text())
    version = meta.get("metadata_schema_version")
    if version != METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"cache_metadata.json at {path} has schema version "
            f"{version!r}, expected {METADATA_SCHEMA_VERSION}. "
            "Run `scripts/slurm/submit.sh rebuild-caches --delete-existing --yes`."
        )
    return meta


def validate_metadata(
    meta: dict[str, Any],
    *,
    dataset: str,
    test_subdirs: list[str] | None = None,
    preprocessing_version: str | None = None,
) -> list[str]:
    """Return a list of validation error strings. Empty list = valid.

    Optional kwargs cross-check the metadata against the catalog
    (``test_subdirs``) and the running code
    (``preprocessing_version``). Omit them to run the standalone checks
    only.
    """
    errors: list[str] = []
    if meta.get("metadata_schema_version") != METADATA_SCHEMA_VERSION:
        errors.append(
            f"metadata_schema_version={meta.get('metadata_schema_version')!r}; "
            f"expected {METADATA_SCHEMA_VERSION}"
        )
    if meta.get("dataset") != dataset:
        errors.append(f"dataset={meta.get('dataset')!r}; expected {dataset!r}")
    if preprocessing_version and meta.get("preprocessing_version") != preprocessing_version:
        errors.append(
            f"preprocessing_version={meta.get('preprocessing_version')!r}; "
            f"expected {preprocessing_version!r}"
        )
    splits = meta.get("splits") or {}
    if "train" not in splits:
        errors.append("splits missing 'train' entry")
    for name, entry in splits.items():
        if name == "val":
            continue
        gs = entry.get("graph_stats") or {}
        nc = gs.get("node_count") or {}
        if nc.get("min", 1) <= 0:
            errors.append(f"splits[{name!r}].graph_stats.node_count.min <= 0")
    if test_subdirs:
        present = set(splits)
        missing = [sd for sd in test_subdirs if f"test_{sd}" not in present and sd not in present]
        if missing:
            errors.append(f"test splits missing for subdirs: {missing}")
    agg = meta.get("aggregate") or {}
    sum_train = splits.get("train", {}).get("num_graphs", 0)
    sum_val = splits.get("val", {}).get("num_graphs", 0)
    sum_test = sum(
        e.get("num_graphs", 0) for name, e in splits.items() if name not in ("train", "val")
    )
    if agg.get("num_train_graphs") != sum_train:
        errors.append(
            f"aggregate.num_train_graphs={agg.get('num_train_graphs')} != "
            f"splits.train.num_graphs={sum_train}"
        )
    if agg.get("num_val_graphs") != sum_val:
        errors.append(
            f"aggregate.num_val_graphs={agg.get('num_val_graphs')} != "
            f"splits.val.num_graphs={sum_val}"
        )
    if agg.get("num_test_graphs") != sum_test:
        errors.append(
            f"aggregate.num_test_graphs={agg.get('num_test_graphs')} != sum(test splits)={sum_test}"
        )
    return errors


def merge_split_into_metadata(
    cache_dir: Path | str,
    split_name: str,
    split_entry: dict[str, Any],
    *,
    invariants: dict[str, Any],
    dataset_name: str,
    num_arb_ids: int,
) -> dict[str, Any]:
    """Merge a single split's entry into ``cache_metadata.json`` atomically.

    NFS-safe protocol: exclusive flock on ``.metadata_lock`` → read →
    validate invariants → merge → atomic tmpfile + rename. Returns the
    merged metadata dict (after write) for caller introspection /
    logging.

    The first writer seeds top-level fields (``dataset``,
    ``metadata_schema_version``, ``built_at``, ``num_arb_ids``, every key
    in ``INVARIANT_KEYS``). Later writers must match those or raise —
    mismatched window/stride/val_fraction means the cache is inconsistent
    and needs a clean rebuild, not silent overwrite.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".metadata_lock"
    meta_path = cache_dir / "cache_metadata.json"

    missing = [k for k in INVARIANT_KEYS if k not in invariants]
    if missing:
        raise ValueError(f"invariants missing required keys: {missing}")

    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            existing: dict[str, Any] = {}
            if meta_path.exists():
                existing = json.loads(meta_path.read_text())
                existing_version = existing.get("metadata_schema_version")
                if existing_version not in (None, METADATA_SCHEMA_VERSION):
                    raise ValueError(
                        f"{meta_path} has schema version {existing_version!r}; "
                        f"expected {METADATA_SCHEMA_VERSION}. "
                        "Remove the file or run rebuild-caches --delete-existing."
                    )
                for k in INVARIANT_KEYS:
                    if k in existing and existing[k] != invariants[k]:
                        raise ValueError(
                            f"{meta_path} invariant mismatch: existing "
                            f"{k}={existing[k]!r}, writer {k}={invariants[k]!r}. "
                            "Run rebuild-caches --delete-existing."
                        )
                if existing.get("dataset") not in (None, dataset_name):
                    raise ValueError(
                        f"{meta_path} dataset={existing.get('dataset')!r}; "
                        f"writer dataset={dataset_name!r}"
                    )

            meta: dict[str, Any] = {
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "dataset": dataset_name,
                "built_at": existing.get("built_at") or datetime.now(UTC).isoformat(),
                "num_arb_ids": num_arb_ids,
                **{k: invariants[k] for k in INVARIANT_KEYS},
                "splits": dict(existing.get("splits") or {}),
            }
            meta["splits"][split_name] = split_entry
            meta["aggregate"] = _aggregate(meta["splits"])

            fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(meta, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                fd = -1
                os.rename(tmp, meta_path)
            except BaseException:
                if fd >= 0:
                    os.close(fd)
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            return meta
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _aggregate(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recompute aggregate totals from current splits."""
    total_raw = 0
    total_graphs = 0
    total_bytes = 0
    n_train = splits.get("train", {}).get("num_graphs", 0)
    n_val = splits.get("val", {}).get("num_graphs", 0)
    n_test = 0
    for name, entry in splits.items():
        total_raw += int(entry.get("num_raw_samples", 0) or 0)
        total_bytes += int(entry.get("bytes_on_disk", 0) or 0)
        if name in ("train", "val"):
            continue
        n_test += int(entry.get("num_graphs", 0) or 0)
    total_graphs = n_train + n_val + n_test
    return {
        "num_raw_samples": total_raw,
        "num_graphs": total_graphs,
        "num_train_graphs": n_train,
        "num_val_graphs": n_val,
        "num_test_graphs": n_test,
        "bytes_on_disk": total_bytes,
    }
