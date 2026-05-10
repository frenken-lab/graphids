"""Cache metadata contract for dataset builds."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from graphids._fs import atomic_write_text

METADATA_SCHEMA_VERSION = 3

INVARIANT_KEYS = (
    "preprocessing_version",
    "window_size",
    "stride",
    "val_fraction",
    "seed",
    "vocab_digest",
    "scaler_strategy",
)


def load_metadata(cache_dir: Path) -> dict[str, Any]:
    """Read and version-gate ``cache_metadata.json``."""
    path = cache_dir / "cache_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"cache_metadata.json missing at {path}; run rebuild-caches")
    meta = json.loads(path.read_text())
    ver = meta.get("metadata_schema_version")
    if ver != METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema {ver!r} != expected {METADATA_SCHEMA_VERSION}; rebuild caches"
        )
    return meta


def _aggregate(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n_train = splits.get("train", {}).get("num_graphs", 0)
    n_val = splits.get("val", {}).get("num_graphs", 0)
    n_test = sum(
        e.get("num_graphs", 0) for n, e in splits.items() if n not in ("train", "val")
    )
    return {
        "num_raw_samples": sum(int(e.get("num_raw_samples", 0) or 0) for e in splits.values()),
        "num_graphs": n_train + n_val + n_test,
        "num_train_graphs": n_train,
        "num_val_graphs": n_val,
        "num_test_graphs": n_test,
        "bytes_on_disk": sum(int(e.get("bytes_on_disk", 0) or 0) for e in splits.values()),
    }


def validate_metadata(
    meta: dict[str, Any],
    *,
    dataset: str,
    test_subdirs: list[str] | None = None,
    preprocessing_version: str | None = None,
) -> list[str]:
    errs: list[str] = []

    def expect(key: str, want: Any) -> None:
        got = meta.get(key)
        if got != want:
            errs.append(f"{key}={got!r}; expected {want!r}")

    expect("metadata_schema_version", METADATA_SCHEMA_VERSION)
    expect("dataset", dataset)
    if preprocessing_version:
        expect("preprocessing_version", preprocessing_version)

    splits = meta.get("splits") or {}
    if "train" not in splits:
        errs.append("splits missing 'train' entry")
    for name, entry in splits.items():
        if name == "val":
            continue
        nc = (entry.get("graph_stats") or {}).get("node_count") or {}
        if nc.get("min", 1) <= 0:
            errs.append(f"splits[{name!r}].graph_stats.node_count.min <= 0")
    if test_subdirs:
        present = set(splits)
        missing = [
            sd for sd in test_subdirs if f"test_{sd}" not in present and sd not in present
        ]
        if missing:
            errs.append(f"test splits missing for subdirs: {missing}")

    expected = _aggregate(splits)
    actual = meta.get("aggregate") or {}
    for k in ("num_train_graphs", "num_val_graphs", "num_test_graphs"):
        if actual.get(k) != expected[k]:
            errs.append(f"aggregate.{k}={actual.get(k)} != computed {expected[k]}")
    return errs


def merge_split_into_metadata(
    cache_dir: Path,
    split_name: str,
    split_entry: dict[str, Any],
    *,
    invariants: dict[str, Any],
    dataset_name: str,
    num_arb_ids: int,
) -> dict[str, Any]:
    """Merge one split's entry into ``cache_metadata.json`` under FileLock.

    First writer seeds top-level fields; later writers must match
    invariants + dataset name or raise.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_metadata.json"

    missing = [k for k in INVARIANT_KEYS if k not in invariants]
    if missing:
        raise ValueError(f"invariants missing required keys: {missing}")

    with FileLock(str(cache_dir / ".metadata_lock")):
        existing: dict[str, Any] = {}
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            ver = existing.get("metadata_schema_version")
            if ver not in (None, METADATA_SCHEMA_VERSION):
                raise ValueError(
                    f"{meta_path} schema {ver!r} != {METADATA_SCHEMA_VERSION}; "
                    "delete or rebuild --delete-existing"
                )
            for k in INVARIANT_KEYS:
                if k in existing and existing[k] != invariants[k]:
                    raise ValueError(
                        f"{meta_path} invariant mismatch: {k}={existing[k]!r} "
                        f"!= writer {invariants[k]!r}; rebuild caches"
                    )
            if existing.get("dataset") not in (None, dataset_name):
                raise ValueError(
                    f"{meta_path} dataset={existing.get('dataset')!r} != writer {dataset_name!r}"
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
        atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True))
        return meta
