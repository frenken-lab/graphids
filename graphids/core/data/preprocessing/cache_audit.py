"""Metadata-only leakage audit for built graph caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphids.core.data.preprocessing.metadata import load_metadata

_AUDIT_ZERO_KEYS = (
    "graph_index_overlap",
    "base_unit_overlap",
    "raw_interval_intersections",
)


def audit_cache_metadata(cache_root: str | Path) -> dict[str, Any]:
    """Summarize split-leakage metadata for a built cache root."""

    root = Path(cache_root)
    meta = load_metadata(root)
    splits = meta.get("splits") or {}
    train = splits.get("train") or {}
    val = splits.get("val") or {}
    audit = train.get("split_audit") or val.get("split_audit") or {}

    errors: list[str] = []
    warnings: list[str] = []
    for name in ("train", "val"):
        if name not in splits:
            errors.append(f"missing split {name!r}")
        elif int((splits.get(name) or {}).get("num_graphs", 0) or 0) <= 0:
            errors.append(f"split {name!r} has no graphs")
        else:
            label_balance = (splits.get(name) or {}).get("label_balance") or {}
            if label_balance and len([v for v in label_balance.values() if int(v) > 0]) < 2:
                errors.append(f"split {name!r} has single-class labels: {label_balance}")

    train_balance = train.get("label_balance") or {}
    val_balance = val.get("label_balance") or {}
    total_pos = int(train_balance.get("1", 0) or 0) + int(val_balance.get("1", 0) or 0)
    val_pos = int(val_balance.get("1", 0) or 0)
    if total_pos:
        min_val_pos = max(1, int(total_pos * 0.05))
        if val_pos < min_val_pos:
            errors.append(f"val positives={val_pos}; expected at least {min_val_pos}")
    if not audit:
        errors.append("missing split_audit; rebuild cache with split-plan metadata")
    for key in _AUDIT_ZERO_KEYS:
        value = int(audit.get(key, -1))
        if value != 0:
            errors.append(f"split_audit.{key}={value}; expected 0")

    source_boundary_violations = int(audit.get("source_boundary_violations", 0) or 0)
    if source_boundary_violations:
        warnings.append(
            f"{source_boundary_violations} source-boundary-crossing graph(s) were detected"
        )

    vocab_scope = meta.get("vocab_scope")
    if vocab_scope == "all":
        warnings.append('vocab_scope="all" includes test dirs in preprocessing vocabulary')

    train_sources = set(train.get("source_dirs") or [])
    for split_name, entry in splits.items():
        if split_name in {"train", "val"}:
            continue
        overlap = train_sources & set(entry.get("source_dirs") or [])
        if overlap:
            errors.append(f"{split_name} source_dirs overlap train: {sorted(overlap)}")

    return {
        "cache_root": str(root),
        "dataset": meta.get("dataset"),
        "representation_kind": meta.get("representation_kind"),
        "split_policy": meta.get("split_policy"),
        "split_unit": meta.get("split_unit"),
        "split_embargo": meta.get("split_embargo"),
        "split_plan_digest": meta.get("split_plan_digest"),
        "vocab_scope": vocab_scope,
        "num_train_graphs": (meta.get("aggregate") or {}).get("num_train_graphs"),
        "num_val_graphs": (meta.get("aggregate") or {}).get("num_val_graphs"),
        "audit": audit,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }
