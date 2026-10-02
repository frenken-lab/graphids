"""Deterministic development holdouts over content-grouped CAN recordings."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import polars as pl


def recording_manifest(
    raw: Path,
    entry: dict,
    *,
    fraction: float,
    seed: int,
    warmup: int,
    train_source_mode: str,
) -> dict:
    """Use training directories only; never select individual attack windows.

    Exact duplicate files have one canonical recording and one assignment.
    Fraction is a target per family in recording counts, not event counts.
    Single-recording families stay in training unless a pooled fallback is needed.
    """
    from graphids.core.data.datasets.can_bus import infer_attack_type

    if not 0 < fraction < 1:
        raise ValueError("recording validation requires 0 < val_fraction < 1")
    groups: dict[str, list[Path]] = defaultdict(list)
    for directory in dict.fromkeys(
        filter(None, (entry.get("train_subdir"), entry.get("train_attack_subdir")))
    ):
        path = raw / directory
        if not path.is_dir():
            raise FileNotFoundError(
                f"declared source_dir {directory!r} missing under {raw}"
            )
        for csv in sorted(path.rglob("*.csv")):
            with csv.open("rb") as handle:
                groups[hashlib.file_digest(handle, "sha256").hexdigest()].append(csv)

    records = []
    for digest, files in sorted(groups.items()):
        # Prefer the attack-free source for copies present in both sources.
        files.sort(
            key=lambda p: (
                p.relative_to(raw).parts[0] != entry.get("train_subdir"),
                str(p),
            )
        )
        canonical = files[0]
        frame = pl.scan_csv(canonical)
        label = (
            (pl.col("attack").fill_null(0) > 0)
            if "attack" in frame.collect_schema().names()
            else pl.lit(False)
        )
        labels = frame.select("timestamp", label.alias("positive")).sort(
            "timestamp", maintain_order=True
        )
        counts = (
            labels.select(
                pl.len().alias("events"), pl.col("positive").sum().alias("attacks")
            )
            .collect()
            .row(0)
        )
        scored = (
            labels.slice(warmup)
            .select(pl.len().alias("events"), pl.col("positive").sum().alias("attacks"))
            .collect()
            .row(0)
        )
        records.append(
            {
                "sha256": digest,
                "source_file": str(canonical.relative_to(raw)),
                "aliases": [str(p.relative_to(raw)) for p in files],
                "events": int(counts[0]),
                "attacks": int(counts[1]),
                "scored_benign": int(scored[0] - scored[1]),
                "scored_attacks": int(scored[1]),
                "family": infer_attack_type(canonical) if counts[1] else 0,
                "split": "train",
            }
        )

    def order(record):
        return hashlib.sha256(f"{seed}:{record['sha256']}".encode()).hexdigest()

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        strata[(bool(record["attacks"]), record["family"])].append(record)
    for stratum in strata.values():
        eligible = sorted(
            (
                r
                for r in stratum
                if r["scored_attacks"] or (not r["attacks"] and r["scored_benign"])
            ),
            key=order,
        )
        count = min(max(0, len(stratum) - 1), math.ceil(len(stratum) * fraction))
        for record in eligible[:count]:
            record["split"] = "val"

    # Families with a single recording can still support a pooled holdout,
    # provided an attack-bearing training recording remains.
    if not any(r["split"] == "val" and r["scored_attacks"] for r in records):
        attacked = sorted((r for r in records if r["attacks"]), key=order)
        eligible = [r for r in attacked if r["scored_attacks"]]
        if len(attacked) >= 2 and eligible:
            eligible[0]["split"] = "val"
    if not any(r["split"] == "val" and r["scored_benign"] for r in records):
        benign = sorted(
            (r for r in records if not r["attacks"] and r["scored_benign"]), key=order
        )
        if len(benign) >= 2:
            benign[0]["split"] = "val"

    if train_source_mode == "attack_free":
        for record in records:
            if record["split"] == "train" and (
                record["attacks"]
                or Path(record["source_file"]).parts[0] != entry.get("train_subdir")
            ):
                record["split"] = "excluded"
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if (
        not train
        or not val
        or not sum(r["scored_attacks"] for r in val)
        or not sum(r["scored_benign"] for r in val)
    ):
        raise ValueError(
            "cannot create attack-inclusive recording validation with both scored classes and nonempty training; provide more independent training recordings"
        )
    if train_source_mode == "mixed" and not sum(r["attacks"] for r in train):
        raise ValueError("recording holdout leaves no attacked training recording")
    payload = {
        "version": 1,
        "policy": "recordings",
        "split_seed": seed,
        "val_fraction": fraction,
        "val_warmup_events": warmup,
        "train_source_mode": train_source_mode,
        "recordings": records,
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload
