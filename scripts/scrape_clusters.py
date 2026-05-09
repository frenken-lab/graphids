#!/usr/bin/env python3
"""Scrape OSC cluster partition specs into configs/resources/clusters.json.

Source of truth for ``slurm/submit._PROFILES`` sizing. Re-run when partitions
change. Reads ``sinfo -M <cluster>`` for pitzer / cardinal / ascend; aggregates
identical (partition, cores, mem_mb, walltime, gres) tuples and records node
counts.

Usage:
    python scripts/scrape_clusters.py        # writes configs/resources/clusters.json
    python scripts/scrape_clusters.py --diff # show diff vs current file
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CLUSTERS = ("pitzer", "cardinal", "ascend")
OUT = Path(__file__).resolve().parent.parent / "configs" / "resources" / "clusters.json"


def _normalize_gres(g: str) -> str:
    """Drop opaque tags (`nsight:no_consume:1`, `pfsdir:scratch:...`, etc.)
    keep only `gpu:...` tokens. Empty string → no GPU.
    """
    if not g or g == "(null)":
        return ""
    # Strip socket info like `(S:2-3,6-7)` first — it contains commas that
    # would otherwise split the token.
    g = re.sub(r"\(S:[^)]*\)", "", g)
    keep = [tok for tok in g.split(",") if tok.startswith("gpu:")]
    return ",".join(keep)


def _parse_walltime(s: str) -> int | None:
    """SLURM walltime → seconds. ``infinite`` → None."""
    if s in ("infinite", "UNLIMITED"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    h, m, sec = (int(x) for x in s.split(":"))
    return days * 86400 + h * 3600 + m * 60 + sec


def scrape(cluster: str) -> list[dict]:
    out = subprocess.run(
        ["sinfo", "-M", cluster, "-h", "--format=%P|%D|%c|%m|%l|%G"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    bucket: dict[tuple, int] = defaultdict(int)
    rows: dict[tuple, dict] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 6:
            continue
        partition, nodes, cores, mem_mb, walltime, gres = parts
        partition = partition.rstrip("*")  # `batch*` default flag
        try:
            n = int(nodes)
            c = int(cores.rstrip("+"))
            m = int(mem_mb.rstrip("+"))
        except ValueError:
            continue
        if n == 0:
            continue
        gres_norm = _normalize_gres(gres)
        wt_sec = _parse_walltime(walltime)
        key = (partition, c, m, walltime, gres_norm)
        bucket[key] += n
        rows[key] = {
            "partition": partition,
            "cores_per_node": c,
            "mem_mb_per_node": m,
            "mem_gb_per_node": round(m / 1024, 1),
            "mem_gb_per_core": round(m / 1024 / c, 2),
            "walltime": walltime,
            "walltime_sec": wt_sec,
            "gres": gres_norm,
        }
    return [{**rows[k], "n_nodes": n} for k, n in sorted(bucket.items())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", action="store_true", help="Show diff vs current JSON; do not write.")
    args = ap.parse_args()

    data = {
        "_source": "sinfo -M <cluster>; ./scripts/scrape_clusters.py",
        "clusters": {c: scrape(c) for c in CLUSTERS},
    }
    new_text = json.dumps(data, indent=2) + "\n"

    if args.diff:
        if OUT.exists():
            old = OUT.read_text()
            if old == new_text:
                print(f"{OUT}: unchanged")
                return 0
            import difflib

            sys.stdout.writelines(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=str(OUT),
                    tofile="(scraped)",
                )
            )
        else:
            print(f"{OUT}: would create")
        return 0

    OUT.write_text(new_text)
    n_part = sum(len(v) for v in data["clusters"].values())
    print(f"wrote {n_part} partition entries across {len(CLUSTERS)} clusters → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
