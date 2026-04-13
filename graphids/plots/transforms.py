"""Direct-measurement transforms for budget plots. No fitted models."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from graphids.config.constants import PROJECT_ROOT

SAFETY_MARGIN = 0.85

_REQUIRED = ("model_type", "scale", "conv_type", "dataset",
             "bytes_per_node", "backward_multiplier", "mean_nodes",
             "fixed_overhead", "bytes_per_edge")


def load_probe_jsonl(path: Path) -> pl.DataFrame:
    """Read probe-budget sidecar — one row per (model, scale, conv, dataset).

    Drops error rows (rows missing ``bytes_per_node``) and warns on the count.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "bytes_per_node" in r and r["bytes_per_node"]]
    if len(ok) < len(rows):
        print(f"[warn] dropped {len(rows) - len(ok)} probe error rows")
    if not ok:
        raise ValueError(f"No usable probe rows in {path}")
    df = pl.DataFrame(ok)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Probe sidecar missing required columns: {missing}")
    return df


def load_gpus(select: str | None = None) -> tuple[dict[str, int], str, int]:
    """Read GPU VRAM from configs/resources/clusters.json, optionally select one."""
    vram = json.loads(
        (PROJECT_ROOT / "configs" / "resources" / "clusters.json").read_text()
    )["gpu_vram"]
    gpus = {n.replace("_", " ").upper(): int(v["free_gb"] * 1024**3)
            for n, v in vram.items()}
    label = select.replace("_", " ").upper() if select else next(iter(gpus))
    if label not in gpus:
        raise KeyError(f"GPU '{select}' not in clusters.json. Available: {list(gpus)}")
    return gpus, label, gpus[label]


def budget_for_gpu(bytes_per_node: int, fixed_overhead: int, free_bytes: int) -> int:
    """Per-combo VRAM budget on a target GPU. Mirrors BudgetProfiler.node_budget."""
    return max(1, int((free_bytes - fixed_overhead) * SAFETY_MARGIN / max(1, bytes_per_node)))
