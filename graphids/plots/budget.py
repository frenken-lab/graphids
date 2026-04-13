"""Budget plots from probe-budget JSONL sidecar — direct measurements only.

Usage:
    python -m graphids.plots.budget --jsonl path/to/budget_probe_*.jsonl
    python -m graphids.plots.budget --jsonl ... --gpu V100_16GB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import altair as alt
import polars as pl

from graphids.config.constants import PROJECT_ROOT
from graphids.plots.transforms import (
    SAFETY_MARGIN, budget_for_gpu, load_gpus, load_probe_jsonl,
)

_DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "budget"


def _save(chart: alt.TopLevelMixin, path: Path) -> None:
    chart.save(str(path), format="png", scale_factor=2)
    print(f"  wrote {path}")


def _combo(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("model_type") + "/" + pl.col("scale") + "/" + pl.col("conv_type")).alias("combo")
    )


def plot_bytes_per_node(df: pl.DataFrame, out_dir: Path) -> None:
    """Bar chart: bytes_per_node per (combo × dataset). Direct measurement."""
    chart = alt.Chart(_combo(df)).mark_bar().encode(
        x=alt.X("combo:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("bytes_per_node:Q", title="VRAM bytes per node"),
        color=alt.Color("dataset:N"),
        xOffset="dataset:N",
        tooltip=["combo", "dataset", "bytes_per_node", "bytes_per_edge"],
    ).properties(width=600, height=300, title="bytes_per_node — measured by torch.profiler")
    _save(chart, out_dir / "bytes_per_node.png")


def plot_backward_multiplier(df: pl.DataFrame, out_dir: Path) -> None:
    """Heatmap: bwd/fwd VRAM ratio per (combo × dataset)."""
    chart = alt.Chart(_combo(df)).mark_rect().encode(
        x=alt.X("dataset:N"),
        y=alt.Y("combo:N"),
        color=alt.Color("backward_multiplier:Q",
                        scale=alt.Scale(scheme="viridis", domain=[1, 4], clamp=True)),
        tooltip=["combo", "dataset", "backward_multiplier"],
    ).properties(width=300, height=300,
                 title="backward_multiplier (bwd_peak / fwd_peak)")
    text = alt.Chart(_combo(df)).mark_text(fontSize=10).encode(
        x="dataset:N", y="combo:N",
        text=alt.Text("backward_multiplier:Q", format=".2f"),
        color=alt.condition(alt.datum.backward_multiplier > 2.5,
                            alt.value("white"), alt.value("black")),
    )
    _save(chart + text, out_dir / "backward_multiplier.png")


def plot_fixed_overhead(df: pl.DataFrame, out_dir: Path) -> None:
    """Bars per combo: teacher param bytes (KD only — zero otherwise)."""
    kd = df.filter(pl.col("fixed_overhead") > 0)
    if kd.is_empty():
        print("  skipping fixed_overhead — no KD rows in this probe")
        return
    chart = alt.Chart(_combo(kd)).mark_bar(color="#d62728").encode(
        x=alt.X("combo:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("fixed_overhead:Q", title="Teacher params (bytes)"),
        tooltip=["combo", "dataset", "fixed_overhead"],
    ).properties(width=600, height=300,
                 title="fixed_overhead — KD teacher VRAM reserved before sizing")
    _save(chart, out_dir / "fixed_overhead.png")


def plot_budget_across_gpus(df: pl.DataFrame, gpus: dict[str, int], out_dir: Path) -> None:
    """Per (combo × dataset): node budget on each GPU. Recomputes from bpn + fixed."""
    rows = []
    for r in df.iter_rows(named=True):
        for gn, fb in gpus.items():
            rows.append({
                "combo": f"{r['model_type']}/{r['scale']}/{r['conv_type']}",
                "dataset": r["dataset"], "gpu": gn,
                "budget_nodes": budget_for_gpu(
                    int(r["bytes_per_node"]), int(r["fixed_overhead"]), fb),
            })
    bdf = pl.DataFrame(rows)
    chart = alt.Chart(bdf).mark_bar().encode(
        x=alt.X("combo:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("budget_nodes:Q", title="Max nodes per batch",
                 scale=alt.Scale(type="log")),
        color=alt.Color("gpu:N"),
        xOffset="gpu:N",
        tooltip=["combo", "dataset", "gpu", "budget_nodes"],
    ).facet(row="dataset:N").properties(
        title=f"node budget across GPUs (safety={SAFETY_MARGIN})",
    )
    _save(chart, out_dir / "budget_across_gpus.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Budget plots from probe-budget JSONL sidecar.")
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="budget_probe_*.jsonl from probe-budget")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU name from clusters.json (default: first)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if not args.jsonl.exists():
        print(f"ERROR: {args.jsonl} not found", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_probe_jsonl(args.jsonl)
    gpus, _, _ = load_gpus(args.gpu)

    print(f"Loaded {len(df)} probe rows from {args.jsonl}")
    plot_bytes_per_node(df, args.out)
    plot_backward_multiplier(df, args.out)
    plot_fixed_overhead(df, args.out)
    plot_budget_across_gpus(df, gpus, args.out)
    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
