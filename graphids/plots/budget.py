"""Budget cost-model plot functions (Altair / Vega-Lite).

Each function takes ModelParams + context and produces a chart.
Data loading lives in loaders.py, math in transforms.py.

Usage:
    python -m graphids.plots.budget --csv path/to/budget_calibration.csv
    python -m graphids.plots.budget --csv ... --model vgae/small
"""

import argparse
import sys
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl

from graphids.config import PROJECT_ROOT
from graphids.plots.transforms import (
    SAFETY_MARGIN, ModelParams, fit_models, load_calibration_csv, load_gpus,
)

_DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "budget"
_RULE_DASH = [6, 3]


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _save(chart: alt.TopLevelMixin, path: Path) -> None:
    """Save chart as PNG and print."""
    chart.save(str(path), format="png", scale_factor=2)
    print(f"  wrote {path}")


def _rule(axis: str, val: float, color: str, **mark_kw) -> alt.Chart:
    """Single reference line via alt.datum (no DataFrame needed)."""
    kw = {"color": color, "strokeDash": _RULE_DASH, "strokeWidth": 1.2, **mark_kw}
    return alt.Chart().mark_rule(**kw).encode(**{axis: alt.datum(val)})


# ---------------------------------------------------------------------------
# Plot 1: Throughput curves
# ---------------------------------------------------------------------------

def plot_throughput_curves(
    models: dict[str, ModelParams], num_workers: int,
    gpu_name: str, free_bytes: int, out_dir: Path,
):
    """One figure per model: throughput vs batch size with annotations."""
    for label, p in models.items():
        max_graphs = int(p.mem_budget(free_bytes) / p.mean_nodes) + 100
        B = np.linspace(1, max_graphs, 500)
        tp_knps = p.throughput(B, num_workers) / 1000
        df = pl.DataFrame({"batch_graphs": B, "throughput_knps": tp_knps})

        line = alt.Chart(df).mark_line(color="#1f77b4", strokeWidth=2).encode(
            x=alt.X("batch_graphs:Q", title="Batch size (graphs)",
                     scale=alt.Scale(domain=[0, max_graphs])),
            y=alt.Y("throughput_knps:Q", title="Throughput (k nodes/sec)",
                     scale=alt.Scale(domain=[0, float(tp_knps.max() * 1.15)])),
        )

        refs = _rule("x", p.mem_budget(free_bytes) / p.mean_nodes, "red")
        floor = p.throughput_floor(num_workers)
        if floor is not None:
            refs += _rule("x", floor / p.mean_nodes, "orange")
        cl_knps = p.collation_limit(num_workers) / 1000
        refs += _rule("y", cl_knps, "gray", opacity=0.6)
        cpl = p.compute_limit()
        if cpl is not None and cpl / 1000 < cl_knps * 5:
            refs += _rule("y", cpl / 1000, "green", opacity=0.6)

        cg = p.cg_ratio(num_workers)
        regime = "collation-bound" if cg > 1 else "compute-bound"

        chart = (line + refs).properties(
            width=600, height=300,
            title=f"{label} on {gpu_name} -- W={num_workers}, regime: {regime} (cg={cg:.2f})",
        )
        m, s = label.split("/")
        _save(chart, out_dir / f"throughput_{m}_{s}_{gpu_name.replace(' ', '_')}_w{num_workers}.png")


# ---------------------------------------------------------------------------
# Plot 2: Regime map
# ---------------------------------------------------------------------------

def plot_regime_map(models: dict[str, ModelParams], out_dir: Path):
    """Heatmap: cg_ratio for each model x worker count."""
    worker_counts = [1, 2, 4, 6, 8, 12]
    rows = [
        {"model": label, "workers": str(w), "cg_ratio": min(p.cg_ratio(w), 100)}
        for label, p in models.items() for w in worker_counts
    ]
    df = pl.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("workers:N", title="num_workers",
                 sort=[str(w) for w in worker_counts]),
        y=alt.Y("model:N", title="model"),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("cg_ratio:Q", title="cg_ratio",
                        scale=alt.Scale(scheme="redyellowgreen", reverse=True,
                                        domain=[0, 5], clamp=True)),
    )
    text = base.mark_text(fontSize=11).transform_calculate(
        label="datum.cg_ratio < 50 ? format(datum.cg_ratio, '.1f') : 'inf'"
    ).encode(
        text="label:N",
        color=alt.condition(
            alt.datum.cg_ratio > 3, alt.value("white"), alt.value("black")),
    )

    chart = (heatmap + text).properties(
        width=300, height=200,
        title="Training regime: cg_ratio (>1 = collation-bound, <1 = compute-bound)",
    )
    _save(chart, out_dir / "regime_map.png")


# ---------------------------------------------------------------------------
# Plot 3: Budget comparison across GPUs
# ---------------------------------------------------------------------------

def plot_budget_comparison(
    models: dict[str, ModelParams], gpus: dict[str, int],
    num_workers: int, out_dir: Path,
):
    """Grouped bar chart: budget per model x GPU, with floor markers."""
    df = pl.DataFrame([
        {"model": label, "gpu": gn, "budget_graphs": p.mem_budget(fb) / p.mean_nodes,
         "floor_graphs": fl / p.mean_nodes if (fl := p.throughput_floor(num_workers)) else None}
        for label, p in models.items() for gn, fb in gpus.items()
    ])

    bars = alt.Chart(df).mark_bar(opacity=0.85).encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("budget_graphs:Q", title="Budget (graphs per batch)",
                 scale=alt.Scale(type="log")),
        color=alt.Color("gpu:N", title="GPU"),
        xOffset="gpu:N",
    )
    floor_marks = alt.Chart(df.drop_nulls("floor_graphs")).mark_point(
        shape="triangle-down", size=60, color="black",
    ).encode(x="model:N", y="floor_graphs:Q", xOffset="gpu:N")

    chart = (bars + floor_marks).properties(
        width=500, height=300,
        title=f"Node budget per model x GPU (W={num_workers}). triangle = throughput floor",
    )
    _save(chart, out_dir / f"budget_comparison_w{num_workers}.png")


# ---------------------------------------------------------------------------
# Plot 4: Single model deep-dive (throughput, VRAM, GPU util)
# ---------------------------------------------------------------------------

def plot_deep_dive(
    label: str, p: ModelParams, num_workers: int,
    gpu_name: str, free_bytes: int, out_dir: Path,
):
    """Three-panel deep dive: throughput, VRAM fraction, GPU utilization."""
    max_graphs = int(p.mem_budget(free_bytes) / p.mean_nodes * 1.2) + 10
    B = np.linspace(1, max_graphs, 500)
    N = B * p.mean_nodes
    t_gpu = p.alpha_train_s + p.beta_train_s * N
    step_time = np.maximum(p.gamma_s * B / num_workers, t_gpu)

    df = pl.DataFrame({
        "batch_graphs": B,
        "throughput_knps": N / step_time / 1000,
        "vram_pct": np.clip(N * p.bpn / free_bytes * 100, 0, 150),
        "gpu_pct": np.clip(t_gpu / step_time * 100, 0, 100),
    })

    base = alt.Chart(df).encode(
        x=alt.X("batch_graphs:Q", title="Batch size (graphs)",
                 scale=alt.Scale(domain=[0, max_graphs])))
    vrules = _rule("x", p.mem_budget(free_bytes) / p.mean_nodes, "red")
    floor = p.throughput_floor(num_workers)
    if floor is not None:
        vrules += _rule("x", floor / p.mean_nodes, "orange")

    # Panel 1: Throughput
    p1 = (base.mark_line(color="#1f77b4", strokeWidth=1.5).encode(
        y=alt.Y("throughput_knps:Q", title="Throughput (kN/s)")) + vrules
    ).properties(width=550, height=180,
                 title=f"{label} on {gpu_name} -- W={num_workers}")

    # Panel 2: VRAM usage
    p2 = (base.mark_area(color="purple", opacity=0.3).encode(
        y=alt.Y("vram_pct:Q", title="VRAM used (%)",
                 scale=alt.Scale(domain=[0, 110])))
        + _rule("y", SAFETY_MARGIN * 100, "red") + vrules
    ).properties(width=550, height=150)

    # Panel 3: GPU utilization (green area = active, white = idle)
    p3 = (base.mark_area(color="green", opacity=0.4).encode(
        y=alt.Y("gpu_pct:Q", title="GPU utilization (%)",
                 scale=alt.Scale(domain=[0, 105]))) + vrules
    ).properties(width=550, height=150)

    chart = alt.vconcat(p1, p2, p3).resolve_scale(x="shared")
    m, s = label.split("/")
    _save(chart, out_dir / f"deepdive_{m}_{s}_{gpu_name.replace(' ', '_')}_w{num_workers}.png")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    """Entry point -- run as ``python -m graphids.plots.budget``."""
    parser = argparse.ArgumentParser(
        description="Budget cost-model plots from probe-budget calibration CSV.")
    parser.add_argument("--csv", type=Path, required=True,
                        help="budget_calibration.csv from probe-budget")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model deep-dive, e.g. 'vgae/small'")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU name from clusters.yaml (default: first)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    gpus, gpu_name, free_bytes = load_gpus(args.gpu)
    models = fit_models(load_calibration_csv(args.csv))

    if args.model:
        if args.model not in models:
            print(f"ERROR: Model '{args.model}' not found. "
                  f"Available: {list(models)}", file=sys.stderr)
            sys.exit(1)
        print(f"Deep dive: {args.model}")
        plot_deep_dive(args.model, models[args.model], args.workers,
                       gpu_name, free_bytes, args.out)
    else:
        print(f"Generating all plots (W={args.workers}, GPU={gpu_name})...")
        plot_throughput_curves(models, args.workers, gpu_name, free_bytes, args.out)
        plot_regime_map(models, args.out)
        plot_budget_comparison(models, gpus, args.workers, args.out)
        for label, p_model in models.items():
            plot_deep_dive(label, p_model, args.workers, gpu_name, free_bytes, args.out)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
