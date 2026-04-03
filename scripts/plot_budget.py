#!/usr/bin/env python3
"""Visualize budget cost model: throughput curves, VRAM, regime boundaries.

Usage:
    # From measured probe values (budget_matrix.csv from probe-budget --matrix):
    python scripts/plot_budget.py --csv experimentruns/reference/budget_matrix.csv

    # From hardcoded probe values (no GPU needed):
    python scripts/plot_budget.py --example

    # Single model deep-dive:
    python scripts/plot_budget.py --example --model vgae/small

Output: saves PNG files to the specified --out directory (default: plots/).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Probe data: measured on Pitzer V100, job 46273452 (2026-04-03)
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    model: str
    scale: str
    bpn: int             # bytes per node (includes backward multiplier)
    bwd_mult: float      # backward VRAM multiplier (fwd+bwd / fwd)
    gamma_us: float      # γ: collation cost per graph (μs)
    alpha_ms: float      # α: GPU kernel overhead per step (ms)
    beta_us: float       # β: GPU cost per node (μs), forward-only
    mean_nodes: float    # dataset mean nodes per graph


EXAMPLE_PROBES = [
    ProbeResult("vgae", "small", 34600, 1.39, 65, 7.1, 0.00, 28.2),
    ProbeResult("vgae", "large", 50000, 1.26, 65, 6.9, 0.16, 28.2),
    ProbeResult("gat",  "small", 59900, 1.29, 65, 2.7, 0.85, 28.2),
    ProbeResult("gat",  "large", 224000, 1.52, 65, 4.6, 0.73, 28.2),
    ProbeResult("dgi",  "small", 13500, 2.0,  65, 7.1, 0.02, 28.2),
    ProbeResult("dgi",  "large", 78000, 2.0,  65, 6.1, 0.10, 28.2),
]

# GPU VRAM (free after model load, approximate)
GPUS = {
    "V100 16GB": 14 * 1024**3,
    "A100 40GB": 37 * 1024**3,
    "A100 80GB": 75 * 1024**3,
}

SAFETY_MARGIN = 0.85


def _throughput_model(
    B: np.ndarray,
    gamma_s: float,
    alpha_s: float,
    beta_s: float,
    bwd_mult: float,
    mean_nodes: float,
    num_workers: int,
) -> np.ndarray:
    """Throughput (nodes/sec) as a function of batch size B (graphs).

    Pipeline model:
        throughput = N / max(T_collate/W, T_gpu)
    where:
        N = B × m̄
        T_collate = γ × B
        T_gpu = α_train + β_train × N
        α_train = α × bwd_mult
        β_train = β × bwd_mult
    """
    N = B * mean_nodes
    t_collate_eff = gamma_s * B / num_workers  # sec, with W workers
    alpha_train = alpha_s * bwd_mult
    beta_train = beta_s * bwd_mult
    t_gpu = alpha_train + beta_train * N       # sec

    step_time = np.maximum(t_collate_eff, t_gpu)
    throughput = N / step_time  # nodes/sec
    return throughput


def _mem_budget_nodes(free_bytes: int, bpn: int) -> int:
    return int(free_bytes * SAFETY_MARGIN / bpn)


def _throughput_floor(
    gamma_s: float,
    alpha_s: float,
    beta_s: float,
    bwd_mult: float,
    mean_nodes: float,
    num_workers: int,
) -> int | None:
    """Minimum batch size (nodes) to amortize GPU overhead α.

    Returns None if compute-bound (no floor exists).
    """
    alpha_train = alpha_s * bwd_mult
    beta_train = beta_s * bwd_mult
    collation_rate = gamma_s / num_workers      # sec/graph
    gpu_rate = beta_train * mean_nodes           # sec/graph
    if collation_rate > gpu_rate and alpha_train > 0:
        b_floor = alpha_train / (collation_rate - gpu_rate)
        return max(1, int(math.ceil(b_floor * mean_nodes)))
    return None


# ---------------------------------------------------------------------------
# Plot 1: Throughput vs batch size — shows floor, ceiling, operating point
# ---------------------------------------------------------------------------

def plot_throughput_curves(probes: list[ProbeResult], num_workers: int,
                          gpu_name: str, free_bytes: int, out_dir: Path):
    """One figure per model: throughput vs batch size with annotations."""
    for p in probes:
        label = f"{p.model}/{p.scale}"
        gamma_s = p.gamma_us * 1e-6
        alpha_s = p.alpha_ms * 1e-3
        beta_s = p.beta_us * 1e-6

        mem_budget = _mem_budget_nodes(free_bytes, p.bpn)
        floor = _throughput_floor(gamma_s, alpha_s, beta_s, p.bwd_mult,
                                  p.mean_nodes, num_workers)

        # Batch sizes from 1 graph to mem_budget (in graphs)
        max_graphs = int(mem_budget / p.mean_nodes) + 100
        B = np.linspace(1, max_graphs, 2000)

        tp = _throughput_model(B, gamma_s, alpha_s, beta_s, p.bwd_mult,
                               p.mean_nodes, num_workers)
        tp_knps = tp / 1000  # kilo-nodes/sec

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(B, tp_knps, "b-", linewidth=1.5, label="throughput")

        # Mark mem_budget
        mem_graphs = mem_budget / p.mean_nodes
        ax.axvline(mem_graphs, color="red", linestyle="--", linewidth=1.2,
                   label=f"VRAM ceiling ({mem_graphs:.0f} graphs = {mem_budget:,} nodes)")

        # Mark throughput floor
        if floor is not None:
            floor_graphs = floor / p.mean_nodes
            ax.axvline(floor_graphs, color="orange", linestyle=":",
                       linewidth=1.2,
                       label=f"throughput floor ({floor_graphs:.0f} graphs = {floor:,} nodes)")

        # Asymptotic throughput lines
        # Collation-limited: m̄·W/γ
        tp_collation = p.mean_nodes * num_workers / gamma_s / 1000
        ax.axhline(tp_collation, color="gray", linestyle="-.", alpha=0.5,
                   label=f"collation limit (m̄·W/γ = {tp_collation:.0f} kN/s)")

        # Compute-limited: 1/β_train
        beta_train = beta_s * p.bwd_mult
        if beta_train > 0:
            tp_compute = 1 / beta_train / 1000
            if tp_compute < tp_collation * 5:  # only show if visible
                ax.axhline(tp_compute, color="green", linestyle="-.", alpha=0.5,
                           label=f"compute limit (1/β_train = {tp_compute:.0f} kN/s)")

        # Regime annotation
        cg_ratio = (gamma_s / (p.mean_nodes * num_workers)) / beta_train if beta_train > 0 else float("inf")
        regime = "collation-bound" if cg_ratio > 1 else "compute-bound"

        ax.set_xlabel("Batch size (graphs)")
        ax.set_ylabel("Throughput (k nodes/sec)")
        ax.set_title(
            f"{label} on {gpu_name} — W={num_workers}, "
            f"regime: {regime} (cg_ratio={cg_ratio:.2f})"
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, max_graphs)
        ax.set_ylim(0, max(tp_knps) * 1.15)
        ax.grid(True, alpha=0.3)

        fname = out_dir / f"throughput_{p.model}_{p.scale}_{gpu_name.replace(' ', '_')}_w{num_workers}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fname}")


# ---------------------------------------------------------------------------
# Plot 2: Regime map — heatmap of cg_ratio across models × worker counts
# ---------------------------------------------------------------------------

def plot_regime_map(probes: list[ProbeResult], out_dir: Path):
    """Heatmap: cg_ratio for each model × worker count."""
    worker_counts = [1, 2, 4, 6, 8, 12]
    labels = [f"{p.model}/{p.scale}" for p in probes]

    matrix = np.zeros((len(probes), len(worker_counts)))
    for i, p in enumerate(probes):
        gamma_s = p.gamma_us * 1e-6
        beta_s = p.beta_us * 1e-6
        beta_train = beta_s * p.bwd_mult
        for j, w in enumerate(worker_counts):
            gamma_eff = gamma_s / (p.mean_nodes * w)
            matrix[i, j] = gamma_eff / beta_train if beta_train > 0 else 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=5)

    ax.set_xticks(range(len(worker_counts)))
    ax.set_xticklabels([str(w) for w in worker_counts])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("num_workers")
    ax.set_ylabel("model")
    ax.set_title("Training regime: cg_ratio (>1 = collation-bound, <1 = compute-bound)")

    # Annotate cells
    for i in range(len(probes)):
        for j in range(len(worker_counts)):
            val = matrix[i, j]
            text = f"{val:.1f}" if val < 50 else "∞"
            color = "white" if val > 3 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    # Draw regime boundary (cg_ratio = 1)
    fig.colorbar(im, ax=ax, label="cg_ratio")
    ax.axhline(-0.5, color="black", linewidth=0.5)  # just grid cleanup

    fname = out_dir / "regime_map.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------------
# Plot 3: Budget comparison across GPUs
# ---------------------------------------------------------------------------

def plot_budget_comparison(probes: list[ProbeResult], num_workers: int,
                           out_dir: Path):
    """Bar chart: budget (in graphs) per model × GPU, with floor markers."""
    labels = [f"{p.model}/{p.scale}" for p in probes]
    gpu_names = list(GPUS.keys())
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    for gi, (gpu_name, free_bytes) in enumerate(GPUS.items()):
        budgets = []
        floors = []
        for p in probes:
            gamma_s = p.gamma_us * 1e-6
            alpha_s = p.alpha_ms * 1e-3
            beta_s = p.beta_us * 1e-6
            mem = _mem_budget_nodes(free_bytes, p.bpn)
            budgets.append(mem / p.mean_nodes)
            fl = _throughput_floor(gamma_s, alpha_s, beta_s, p.bwd_mult,
                                   p.mean_nodes, num_workers)
            floors.append(fl / p.mean_nodes if fl else 0)

        bars = ax.bar(x + gi * width, budgets, width, label=gpu_name, alpha=0.8)
        # Mark throughput floors as horizontal markers
        for xi, fl in zip(x + gi * width, floors):
            if fl > 0:
                ax.plot(xi, fl, "kv", markersize=6)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Budget (graphs per batch)")
    ax.set_title(f"Node budget per model × GPU (W={num_workers}). ▼ = throughput floor")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fname = out_dir / f"budget_comparison_w{num_workers}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------------
# Plot 4: Single model deep-dive (throughput, VRAM, GPU util)
# ---------------------------------------------------------------------------

def plot_deep_dive(p: ProbeResult, num_workers: int, gpu_name: str,
                   free_bytes: int, out_dir: Path):
    """Three-panel deep dive for one model: throughput, VRAM fraction, GPU util."""
    label = f"{p.model}/{p.scale}"
    gamma_s = p.gamma_us * 1e-6
    alpha_s = p.alpha_ms * 1e-3
    beta_s = p.beta_us * 1e-6
    alpha_train = alpha_s * p.bwd_mult
    beta_train = beta_s * p.bwd_mult

    mem_budget = _mem_budget_nodes(free_bytes, p.bpn)
    floor = _throughput_floor(gamma_s, alpha_s, beta_s, p.bwd_mult,
                              p.mean_nodes, num_workers)

    max_graphs = int(mem_budget / p.mean_nodes * 1.2) + 10
    B = np.linspace(1, max_graphs, 2000)
    N = B * p.mean_nodes

    # Timing components
    t_collate_eff = gamma_s * B / num_workers
    t_gpu = alpha_train + beta_train * N
    step_time = np.maximum(t_collate_eff, t_gpu)
    throughput = N / step_time / 1000

    # VRAM fraction
    vram_frac = np.clip(N * p.bpn / free_bytes, 0, 1.5)

    # GPU utilization = t_gpu / step_time (fraction of time GPU is working)
    gpu_util = np.clip(t_gpu / step_time, 0, 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Panel 1: Throughput
    ax = axes[0]
    ax.plot(B, throughput, "b-", linewidth=1.5)
    ax.set_ylabel("Throughput (kN/s)")
    ax.set_title(f"{label} on {gpu_name} — W={num_workers}")
    if floor:
        ax.axvline(floor / p.mean_nodes, color="orange", ls=":", lw=1.2,
                   label=f"floor ({floor:,} nodes)")
    ax.axvline(mem_budget / p.mean_nodes, color="red", ls="--", lw=1.2,
               label=f"VRAM ceiling ({mem_budget:,} nodes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: VRAM usage
    ax = axes[1]
    ax.fill_between(B, vram_frac * 100, alpha=0.3, color="purple")
    ax.plot(B, vram_frac * 100, "purple", linewidth=1)
    ax.axhline(SAFETY_MARGIN * 100, color="red", ls="--", lw=1, label=f"safety limit ({SAFETY_MARGIN*100:.0f}%)")
    ax.set_ylabel("VRAM used (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: GPU utilization + timing decomposition
    ax = axes[2]
    ax.fill_between(B, gpu_util * 100, alpha=0.3, color="green", label="GPU active")
    ax.fill_between(B, gpu_util * 100, 100, alpha=0.15, color="red", label="GPU idle (waiting for data)")
    ax.set_ylabel("GPU utilization (%)")
    ax.set_xlabel("Batch size (graphs)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = out_dir / f"deepdive_{p.model}_{p.scale}_{gpu_name.replace(' ', '_')}_w{num_workers}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="Budget matrix CSV from probe-budget --matrix")
    parser.add_argument("--example", action="store_true",
                        help="Use hardcoded probe values (no CSV needed)")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model deep-dive, e.g. 'vgae/small'")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu", type=str, default="V100 16GB",
                        choices=list(GPUS.keys()))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()

    if not args.example and not args.csv:
        parser.error("Specify --example or --csv")

    args.out.mkdir(parents=True, exist_ok=True)
    free_bytes = GPUS[args.gpu]

    if args.example:
        probes = EXAMPLE_PROBES
    else:
        # TODO: parse CSV into ProbeResult list
        raise NotImplementedError("CSV parsing not yet implemented. Use --example.")

    if args.model:
        matching = [p for p in probes if f"{p.model}/{p.scale}" == args.model]
        if not matching:
            available = [f"{p.model}/{p.scale}" for p in probes]
            parser.error(f"Model '{args.model}' not found. Available: {available}")
        print(f"Deep dive: {args.model}")
        plot_deep_dive(matching[0], args.workers, args.gpu, free_bytes, args.out)
    else:
        print(f"Generating all plots (W={args.workers}, GPU={args.gpu})...")
        print("\n1. Throughput curves:")
        plot_throughput_curves(probes, args.workers, args.gpu, free_bytes, args.out)
        print("\n2. Regime map:")
        plot_regime_map(probes, args.out)
        print("\n3. Budget comparison:")
        plot_budget_comparison(probes, args.workers, args.out)
        print("\n4. Deep dives:")
        for p in probes:
            plot_deep_dive(p, args.workers, args.gpu, free_bytes, args.out)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
