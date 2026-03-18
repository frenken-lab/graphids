#!/usr/bin/env python3
"""Analyze profiler traces to decide whether cuGraph acceleration is worthwhile.

Compares GAT (no edge_attr) vs TransformerConv (with edge_attr) runs
by parsing Chrome trace JSON files from profiler_traces/ directories.

Usage:
    python scripts/profiling/analyze_profile.py [--dataset DATASET] [--scale SCALE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import os

LAKE_ROOT = Path(os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"))

# Keywords indicating message passing operations in trace events
MP_KEYWORDS = {
    "MessagePassing", "propagate", "message_and_aggregate",
    "GATConv", "GATv2Conv", "TransformerConv", "scatter",
    "message", "aggregate",
}


def find_trace_files(dataset: str, scale: str) -> dict[str, Path | None]:
    """Find profiler trace JSON files for gat and transformer runs."""
    results: dict[str, Path | None] = {"gat": None, "transformer": None}

    for conv_type in ("gat", "transformer"):
        # Try to find the most recent profiler trace
        pattern = f"{dataset}/gat_{scale}_curriculum*"
        for run_dir in sorted(LAKE_ROOT.glob(pattern), reverse=True):
            traces_dir = run_dir / "profiler_traces"
            if not traces_dir.is_dir():
                continue
            # Check if this run used the right conv_type
            cfg_path = run_dir / "config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                run_conv = cfg.get("gat", {}).get("conv_type", "gat")
                if run_conv != conv_type:
                    continue
            trace_files = sorted(traces_dir.glob("*.json"))
            if trace_files:
                results[conv_type] = trace_files[-1]  # Latest trace
                break

    return results


def parse_trace(trace_path: Path) -> dict:
    """Parse Chrome trace JSON and extract timing breakdown."""
    data = json.loads(trace_path.read_text())

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    total_dur_us = 0.0
    mp_dur_us = 0.0

    for ev in events:
        if ev.get("ph") != "X":  # Duration events only
            continue
        dur = ev.get("dur", 0)
        name = ev.get("name", "")
        total_dur_us += dur

        # Check if this is a message passing operation
        if any(kw in name for kw in MP_KEYWORDS):
            mp_dur_us += dur

    return {
        "total_step_us": total_dur_us,
        "mp_us": mp_dur_us,
        "mp_pct": (mp_dur_us / total_dur_us * 100) if total_dur_us > 0 else 0.0,
    }


def load_metrics(dataset: str, scale: str, conv_type: str) -> dict | None:
    """Load evaluation metrics for a given conv_type run."""
    pattern = f"{dataset}/eval_{scale}_evaluation*"
    for run_dir in sorted(LAKE_ROOT.glob(pattern), reverse=True):
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        metrics = json.loads(mp.read_text())
        gat_metrics = metrics.get("gat", {}).get("core", {})
        if gat_metrics:
            return gat_metrics

    # Also try loading from the curriculum run's directory parent eval
    pattern2 = f"{dataset}/gat_{scale}_curriculum*"
    for run_dir in sorted(LAKE_ROOT.glob(pattern2), reverse=True):
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("gat", {}).get("conv_type") == conv_type:
                mp = run_dir / "metrics.json"
                if mp.exists():
                    return json.loads(mp.read_text()).get("gat", {}).get("core", {})
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze conv_type profiling results")
    parser.add_argument("--dataset", default="hcrl_sa")
    parser.add_argument("--scale", default="large")
    parser.add_argument("--mp-threshold", type=float, default=30.0,
                        help="Message passing %% threshold for 'bottleneck' (default: 30)")
    parser.add_argument("--f1-threshold", type=float, default=0.02,
                        help="F1 delta threshold for 'edge_attr valuable' (default: 0.02)")
    args = parser.parse_args(argv)

    print(f"\n{'=' * 60}")
    print(f"  cuGraph Decision Analysis: {args.dataset} / {args.scale}")
    print(f"{'=' * 60}\n")

    traces = find_trace_files(args.dataset, args.scale)

    gat_profile = None
    transformer_profile = None

    if traces["gat"]:
        gat_profile = parse_trace(traces["gat"])
        print(f"GAT trace: {traces['gat']}")
    else:
        print("WARNING: No GAT profiler trace found")

    if traces["transformer"]:
        transformer_profile = parse_trace(traces["transformer"])
        print(f"Transformer trace: {traces['transformer']}")
    else:
        print("WARNING: No Transformer profiler trace found")

    # Load F1 metrics
    gat_metrics = load_metrics(args.dataset, args.scale, "gat")
    transformer_metrics = load_metrics(args.dataset, args.scale, "transformer")

    gat_f1 = gat_metrics.get("f1", 0) if gat_metrics else 0
    transformer_f1 = transformer_metrics.get("f1", 0) if transformer_metrics else 0
    f1_delta = transformer_f1 - gat_f1

    print(f"\n{'─' * 60}")
    print("  Results")
    print(f"{'─' * 60}")

    if gat_profile:
        print(f"  Message passing % (gat):         {gat_profile['mp_pct']:.1f}%")
    if transformer_profile:
        print(f"  Message passing % (transformer): {transformer_profile['mp_pct']:.1f}%")
    print(f"  F1 (gat):                        {gat_f1:.4f}")
    print(f"  F1 (transformer):                {transformer_f1:.4f}")
    print(f"  Edge_attr F1 delta:              {f1_delta:+.4f}")

    # Decision
    mp_pct = gat_profile["mp_pct"] if gat_profile else 0
    mp_is_bottleneck = mp_pct > args.mp_threshold
    edge_attr_valuable = f1_delta > args.f1_threshold

    print(f"\n{'─' * 60}")
    print("  Decision Matrix")
    print(f"{'─' * 60}")
    print(f"  MP > {args.mp_threshold}% of step:  {'YES' if mp_is_bottleneck else 'NO'}")
    print(f"  Edge_attr delta > {args.f1_threshold:.0%}: {'YES' if edge_attr_valuable else 'NO'}")

    if mp_is_bottleneck and not edge_attr_valuable:
        recommendation = "PROCEED with cuGraph (Phase B)"
    elif mp_is_bottleneck and edge_attr_valuable:
        recommendation = "SKIP cuGraph (edge_attr too valuable)"
    else:
        recommendation = "SKIP cuGraph (message passing not the bottleneck)"

    print(f"\n  RECOMMENDATION: {recommendation}")
    print(f"\n{'=' * 60}\n")

    # Write decision document
    docs_dir = Path("docs/decisions")
    docs_dir.mkdir(parents=True, exist_ok=True)
    decision_path = docs_dir / "cugraph_decision.md"
    decision_path.write_text(
        f"# cuGraph Decision Gate\n\n"
        f"**Date**: Auto-generated by `scripts/profiling/analyze_profile.py`\n"
        f"**Dataset**: {args.dataset} / {args.scale}\n\n"
        f"## Profiling Results\n\n"
        f"| Metric | GAT | Transformer |\n"
        f"|--------|-----|-------------|\n"
        f"| Message Passing % | {gat_profile['mp_pct']:.1f}% | {transformer_profile['mp_pct']:.1f}% |\n"
        f"| F1 Score | {gat_f1:.4f} | {transformer_f1:.4f} |\n\n"
        f"## Decision\n\n"
        f"- MP bottleneck (>{args.mp_threshold}%): **{'YES' if mp_is_bottleneck else 'NO'}**\n"
        f"- Edge_attr valuable (delta >{args.f1_threshold:.0%}): **{'YES' if edge_attr_valuable else 'NO'}**\n\n"
        f"**Recommendation**: {recommendation}\n"
        if gat_profile and transformer_profile else
        f"# cuGraph Decision Gate\n\n"
        f"Profiling data incomplete. Re-run `sbatch scripts/profiling/profile_conv_type.sbatch`.\n"
    )
    print(f"Decision document: {decision_path}")


if __name__ == "__main__":
    main()
