#!/usr/bin/env python3
"""Render codebase-graph.yaml as visual diagrams.

Usage:
    python .claude/render_graph.py                  # all views
    python .claude/render_graph.py --view arch      # architecture only
    python .claude/render_graph.py --view data      # data flow only
    python .claude/render_graph.py --view full      # everything (dense)

Outputs PNGs to .claude/ directory.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

GRAPH_PATH = Path(__file__).parent / "codebase-graph.yaml"
OUT_DIR = Path(__file__).parent

LAYER_COLORS = {
    "config": "#4a90d9",
    "pipeline": "#2ecc71",
    "core": "#e74c3c",
    "infra": "#f39c12",
    "external": "#9b59b6",
}
LAYER_BG = {
    "config": "#e8f4fd",
    "pipeline": "#e8fde8",
    "core": "#fde8e8",
    "infra": "#fdf8e8",
    "external": "#f0e8fd",
}
TYPE_SHAPES = {
    "source": "box",
    "config": "component",
    "database": "cylinder",
    "artifact": "note",
    "output": "doubleoctagon",
    "script": "hexagon",
    "test": "diamond",
}
EDGE_STYLES = {
    "imports": {"style": "solid", "color": "#666666", "arrowhead": "normal"},
    "calls": {"style": "solid", "color": "#333333", "arrowhead": "normal"},
    "dispatches": {"style": "dashed", "color": "#e67e22", "arrowhead": "open"},
    "reads": {"style": "solid", "color": "#3498db", "arrowhead": "normal"},
    "writes": {"style": "solid", "color": "#e74c3c", "arrowhead": "normal"},
    "produces": {"style": "bold", "color": "#9b59b6", "arrowhead": "normal"},
    "consumes": {"style": "bold", "color": "#9b59b6", "arrowhead": "normal"},
}


def load_graph() -> dict:
    with open(GRAPH_PATH) as f:
        return yaml.safe_load(f)


def render_dot(dot_source: str, out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_source)
        dot_file = f.name
    result = subprocess.run(
        ["dot", "-Tpng", "-Gdpi=150", dot_file, "-o", str(out_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"dot error: {result.stderr}")
    else:
        print(f"  -> {out_path} ({out_path.stat().st_size // 1024} KB)")
    Path(dot_file).unlink()


def make_node_label(n: dict, compact: bool = False) -> str:
    name = n["id"].replace("_", " ")
    if compact:
        return name
    parts = [name]
    if n.get("lines"):
        parts.append(f"{n['lines']}L")
    if n.get("format"):
        parts.append(f".{n['format']}")
    return "\\n".join([parts[0], " | ".join(parts[1:])] if len(parts) > 1 else parts)


def arch_view(data: dict) -> str:
    """Architecture view: source + config nodes grouped by layer, imports/calls only."""
    keep_types = {"source", "config"}
    keep_rels = {"imports", "calls"}
    nodes = {n["id"]: n for n in data["nodes"] if n["type"] in keep_types}
    by_layer = defaultdict(list)
    for n in nodes.values():
        by_layer[n["layer"]].append(n)

    lines = [
        "digraph arch {",
        "    rankdir=TB;",
        '    fontname="Helvetica"; fontsize=12;',
        '    node [fontname="Helvetica", fontsize=10, style=filled, fillcolor=white];',
        "    edge [fontsize=8];",
        "",
    ]

    for layer in ["config", "pipeline", "core"]:
        items = by_layer.get(layer, [])
        lines.append(f"    subgraph cluster_{layer} {{")
        lines.append(f'        label="{layer.upper()} LAYER";')
        lines.append(f'        style=filled; fillcolor="{LAYER_BG[layer]}";')
        lines.append(f'        color="{LAYER_COLORS[layer]}"; penwidth=2;')
        for n in sorted(items, key=lambda x: x["id"]):
            shape = TYPE_SHAPES.get(n["type"], "box")
            label = make_node_label(n)
            fc = LAYER_COLORS[layer]
            lines.append(
                f'        {n["id"]} [label="{label}", shape={shape}, '
                f'fillcolor="{LAYER_BG[layer]}", color="{fc}"];'
            )
        lines.append("    }")

    seen = set()
    for e in data["edges"]:
        f, t, r = e["from"], e["to"], e["rel"]
        if f not in nodes or t not in nodes or r not in keep_rels:
            continue
        key = (f, t)
        if key in seen:
            continue
        seen.add(key)
        es = EDGE_STYLES.get(r, {})
        lines.append(
            f'    {f} -> {t} [color="{es.get("color", "#666")}", style={es.get("style", "solid")}];'
        )

    lines.append("}")
    return "\n".join(lines)


def data_flow_view(data: dict) -> str:
    """Data flow: sources that read/write + databases/artifacts/outputs."""
    data_rels = {"reads", "writes", "produces", "consumes"}
    data_types = {"database", "artifact", "output"}

    # Find all nodes involved in data edges
    involved = set()
    relevant_edges = []
    for e in data["edges"]:
        if e["rel"] in data_rels:
            involved.add(e["from"])
            involved.add(e["to"])
            relevant_edges.append(e)

    nodes = {n["id"]: n for n in data["nodes"] if n["id"] in involved}

    lines = [
        "digraph dataflow {",
        "    rankdir=LR;",
        '    fontname="Helvetica"; fontsize=12;',
        '    node [fontname="Helvetica", fontsize=10, style=filled];',
        "    edge [fontsize=8];",
        "",
        "    // Legend",
        "    subgraph cluster_legend {",
        '        label="Legend"; style=filled; fillcolor="#f5f5f5";',
        '        leg_r [label="reads", shape=plaintext, fillcolor="#f5f5f5"];',
        '        leg_w [label="writes", shape=plaintext, fillcolor="#f5f5f5"];',
        '        leg_p [label="produces", shape=plaintext, fillcolor="#f5f5f5"];',
        '        leg_r -> leg_w [color="#3498db", label="reads"];',
        '        leg_w -> leg_p [color="#e74c3c", label="writes"];',
        "    }",
        "",
    ]

    # Group data nodes
    lines.append("    subgraph cluster_data {")
    lines.append('        label="DATA STORES & OUTPUTS";')
    lines.append('        style=filled; fillcolor="#fdf8e8"; color="#d9b34a"; penwidth=2;')
    for n in sorted(nodes.values(), key=lambda x: x["id"]):
        if n["type"] in data_types:
            shape = TYPE_SHAPES.get(n["type"], "box")
            label = make_node_label(n, compact=True)
            lines.append(
                f'        {n["id"]} [label="{label}", shape={shape}, '
                f'fillcolor="#fff3cd", color="#d9b34a"];'
            )
    lines.append("    }")

    # Source nodes
    lines.append("    subgraph cluster_sources {")
    lines.append('        label="SOURCE FILES";')
    lines.append('        style=filled; fillcolor="#e8f4fd"; color="#4a90d9"; penwidth=2;')
    for n in sorted(nodes.values(), key=lambda x: x["id"]):
        if n["type"] not in data_types:
            layer = n.get("layer", "pipeline")
            fc = LAYER_COLORS.get(layer, "#666")
            label = make_node_label(n, compact=True)
            lines.append(
                f'        {n["id"]} [label="{label}", shape=box, '
                f'fillcolor="#e8f4fd", color="{fc}"];'
            )
    lines.append("    }")

    seen = set()
    for e in relevant_edges:
        key = (e["from"], e["to"], e["rel"])
        if key in seen:
            continue
        seen.add(key)
        es = EDGE_STYLES.get(e["rel"], {})
        label = e["rel"]
        lines.append(
            f'    {e["from"]} -> {e["to"]} [color="{es.get("color", "#666")}", '
            f'style={es.get("style", "solid")}, label="{label}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def full_view(data: dict) -> str:
    """Full graph — all nodes and edges, grouped by layer."""
    nodes = {n["id"]: n for n in data["nodes"]}
    by_layer = defaultdict(list)
    for n in data["nodes"]:
        by_layer[n["layer"]].append(n)

    lines = [
        "digraph full {",
        "    rankdir=TB;",
        '    fontname="Helvetica"; fontsize=10;',
        '    node [fontname="Helvetica", fontsize=8, style=filled, fillcolor=white];',
        "    edge [fontsize=7];",
        "    overlap=false; splines=true;",
        "",
    ]

    for layer in ["config", "pipeline", "core", "infra", "external"]:
        items = by_layer.get(layer, [])
        if not items:
            continue
        lines.append(f"    subgraph cluster_{layer} {{")
        lines.append(f'        label="{layer.upper()}";')
        lines.append(f'        style=filled; fillcolor="{LAYER_BG[layer]}";')
        lines.append(f'        color="{LAYER_COLORS[layer]}"; penwidth=2;')
        for n in sorted(items, key=lambda x: x["id"]):
            shape = TYPE_SHAPES.get(n["type"], "box")
            label = make_node_label(n, compact=True)
            fc = LAYER_COLORS[layer]
            lines.append(
                f'        {n["id"]} [label="{label}", shape={shape}, '
                f'fillcolor="{LAYER_BG[layer]}", color="{fc}"];'
            )
        lines.append("    }")

    seen = set()
    for e in data["edges"]:
        key = (e["from"], e["to"])
        if key in seen:
            continue
        seen.add(key)
        es = EDGE_STYLES.get(e["rel"], {})
        lines.append(
            f'    {e["from"]} -> {e["to"]} [color="{es.get("color", "#666")}", '
            f"style={es.get('style', 'solid')}];"
        )

    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render codebase graph")
    parser.add_argument(
        "--view",
        choices=["arch", "data", "full", "all"],
        default="all",
        help="Which view to render",
    )
    args = parser.parse_args()

    data = load_graph()
    views = {
        "arch": ("Architecture (source imports)", arch_view, "codebase-arch.png"),
        "data": ("Data flow (reads/writes)", data_flow_view, "codebase-dataflow.png"),
        "full": ("Full graph (all nodes/edges)", full_view, "codebase-full.png"),
    }

    targets = views.keys() if args.view == "all" else [args.view]
    for key in targets:
        desc, fn, filename = views[key]
        print(f"\nRendering: {desc}")
        dot_src = fn(data)
        render_dot(dot_src, OUT_DIR / filename)


if __name__ == "__main__":
    main()
