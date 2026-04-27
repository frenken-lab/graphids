"""Query MLflow per node, format the result.

The plan jsonnet IS the topology source — no bash parsing, no regex.
Each preset node maps directly to ``(dataset, group, variant, seed, phase)``
for :func:`graphids._mlflow.build_search_filter`. Command nodes write no
MLflow row (see ``data-layout.md``) and short-circuit to ``"NA"``.

Status strings are closed: FINISHED, RUNNING, FAILED, KILLED, PENDING, NA.
Anything else from MLflow falls through verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphids.slurm.dag import Node

_PENDING = "PENDING"
_NA = "NA"


@dataclass(frozen=True)
class NodeStatus:
    """One status entry — paired with the source ``Node``."""

    node: Node
    status: str
    run_id: str | None = None
    end_time: str | None = None


def query_node_status(node: Node, *, dataset: str, seed: int) -> NodeStatus:
    """Look up the latest MLflow row for ``(dataset, group, variant, seed, phase)``."""
    if node.is_command:
        return NodeStatus(node=node, status=_NA)

    from graphids._mlflow import latest_run

    phase = "test" if node.action == "test" else "fit"
    row = latest_run(
        dataset=dataset, group=node.group, variant=node.variant, seed=seed, phase=phase
    )
    if row is None:
        return NodeStatus(node=node, status=_PENDING)
    return NodeStatus(
        node=node,
        status=str(row["status"]),
        run_id=str(row["run_id"]) if row.get("run_id") else None,
        end_time=str(row["end_time"]) if row.get("end_time") is not None else None,
    )


def query_all(nodes: tuple[Node, ...], *, dataset: str, seed: int) -> list[NodeStatus]:
    return [query_node_status(n, dataset=dataset, seed=seed) for n in nodes]


# --------------------------------------------------------------------------
# Formatters.
# --------------------------------------------------------------------------


_STATUS_STYLE = {
    "FINISHED": "green",
    "RUNNING": "yellow",
    "FAILED": "red",
    "KILLED": "red",
    "PENDING": "dim",
    _NA: "dim",
}


def _summarize(statuses: list[NodeStatus]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in statuses:
        counts[s.status] = counts.get(s.status, 0) + 1
    return counts


def format_table(statuses: list[NodeStatus], *, dataset: str, seed: int) -> str:
    """Human-readable status table via ``rich`` (already a Typer dep)."""
    from io import StringIO

    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Plan: dataset={dataset} seed={seed}", expand=False)
    table.add_column("Node")
    table.add_column("Status")
    table.add_column("Run ID")
    for s in statuses:
        style = _STATUS_STYLE.get(s.status, "")
        table.add_row(
            s.node.name, f"[{style}]{s.status}[/{style}]" if style else s.status, s.run_id or "—"
        )
    counts = _summarize(statuses)
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    buf = StringIO()
    Console(file=buf, force_terminal=True, width=120).print(table)
    buf.write(f"\nSummary: {len(statuses)} nodes — {summary}\n")
    return buf.getvalue()


def format_json(statuses: list[NodeStatus], *, dataset: str, seed: int) -> str:
    """Machine-readable; for piping into jq / scripts."""
    payload = {
        "plan": {"dataset": dataset, "seed": seed},
        "nodes": [
            {
                "name": s.node.name,
                "group": s.node.group,
                "variant": s.node.variant,
                "action": s.node.action,
                "is_command": s.node.is_command,
                "deps": list(s.node.deps),
                "status": s.status,
                "run_id": s.run_id,
                "end_time": s.end_time,
            }
            for s in statuses
        ],
        "summary": _summarize(statuses),
    }
    return json.dumps(payload, indent=2) + "\n"
