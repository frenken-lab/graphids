"""Render a plan jsonnet to a JSONL blueprint.

The plan is *data*; the JSONL is the *blueprint*. ``graphids run``
emits this — it does not submit anything itself, and it does not
produce an executable artifact. The user (or an LLM walking the
JSONL) iterates row by row and decides how to invoke each
``submit_command``.

One row per plan node. Schema::

    {
      "name":           "vgae",
      "preset":         "configs/ablations/unsupervised/vgae.jsonnet",  // or null
      "command":        null,                                            // or shell string
      "action":         "fit",
      "deps":           ["upstream-node-name", ...],
      "mode":           "gpu",
      "length":         "long",
      "mem_gb":         null,
      "timeout_min":    210,
      "submit_command": "graphids submit configs/ablations/... --skip-if-finished"
    }

``submit_command`` is the literal one-shot invocation for this node.
It includes ``--skip-if-finished`` for preset nodes (MLflow short-
circuit on re-runs); command nodes omit it (no MLflow row to check).
``--depends-on`` is auto-emitted for nodes with ``cross_plan_deps``
declared in the plan jsonnet (e.g. curriculum_vgae → vgae:N).
For ad-hoc SLURM-level afterok on a RUNNING upstream, add it manually.

See ``.claude/rules/single-submission-primitive.md`` for the full
architectural commitment behind this shape.
"""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphids.slurm.dag import Node


def _preset_submit_command(
    node: Node,
    *,
    dataset: str,
    seed: int,
    cluster: str,
) -> str:
    """Render the literal `graphids submit` invocation for a preset node."""
    parts = [
        "graphids submit",
        f"configs/ablations/{node.preset_path}",
        f"--dataset {dataset}",
        f"--seed {seed}",
        f"--cluster {cluster}",
    ]
    if node.action == "test":
        parts.append("--action test")
    if node.mode != "gpu":
        parts.append(f"--mode {node.mode}")
    if node.length != "long":
        parts.append(f"--length {node.length}")
    if node.mem_gb is not None:
        parts.append(f"--mem-gb {node.mem_gb}")
    if node.timeout_min is not None:
        parts.append(f"--timeout-min {node.timeout_min}")
    if node.cross_plan_deps:
        parts.append(f"--depends-on {','.join(f'{v}:{seed}' for v in node.cross_plan_deps)}")
    parts.append("--skip-if-finished")
    return " ".join(parts)


def _command_submit_command(node: Node, *, cluster: str, seed: int) -> str:
    """Render the literal `graphids submit --command` invocation for a command node."""
    assert node.command is not None
    parts = [
        "graphids submit",
        f"--command {shlex.quote(node.command)}",
        f"--mode {node.mode}",
        f"--cluster {cluster}",
    ]
    if node.mem_gb is not None:
        parts.append(f"--mem-gb {node.mem_gb}")
    if node.timeout_min is not None:
        parts.append(f"--timeout-min {node.timeout_min}")
    if node.cross_plan_deps:
        parts.append(f"--depends-on {','.join(f'{v}:{seed}' for v in node.cross_plan_deps)}")
    return " ".join(parts)


def node_submit_command(node: Node, *, dataset: str, seed: int, cluster: str) -> str:
    """Render the submit_command for any node type."""
    if node.is_command:
        return _command_submit_command(node, cluster=cluster, seed=seed)
    return _preset_submit_command(node, dataset=dataset, seed=seed, cluster=cluster)


def _row(node: Node, *, dataset: str, seed: int, cluster: str) -> dict[str, Any]:
    """Build the JSONL row dict for one node."""
    submit_command = node_submit_command(node, dataset=dataset, seed=seed, cluster=cluster)
    row = {
        "name": node.name,
        "preset": (
            f"configs/ablations/{node.preset_path}" if node.preset_path is not None else None
        ),
        "command": node.command,
        "action": node.action,
        "deps": list(node.deps),
        "mode": node.mode,
        "length": node.length,
        "mem_gb": node.mem_gb,
        "timeout_min": node.timeout_min,
        "submit_command": submit_command,
    }
    return {k: v for k, v in row.items() if v is not None}


def render_plan_jsonl(
    nodes: tuple[Node, ...],
    *,
    dataset: str,
    seed: int,
    cluster: str,
) -> str:
    """Return the plan as JSONL — one JSON object per topo-sorted node."""
    from graphids.slurm.dag import toposort

    ordered = toposort(nodes)
    return (
        "\n".join(json.dumps(_row(n, dataset=dataset, seed=seed, cluster=cluster)) for n in ordered)
        + "\n"
    )
