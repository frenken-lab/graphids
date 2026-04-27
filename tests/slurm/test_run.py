"""Tests for ``graphids/slurm/run.py`` — JSONL blueprint renderer.

The renderer is a pure function: ``(nodes, dataset, seed, cluster) →
JSONL string``. Tests assert on the parsed JSONL rows; no submission is
exercised. Each row is a literal `graphids submit` invocation that the
user (or an LLM walking the JSONL) runs directly.

See ``.claude/rules/single-submission-primitive.md`` for the
architectural commitment behind the JSONL shape.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from graphids.config.jsonnet import render
from graphids.slurm.dag import Node, parse_plan
from graphids.slurm.run import render_plan_jsonl


@pytest.fixture(scope="module")
def ofat_nodes() -> tuple[Node, ...]:
    return parse_plan(render("configs/plans/ofat.jsonnet", tla={"dataset": "set_01", "seed": 42}))


@pytest.fixture(scope="module")
def ofat_rows(ofat_nodes: tuple[Node, ...]) -> list[dict[str, Any]]:
    out = render_plan_jsonl(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    return [json.loads(line) for line in out.splitlines() if line.strip()]


_REQUIRED_FIELDS = {
    "name",
    "preset",
    "command",
    "action",
    "deps",
    "mode",
    "length",
    "mem_gb",
    "timeout_min",
    "submit_command",
}


# CONTRACT: every row has the documented schema. Missing fields would break
# downstream LLM/user iteration that reads them by name.
def test_every_row_has_full_schema(ofat_rows: list[dict[str, Any]]) -> None:
    for row in ofat_rows:
        assert set(row.keys()) == _REQUIRED_FIELDS, f"{row['name']}: {set(row.keys())}"


# CONTRACT: every row is valid JSON on its own line — the JSONL invariant.
# REGRESSION risk: embedding a literal newline in a submit_command would split
# one row into two parse-failed lines.
def test_each_line_parses_as_json(ofat_nodes: tuple[Node, ...]) -> None:
    out = render_plan_jsonl(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    for line in out.splitlines():
        json.loads(line)


# CONTRACT: every preset row's submit_command bakes (dataset, seed, cluster).
# REGRESSION risk: forgetting `--cluster` would route every job to the env-var
# fallback ("pitzer"), silently submitting Cardinal-targeted plans wrong.
def test_preset_submit_commands_bake_dataset_seed_cluster(
    ofat_rows: list[dict[str, Any]],
) -> None:
    vgae = next(r for r in ofat_rows if r["name"] == "vgae")
    assert "--dataset set_01" in vgae["submit_command"]
    assert "--seed 42" in vgae["submit_command"]
    assert "--cluster cardinal" in vgae["submit_command"]


# CONTRACT: deps are listed by node-name, not by jid or shell var. The blueprint
# is data — the user/LLM honors deps by ordering, not by SLURM afterok plumbing.
def test_deps_are_node_names(ofat_rows: list[dict[str, Any]]) -> None:
    vgae_test = next(r for r in ofat_rows if r["name"] == "vgae-test")
    assert vgae_test["deps"] == ["vgae"]
    extract = next(r for r in ofat_rows if r["name"] == "extract-states")
    assert sorted(extract["deps"]) == ["focal", "vgae"]


# CONTRACT: command-mode rows have preset=None and a non-null command string;
# their submit_command uses --command (shell-quoted), not a preset path.
def test_command_row_shape(ofat_rows: list[dict[str, Any]]) -> None:
    extract = next(r for r in ofat_rows if r["name"] == "extract-states")
    assert extract["preset"] is None
    assert extract["command"] is not None
    assert "extract-fusion-states" in extract["command"]
    assert "--command " in extract["submit_command"]
    assert "extract-fusion-states" in extract["submit_command"]


# CONTRACT: command-mode rows do NOT get --skip-if-finished — they have no
# (group, variant) for the MLflow lookup.
def test_command_row_omits_skip_if_finished(ofat_rows: list[dict[str, Any]]) -> None:
    extract = next(r for r in ofat_rows if r["name"] == "extract-states")
    assert "--skip-if-finished" not in extract["submit_command"]


# CONTRACT: every preset row's submit_command carries --skip-if-finished by
# default. Re-running a partially-completed plan must short-circuit FINISHED
# nodes via MLflow without forcing the user to filter the JSONL.
def test_preset_rows_carry_skip_if_finished(ofat_rows: list[dict[str, Any]]) -> None:
    preset_rows = [r for r in ofat_rows if r["preset"] is not None]
    for row in preset_rows:
        assert "--skip-if-finished" in row["submit_command"], row["name"]


# CONTRACT: --depends-on is NOT auto-emitted. The user adds it for same-batch
# parallel queueing; sequential workflows don't need it. Auto-injection would
# turn `submit_command` into pipeline-specific orchestration logic, not a
# primitive — see .claude/rules/single-submission-primitive.md.
def test_depends_on_not_auto_emitted(ofat_rows: list[dict[str, Any]]) -> None:
    for row in ofat_rows:
        assert "--depends-on" not in row["submit_command"], row["name"]


# CONTRACT: test-action peers get --action test and the test-resource overrides
# (cpu/long/32GB/30min) baked into submit_command.
def test_test_peer_overrides(ofat_rows: list[dict[str, Any]]) -> None:
    vgae_test = next(r for r in ofat_rows if r["name"] == "vgae-test")
    cmd = vgae_test["submit_command"]
    assert "--action test" in cmd
    assert "--mode cpu" in cmd
    assert "--mem-gb 32" in cmd
    assert "--timeout-min 30" in cmd


# CONTRACT: render is deterministic. Same inputs → byte-identical JSONL.
def test_render_deterministic(ofat_nodes: tuple[Node, ...]) -> None:
    a = render_plan_jsonl(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    b = render_plan_jsonl(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    assert a == b


# CONTRACT: rows appear in topological order — upstream nodes precede their
# downstream consumers. The user/LLM can iterate top-to-bottom and never
# encounter an unresolved dep.
def test_rows_emit_in_topo_order(ofat_rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in ofat_rows:
        for dep in row["deps"]:
            assert dep in seen, f"{row['name']} depends on {dep!r} which appears later"
        seen.add(row["name"])
