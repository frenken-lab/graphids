"""Tests for ``graphids/slurm/dag.py`` — Node dataclass + plan parser + topo helpers.

Pure tests — no SLURM, no MLflow. The OFAT plan jsonnet is rendered as the
integration fixture so we test against the actual shipped topology.
"""

from __future__ import annotations

import pytest

from graphids.config.jsonnet import render
from graphids.slurm.dag import Node, filter_with_upstream, parse_plan, toposort


@pytest.fixture(scope="module")
def ofat_nodes() -> tuple[Node, ...]:
    return parse_plan(render("configs/plans/ofat.jsonnet", tla={"dataset": "set_01", "seed": 42}))


# CONTRACT: Node enforces preset XOR command.
def test_node_rejects_neither_preset_nor_command():
    with pytest.raises(ValueError, match="exactly one"):
        Node(name="x")


def test_node_rejects_both_preset_and_command():
    with pytest.raises(ValueError, match="exactly one"):
        Node(
            name="x",
            preset_path="a/b.jsonnet",
            group="a",
            variant="b",
            command="echo hi",
        )


# CONTRACT: preset nodes derive group + variant from `<group>/<variant>.jsonnet`.
def test_preset_node_derives_group_variant_from_path():
    n = Node(name="x", preset_path="a/b.jsonnet")
    assert (n.group, n.variant) == ("a", "b")


# CONTRACT: explicit group / variant override path inference (off-convention paths).
def test_preset_node_explicit_overrides_inference():
    n = Node(name="x", preset_path="weird/path.jsonnet", group="A", variant="B")
    assert (n.group, n.variant) == ("A", "B")


# REGRESSION: preset path that can't be parsed (no slash) must fail validation
# rather than silently default to (None, None) and break MLflow lookups later.
def test_preset_node_unparseable_path_raises():
    with pytest.raises(ValueError, match="off-convention"):
        Node(name="x", preset_path="bare.jsonnet")


# CONTRACT: pydantic uses `preset` as the on-disk alias for `preset_path`,
# matching the jsonnet plan field name.
def test_node_accepts_preset_alias():
    n = Node.model_validate({"name": "x", "preset": "a/b.jsonnet", "group": "a", "variant": "b"})
    assert n.preset_path == "a/b.jsonnet"


# CONTRACT: parse_plan rejects unknown fields. Catches plan typos before run.
def test_parse_plan_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_plan({"nodes": [{"name": "x", "command": "echo", "typo": 1}]})


def test_parse_plan_rejects_missing_nodes_key():
    with pytest.raises(ValueError, match="nodes"):
        parse_plan({"foo": []})


# CONTRACT: toposort orders by deps; cycles + missing deps raise.
def test_toposort_orders_by_deps():
    a = Node(name="a", command="echo a")
    b = Node(name="b", command="echo b", deps=("a",))
    c = Node(name="c", command="echo c", deps=("b",))
    order = [n.name for n in toposort((c, a, b))]
    assert order.index("a") < order.index("b") < order.index("c")


def test_toposort_unknown_dep_raises():
    n = Node(name="x", command="echo", deps=("ghost",))
    with pytest.raises(RuntimeError, match="ghost"):
        toposort((n,))


# DIFFERENTIAL: filter_with_upstream walks transitive deps.
def test_filter_with_upstream_walks_transitive(ofat_nodes: tuple[Node, ...]) -> None:
    selected = filter_with_upstream(ofat_nodes, ("bandit",))
    assert {n.name for n in selected} == {"vgae", "focal", "extract-states", "bandit"}


def test_filter_with_upstream_unknown_raises(ofat_nodes: tuple[Node, ...]) -> None:
    with pytest.raises(ValueError, match="unknown"):
        filter_with_upstream(ofat_nodes, ("nope",))


# CONTRACT: shipped OFAT plan parses cleanly + has the expected node count.
def test_ofat_plan_loads(ofat_nodes: tuple[Node, ...]) -> None:
    # 15 fits + 15 tests + 1 command = 31.
    assert len(ofat_nodes) == 31
    by_name = {n.name: n for n in ofat_nodes}
    # Cross-stage deps that motivate the topology.
    assert by_name["curriculum_vgae"].deps == ("vgae",)
    assert by_name["extract-states"].deps == ("vgae", "focal")
    assert by_name["bandit"].deps == ("extract-states",)
    # Every fit has a paired test.
    fit_names = {n.name for n in ofat_nodes if n.action == "fit" and n.preset_path}
    test_names = {n.name for n in ofat_nodes if n.action == "test"}
    assert {f + "-test" for f in fit_names} == test_names
