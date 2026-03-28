"""Config layer tests: constants, DAG topology."""

from __future__ import annotations

import graphlib

import pytest


def test_stages_and_dependencies():
    """pipeline.yaml parsed: stages exist, DAG is valid."""
    from graphids.config import STAGES, STAGE_DEPENDENCIES
    assert "autoencoder" in STAGES
    assert "fusion" in STAGES
    deps = STAGE_DEPENDENCIES["fusion"]
    assert ("vgae", "autoencoder") in deps
    assert ("gat", "curriculum") in deps


# ---------------------------------------------------------------------------
# DAG topology (from test_pipeline_dag.py)
# ---------------------------------------------------------------------------


def _topo_order() -> list[str]:
    from graphids.config import STAGES, STAGE_DEPENDENCIES
    graph = {s: {ds for _, ds in STAGE_DEPENDENCIES.get(s, [])} for s in STAGES}
    return list(graphlib.TopologicalSorter(graph).static_order())


def test_no_cycles():
    order = _topo_order()
    from graphids.config import STAGES
    assert len(order) == len(STAGES)


@pytest.mark.parametrize("before,after", [
    ("autoencoder", "curriculum"),
    ("autoencoder", "fusion"),
    ("curriculum", "fusion"),
    ("fusion", "evaluation"),
])
def test_ordering(before, after):
    order = _topo_order()
    assert order.index(before) < order.index(after)


def test_default_stages_are_valid():
    import yaml
    from graphids.config import CONFIG_DIR, STAGES
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    bad = [s for s in pipeline["default_stages"] if s not in STAGES]
    assert not bad, f"default_stages has unknown stages: {bad}"


def test_stages_have_identity_keys():
    import yaml
    from graphids.config import CONFIG_DIR
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    for name, sdef in pipeline["stages"].items():
        assert "identity_keys" in sdef, f"Stage '{name}' missing identity_keys"
