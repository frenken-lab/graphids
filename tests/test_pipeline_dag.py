"""Pipeline DAG tests: topological order, stage dispatch, variant validity."""

from __future__ import annotations

import graphlib

import pytest


def _topo_order() -> list[str]:
    from graphids.config.constants import STAGES, STAGE_DEPENDENCIES
    graph = {s: {ds for _, ds in STAGE_DEPENDENCIES.get(s, [])} for s in STAGES}
    return list(graphlib.TopologicalSorter(graph).static_order())


def test_no_cycles():
    order = _topo_order()
    from graphids.config.constants import STAGES
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


def test_unknown_stage_raises():
    from graphids.config import resolve
    from graphids.pipeline.stages import run_stage
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage(resolve(), "nonexistent")


def test_all_stages_have_functions():
    from graphids.config.constants import STAGES
    from graphids.pipeline.stages import STAGE_FNS
    missing = [s for s in STAGES if s != "preprocess" and s not in STAGE_FNS]
    assert not missing, f"Missing stage functions: {missing}"


def test_default_stages_are_valid():
    import yaml
    from graphids.config.constants import CONFIG_DIR, STAGES
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    bad = [s for s in pipeline["default_stages"] if s not in STAGES]
    assert not bad, f"default_stages has unknown stages: {bad}"


def test_stages_have_identity_keys():
    import yaml
    from graphids.config.constants import CONFIG_DIR
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    for name, sdef in pipeline["stages"].items():
        assert "identity_keys" in sdef, f"Stage '{name}' missing identity_keys"
