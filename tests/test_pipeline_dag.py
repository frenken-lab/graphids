"""Pipeline DAG tests: topological order, dependency validation, stage routing."""

from __future__ import annotations

import graphlib

import pytest


# ---------------------------------------------------------------------------
# Topology from pipeline.yaml
# ---------------------------------------------------------------------------


class TestTopology:
    def test_topological_order_valid(self):
        """STAGE_DEPENDENCIES can be topologically sorted without cycles."""
        from graphids.config.constants import STAGES, STAGE_DEPENDENCIES

        # Build graph: stage -> set of upstream stages
        graph: dict[str, set[str]] = {}
        for stage in STAGES:
            deps = STAGE_DEPENDENCIES.get(stage, [])
            graph[stage] = {dep_stage for _, dep_stage in deps}

        sorter = graphlib.TopologicalSorter(graph)
        order = list(sorter.static_order())
        assert len(order) == len(STAGES)

    def test_autoencoder_before_curriculum(self):
        """Autoencoder appears before curriculum in topological order."""
        order = _topo_order()
        assert order.index("autoencoder") < order.index("curriculum")

    def test_autoencoder_before_fusion(self):
        """Autoencoder appears before fusion."""
        order = _topo_order()
        assert order.index("autoencoder") < order.index("fusion")

    def test_curriculum_before_fusion(self):
        """Curriculum appears before fusion."""
        order = _topo_order()
        assert order.index("curriculum") < order.index("fusion")

    def test_fusion_before_evaluation(self):
        """Fusion appears before evaluation."""
        order = _topo_order()
        assert order.index("fusion") < order.index("evaluation")

    def test_curriculum_before_temporal(self):
        """Curriculum appears before temporal."""
        order = _topo_order()
        assert order.index("curriculum") < order.index("temporal")

    def test_preprocess_is_root(self):
        """Preprocess has no dependencies."""
        from graphids.config.constants import STAGE_DEPENDENCIES

        assert "preprocess" not in STAGE_DEPENDENCIES


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_injected_cycle_raises(self):
        """Adding a cycle to the dependency graph raises CycleError."""
        graph = {
            "a": {"b"},
            "b": {"c"},
            "c": {"a"},  # cycle
        }
        sorter = graphlib.TopologicalSorter(graph)
        with pytest.raises(graphlib.CycleError):
            list(sorter.static_order())


# ---------------------------------------------------------------------------
# Stage model mapping
# ---------------------------------------------------------------------------


class TestStageModelMap:
    def test_autoencoder_uses_vgae(self):
        from graphids.config.constants import STAGE_MODEL_MAP

        assert STAGE_MODEL_MAP["autoencoder"] == "vgae"

    def test_curriculum_uses_gat(self):
        from graphids.config.constants import STAGE_MODEL_MAP

        assert STAGE_MODEL_MAP["curriculum"] == "gat"

    def test_fusion_uses_dqn(self):
        from graphids.config.constants import STAGE_MODEL_MAP

        assert STAGE_MODEL_MAP["fusion"] == "dqn"

    def test_temporal_uses_gat(self):
        from graphids.config.constants import STAGE_MODEL_MAP

        assert STAGE_MODEL_MAP["temporal"] == "gat"


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------


class TestStageDispatch:
    def test_unknown_stage_raises(self):
        """run_stage rejects unknown stage names."""
        from graphids.config import resolve
        from graphids.pipeline.stages import run_stage

        cfg = resolve()
        with pytest.raises(ValueError, match="Unknown stage"):
            run_stage(cfg, "nonexistent_stage")

    def test_all_stages_have_functions(self):
        """Every stage in STAGES has a corresponding function in STAGE_FNS."""
        from graphids.config.constants import STAGES
        from graphids.pipeline.stages import STAGE_FNS

        for stage in STAGES:
            if stage == "preprocess":
                continue  # preprocess is handled separately
            assert stage in STAGE_FNS, f"Stage '{stage}' missing from STAGE_FNS"


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------


class TestVariants:
    def test_variants_reference_valid_stages(self):
        """All stages listed in pipeline variants exist in STAGES."""
        import yaml

        from graphids.config.constants import CONFIG_DIR, STAGES

        pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
        for name, variant in pipeline["variants"].items():
            for stage in variant["stages"]:
                assert stage in STAGES, f"Variant '{name}' references unknown stage '{stage}'"

    def test_variants_reference_valid_scales(self):
        """All scales in variants are declared in the scales list."""
        import yaml

        from graphids.config.constants import CONFIG_DIR, VALID_SCALES

        pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
        for name, variant in pipeline["variants"].items():
            assert variant["scale"] in VALID_SCALES, (
                f"Variant '{name}' uses unknown scale '{variant['scale']}'"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topo_order() -> list[str]:
    """Return topologically sorted stage names from STAGE_DEPENDENCIES."""
    from graphids.config.constants import STAGES, STAGE_DEPENDENCIES

    graph: dict[str, set[str]] = {}
    for stage in STAGES:
        deps = STAGE_DEPENDENCIES.get(stage, [])
        graph[stage] = {dep_stage for _, dep_stage in deps}

    return list(graphlib.TopologicalSorter(graph).static_order())
