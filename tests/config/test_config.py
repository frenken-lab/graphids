"""Config layer tests: topology, config tree validation, recipe schema."""

from __future__ import annotations

import graphlib

import pytest
from pydantic import ValidationError

from graphids.config.constants import (
    CONFIG_DIR,
    VALID_FUSION_METHODS,
    VALID_MODEL_TYPES,
    VALID_SCALES,
)
from graphids.config.jsonnet import render
from graphids.config.paths import dataset_names, load_catalog
from graphids.config.topology import (
    PIPELINE_TOPOLOGY,
    STAGE_DEPENDENCIES,
    STAGES,
)
from graphids.orchestrate.recipes import KDEntry, TrainingRunConfig

# ---------------------------------------------------------------------------
# Dataset catalog (configs/datasets/dataset_registry.json)
# ---------------------------------------------------------------------------


def test_catalog_loads_all_datasets():
    catalog = load_catalog()
    assert len(catalog) >= 6
    for name in ["hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04"]:
        assert name in catalog, f"Missing dataset: {name}"


def test_catalog_entries_have_required_fields():
    required = {
        "name",
        "domain",
        "csv_dir",
        "csv_columns",
        "train_subdir",
        "test_subdirs",
        "attack_types",
    }
    for name, entry in load_catalog().items():
        missing = required - set(entry.keys())
        assert not missing, f"Dataset '{name}' missing fields: {missing}"


def test_dataset_names_excludes_internal():
    names = dataset_names()
    assert all(not n.startswith("_") for n in names)


def test_catalog_test_subdirs_populated():
    """Every dataset must have at least one test subdir for evaluation."""
    for name, entry in load_catalog().items():
        assert entry["test_subdirs"], f"Dataset '{name}' has empty test_subdirs"


# ---------------------------------------------------------------------------
# DAG topology (from topology.py)
# ---------------------------------------------------------------------------


def test_fusion_has_dependencies():
    """Structural invariant: fusion is always downstream of training stages.

    Uses a set-cardinality check instead of asserting specific (model, stage)
    tuples so that adding/renaming upstream stages doesn't break the test.
    """
    assert "fusion" in STAGES
    assert STAGE_DEPENDENCIES["fusion"], "fusion stage has no declared dependencies"


def _topo_order() -> list[str]:
    graph = {s: {ds for _, ds in STAGE_DEPENDENCIES.get(s, [])} for s in STAGES}
    return list(graphlib.TopologicalSorter(graph).static_order())


def test_no_cycles():
    order = _topo_order()
    assert len(order) == len(STAGES)


@pytest.mark.parametrize(
    "before,after",
    [
        ("autoencoder", "supervised"),
        ("autoencoder", "fusion"),
        ("supervised", "fusion"),
    ],
)
def test_ordering(before, after):
    order = _topo_order()
    assert order.index(before) < order.index(after)


def test_default_stages_are_valid():
    bad = [s for s in PIPELINE_TOPOLOGY["default_stages"] if s not in STAGES]
    assert not bad, f"default_stages has unknown stages: {bad}"


def test_stages_have_identity_keys():
    for name, sdef in PIPELINE_TOPOLOGY["stages"].items():
        assert "identity_keys" in sdef, f"Stage '{name}' missing identity_keys"


# Config-tree file-existence checks removed — `graphids.config.topology` runs
# these same assertions at import time, so `from graphids.config import …` at
# the top of this module already exercises them. A missing file prevents
# collection (louder signal than a test failure).


# ---------------------------------------------------------------------------
# Recipe expansion + dagster planning
# ---------------------------------------------------------------------------


def test_resource_model_set_for_fusion_assets():
    """Fusion StageConfigs should have resource_model set to the fusion method."""
    from graphids.orchestrate.planning import enumerate_assets
    from graphids.orchestrate.recipes import expand_recipe_configs

    recipe = render(CONFIG_DIR / "recipes" / "ablation.jsonnet")
    specs = enumerate_assets(PIPELINE_TOPOLOGY, expand_recipe_configs(recipe))
    fusion_specs = [s for s in specs if s.stage == "fusion"]
    assert fusion_specs, "No fusion specs found in ablation recipe"
    for spec in fusion_specs:
        assert spec.resource_model, f"{spec.asset_name} has empty resource_model"


# ---------------------------------------------------------------------------
# TrainingRunConfig schema tests
# ---------------------------------------------------------------------------


class TestTrainingRunConfigCustom:
    """Tests for project-specific validation logic — NOT Pydantic built-ins.

    Pydantic's own machinery (``frozen=True``, ``extra="forbid"``, ``Literal``
    type coercion, required kwargs) is tested upstream by Pydantic; those
    tests are intentionally absent here.
    """

    def test_default_construction(self):
        cfg = TrainingRunConfig()
        assert cfg.model_type is None
        assert cfg.auxiliaries == ()

    def test_custom_stage_validator_rejects_unknown(self):
        """Project-specific ``@field_validator`` on ``stages`` — not a Literal."""
        with pytest.raises(ValidationError, match="Unknown stages"):
            TrainingRunConfig(stages=["autoencoder", "bogus"])

    def test_merge_preserves_non_overlaid_fields(self):
        """``merge()`` delegates to Pydantic but the inheritance semantics are ours."""
        base = TrainingRunConfig(scale="small", loss_fn="focal")
        merged = base.merge({"scale": "large"})
        assert merged.scale == "large"
        assert merged.loss_fn == "focal"  # inherited from base


class TestKDEntry:
    def test_valid(self):
        kd = KDEntry(alpha=0.5, teacher_scale="small")
        assert kd.type == "kd"
        assert kd.alpha == 0.5

    def test_teacher_config_optional_default(self):
        """teacher_config defaults to None; orchestration validates it, not the contract."""
        kd = KDEntry()
        assert kd.teacher_config is None

    def test_teacher_config_roundtrip(self):
        kd = KDEntry(teacher_config="baseline_large")
        assert kd.teacher_config == "baseline_large"


class TestRecipeRoundTrip:
    """Validate that all real recipe files parse through TrainingRunConfig."""

    @pytest.mark.parametrize(
        "recipe_name",
        sorted(
            p.name for p in (CONFIG_DIR / "recipes").glob("*.jsonnet") if not p.name.startswith("_")
        ),
    )
    def test_recipe_configs_validate(self, recipe_name):
        raw = render(CONFIG_DIR / "recipes" / recipe_name)
        # Recipes may use 'defaults' (old format) or 'selection' (new format)
        defaults = raw.get("defaults", {})
        if defaults:
            default_cfg = TrainingRunConfig(**defaults)
            for _name, overrides in raw.get("configs", {}).items():
                cfg = default_cfg.merge(overrides or {})
                assert cfg.scale in VALID_SCALES
