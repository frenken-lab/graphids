"""Config layer tests: topology, config tree validation, recipe schema."""

from __future__ import annotations

import graphlib

import pytest
import yaml

from graphids.config import (
    CONFIG_DIR,
    KDEntry,
    PIPELINE_YAML,
    STAGES,
    STAGE_DEPENDENCIES,
    TrainingRunConfig,
    VALID_FUSION_METHODS,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    dataset_names,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Dataset catalog (per-file configs in config/datasets/)
# ---------------------------------------------------------------------------


def test_catalog_loads_all_datasets():
    catalog = load_catalog()
    assert len(catalog) >= 6
    for name in ["hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04"]:
        assert name in catalog, f"Missing dataset: {name}"


def test_catalog_entries_have_required_fields():
    required = {"name", "csv_dir", "csv_columns", "train_subdir", "test_subdirs", "attack_types"}
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


def test_stages_and_dependencies():
    assert "autoencoder" in STAGES
    assert "fusion" in STAGES
    deps = STAGE_DEPENDENCIES["fusion"]
    assert ("vgae", "autoencoder") in deps
    assert ("gat", "curriculum") in deps


def _topo_order() -> list[str]:
    graph = {s: {ds for _, ds in STAGE_DEPENDENCIES.get(s, [])} for s in STAGES}
    return list(graphlib.TopologicalSorter(graph).static_order())


def test_no_cycles():
    order = _topo_order()
    assert len(order) == len(STAGES)


@pytest.mark.parametrize("before,after", [
    ("autoencoder", "curriculum"),
    ("autoencoder", "fusion"),
    ("curriculum", "fusion"),
])
def test_ordering(before, after):
    order = _topo_order()
    assert order.index(before) < order.index(after)


def test_default_stages_are_valid():
    bad = [s for s in PIPELINE_YAML["default_stages"] if s not in STAGES]
    assert not bad, f"default_stages has unknown stages: {bad}"


def test_stages_have_identity_keys():
    for name, sdef in PIPELINE_YAML["stages"].items():
        assert "identity_keys" in sdef, f"Stage '{name}' missing identity_keys"


# ---------------------------------------------------------------------------
# Config tree: models/, fusion/, resources/ (new modular structure)
# ---------------------------------------------------------------------------


def test_model_base_and_scale_configs_exist():
    """Every (model_type, scale) has models/{type}/base.yaml + scales/{scale}.yaml."""
    for model in VALID_MODEL_TYPES:
        base = CONFIG_DIR / "models" / model / "base.yaml"
        assert base.exists(), f"Missing model base config: {base}"
        for scale in VALID_SCALES:
            path = CONFIG_DIR / "models" / model / "scales" / f"{scale}.yaml"
            assert path.exists(), f"Missing model scale config: {path}"


def test_fusion_method_and_scale_configs_exist():
    """Fusion has base.yaml, per-method configs, and per-scale configs."""
    base = CONFIG_DIR / "fusion" / "base.yaml"
    assert base.exists(), f"Missing fusion base config: {base}"
    for method in VALID_FUSION_METHODS:
        path = CONFIG_DIR / "fusion" / "methods" / f"{method}.yaml"
        assert path.exists(), f"Missing fusion method config: {path}"
    for scale in VALID_SCALES:
        path = CONFIG_DIR / "fusion" / "scales" / f"{scale}.yaml"
        assert path.exists(), f"Missing fusion scale config: {path}"


def test_resource_profiles_exist_for_all_families():
    """Every model family (+ fusion) has a resource profile."""
    profiles_dir = CONFIG_DIR / "resources" / "profiles"
    for family in [*VALID_MODEL_TYPES, "fusion"]:
        path = profiles_dir / f"{family}.yaml"
        assert path.exists(), f"Missing resource profile: {path}"


def test_stage_yamls_exist():
    """Every stage in STAGES has a corresponding stages/{stage}.yaml."""
    for stage in STAGES:
        path = CONFIG_DIR / "stages" / f"{stage}.yaml"
        assert path.exists(), f"Missing stage config: {path}"


def test_analyze_stage_yamls_exist():
    """Analyze configs exist for model types that support analysis."""
    for model_type in ["vgae", "gat", "fusion"]:
        path = CONFIG_DIR / "stages" / f"analyze_{model_type}.yaml"
        assert path.exists(), f"Missing analyze config: {path}"


def test_defaults_yamls_exist():
    """Core default configs are present."""
    for name in ["trainer.yaml"]:
        path = CONFIG_DIR / "defaults" / name
        assert path.exists(), f"Missing defaults config: {path}"


# ---------------------------------------------------------------------------
# Recipe expansion + dagster planning
# ---------------------------------------------------------------------------


def test_resource_model_set_for_fusion_assets():
    """Fusion StageConfigs should have resource_model set to the fusion method."""
    from graphids.config import expand_recipe_configs
    from graphids.orchestrate.planning import enumerate_assets

    recipe = yaml.safe_load((CONFIG_DIR / "recipes" / "ablation.yaml").read_text())
    specs = enumerate_assets(PIPELINE_YAML, expand_recipe_configs(recipe))
    fusion_specs = [s for s in specs if s.stage == "fusion"]
    assert fusion_specs, "No fusion specs found in ablation recipe"
    for spec in fusion_specs:
        assert spec.resource_model, f"{spec.asset_name} has empty resource_model"


# ---------------------------------------------------------------------------
# TrainingRunConfig schema tests
# ---------------------------------------------------------------------------


class TestTrainingRunConfigDefaults:
    def test_default_construction(self):
        cfg = TrainingRunConfig()
        assert cfg.scale == "small"
        assert cfg.fusion_method == "bandit"
        assert cfg.stages == ("autoencoder", "curriculum", "fusion")
        assert cfg.auxiliaries == ()
        assert cfg.model_type is None

    def test_frozen(self):
        cfg = TrainingRunConfig()
        with pytest.raises(Exception):
            cfg.scale = "large"


class TestTrainingRunConfigValidation:
    def test_extra_forbid_on_construction(self):
        with pytest.raises(Exception, match="conv_typ"):
            TrainingRunConfig(conv_typ="gat")

    def test_extra_forbid_on_merge(self):
        cfg = TrainingRunConfig()
        with pytest.raises(Exception, match="scael"):
            cfg.merge({"scael": "large"})

    def test_invalid_scale(self):
        with pytest.raises(Exception, match="scale"):
            TrainingRunConfig(scale="huge")

    def test_invalid_fusion_method(self):
        with pytest.raises(Exception, match="fusion_method"):
            TrainingRunConfig(fusion_method="random")

    def test_invalid_conv_type(self):
        with pytest.raises(Exception, match="conv_type"):
            TrainingRunConfig(conv_type="transformer")

    def test_invalid_loss_fn(self):
        with pytest.raises(Exception, match="loss_fn"):
            TrainingRunConfig(loss_fn="mse")

    def test_invalid_model_type(self):
        with pytest.raises(Exception, match="model_type"):
            TrainingRunConfig(model_type="resnet")

    def test_unknown_stage(self):
        with pytest.raises(Exception, match="Unknown stages"):
            TrainingRunConfig(stages=["autoencoder", "bogus"])


class TestTrainingRunConfigMerge:
    def test_overlaid_field_wins(self):
        base = TrainingRunConfig(scale="small", loss_fn="focal")
        merged = base.merge({"scale": "large"})
        assert merged.scale == "large"
        assert merged.loss_fn == "focal"  # inherited

    def test_merge_validates(self):
        base = TrainingRunConfig()
        with pytest.raises(Exception):
            base.merge({"scale": "huge"})


class TestKDEntry:
    def test_extra_forbid(self):
        with pytest.raises(Exception, match="alppha"):
            KDEntry(alppha=0.7)

    def test_alpha_range(self):
        with pytest.raises(Exception, match="alpha"):
            KDEntry(alpha=1.5)

    def test_invalid_teacher_scale(self):
        with pytest.raises(Exception, match="teacher_scale"):
            KDEntry(teacher_scale="huge")

    def test_valid(self):
        kd = KDEntry(alpha=0.5, teacher_scale="small")
        assert kd.type == "kd"
        assert kd.alpha == 0.5


class TestTrainingRunConfigCoercion:
    def test_stages_list_to_tuple(self):
        cfg = TrainingRunConfig(stages=["autoencoder", "fusion"])
        assert isinstance(cfg.stages, tuple)

    def test_auxiliaries_dict_to_kdentry(self):
        cfg = TrainingRunConfig(
            auxiliaries=[{"type": "kd", "alpha": 0.5, "teacher_scale": "small"}],
        )
        assert isinstance(cfg.auxiliaries[0], KDEntry)
        assert cfg.auxiliaries[0].alpha == 0.5


class TestRecipeRoundTrip:
    """Validate that all real recipe files parse through TrainingRunConfig."""

    @pytest.mark.parametrize("recipe_name", ["ablation.yaml", "smoke_test.yaml", "final_eval.yaml"])
    def test_recipe_configs_validate(self, recipe_name):
        path = CONFIG_DIR / "recipes" / recipe_name
        if not path.exists():
            pytest.skip(f"{recipe_name} not found")
        raw = yaml.safe_load(path.read_text())
        # Recipes may use 'defaults' (old format) or 'selection' (new format)
        defaults = raw.get("defaults", {})
        if defaults:
            default_cfg = TrainingRunConfig(**defaults)
            for name, overrides in raw.get("configs", {}).items():
                cfg = default_cfg.merge(overrides or {})
                assert cfg.scale in VALID_SCALES
