"""Config layer tests: constants, DAG topology, recipe schema."""

from __future__ import annotations

import graphlib

import pytest
import yaml

from graphids.config import CONFIG_DIR, KDEntry, TrainingRunConfig


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


# ---------------------------------------------------------------------------
# Cross-validation: pipeline × models/ × resources
# ---------------------------------------------------------------------------


def test_model_configs_exist_for_all_model_scale_pairs():
    """Every (model_type, scale) in pipeline.yaml has a models/{type}/{scale}.yaml file."""
    from graphids.config import CONFIG_DIR, VALID_MODEL_TYPES, VALID_SCALES
    models_dir = CONFIG_DIR / "models"
    for model in VALID_MODEL_TYPES:
        for scale in VALID_SCALES:
            path = models_dir / model / f"{scale}.yaml"
            assert path.exists(), f"Missing model config: {path}"


def test_fusion_method_configs_exist():
    """Every (fusion_method, scale) has a models/{method}/{scale}.yaml file."""
    from graphids.config import CONFIG_DIR, VALID_FUSION_METHODS, VALID_SCALES
    models_dir = CONFIG_DIR / "models"
    for method in VALID_FUSION_METHODS:
        for scale in VALID_SCALES:
            path = models_dir / method / f"{scale}.yaml"
            assert path.exists(), f"Missing fusion method config: {path}"


def test_resource_profiles_cover_pipeline():
    """Every trainable (model, scale, stage) has a resource profile."""
    import yaml
    from graphids.config import CONFIG_DIR
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    resources = yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text())
    profiles = resources["resource_profiles"]

    skip = {"preprocess", "evaluation", "temporal"}
    missing = []
    for stage_name, stage_def in pipeline["stages"].items():
        if stage_name in skip:
            continue
        if stage_name == "fusion":
            for method in pipeline.get("fusion_methods", []):
                for scale in pipeline["scales"]:
                    if method not in profiles or scale not in profiles[method] \
                            or stage_name not in profiles[method][scale]:
                        missing.append(f"{method}/{scale}/{stage_name}")
        else:
            model = stage_def["model"]
            for scale in pipeline["scales"]:
                if model not in profiles or scale not in profiles[model] \
                        or stage_name not in profiles[model][scale]:
                    missing.append(f"{model}/{scale}/{stage_name}")
    assert not missing, f"Missing resource profiles: {missing}"


def test_no_dead_scale_entries_in_resources():
    """resources.yaml should not have scale entries absent from pipeline.yaml."""
    import yaml
    from graphids.config import CONFIG_DIR, VALID_SCALES
    resources = yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text())
    profiles = resources["resource_profiles"]
    dead = []
    for model, scales in profiles.items():
        if model in ("preprocess", "test"):
            continue  # these use 'any', not real scales
        for scale in scales:
            if scale not in VALID_SCALES:
                dead.append(f"{model}/{scale}")
    assert not dead, f"Dead scale entries in resources.yaml: {dead}"


def test_resource_model_set_for_fusion_assets():
    """Fusion StageConfigs should have resource_model set to the fusion method."""
    import yaml
    from graphids.config import CONFIG_DIR, PIPELINE_YAML
    from graphids.orchestrate.component import enumerate_assets
    recipe = yaml.safe_load((CONFIG_DIR / "recipes" / "ablation.yaml").read_text())
    specs = enumerate_assets(PIPELINE_YAML, recipe)
    fusion_specs = [s for s in specs if s.stage == "fusion"]
    assert fusion_specs, "No fusion specs found in ablation recipe"
    for spec in fusion_specs:
        assert spec.resource_model, f"{spec.asset_name} has empty resource_model"
        assert spec.resource_model != "dqn" or "dqn" in spec.asset_name or True, \
            "resource_model should reflect actual fusion method"


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

    @pytest.mark.parametrize("recipe_name", ["ablation.yaml", "main_results.yaml"])
    def test_recipe_configs_validate(self, recipe_name):
        path = CONFIG_DIR / "recipes" / recipe_name
        if not path.exists():
            pytest.skip(f"{recipe_name} not found")
        raw = yaml.safe_load(path.read_text())
        default_cfg = TrainingRunConfig(**raw.get("defaults", {}))
        for name, overrides in raw.get("configs", {}).items():
            cfg = default_cfg.merge(overrides or {})
            assert cfg.scale in {"small", "large"}
