"""Tests for recipe expansion: _flatten_dict, expand_recipe_configs,
and trainer/resource override handling."""

from __future__ import annotations

import yaml

from graphids.config import CONFIG_DIR, expand_recipe_configs
from graphids.config.recipe_expand import _flatten_dict


class TestFlattenDict:
    def test_simple(self):
        assert _flatten_dict({"max_epochs": 2}, "trainer") == {
            "trainer.max_epochs": "2",
        }

    def test_nested(self):
        result = _flatten_dict({"callbacks": {"checkpoint": {"save_top_k": 1}}}, "trainer")
        assert result == {"trainer.callbacks.checkpoint.save_top_k": "1"}

    def test_bool_lowercased(self):
        assert _flatten_dict({"enable_progress_bar": False}, "trainer") == {
            "trainer.enable_progress_bar": "false",
        }

    def test_no_prefix(self):
        assert _flatten_dict({"a": 1}) == {"a": "1"}

    def test_empty(self):
        assert _flatten_dict({}) == {}


class TestRecipeOverrideExpansion:
    def test_trainer_overrides_flattened(self):
        raw = {
            "sweeps": [{"model_family": "gat", "stage": "normal", "scale": "small"}],
            "trainer_overrides": {"trainer.max_epochs": 2},
        }
        expanded = expand_recipe_configs(raw)
        assert expanded["trainer_overrides"] == {"trainer.max_epochs": "2"}

    def test_resource_overrides_passthrough(self):
        raw = {
            "sweeps": [{"model_family": "gat", "stage": "normal", "scale": "small"}],
            "resource_overrides": {"time": "0:15:00", "partition": "gpudebug"},
        }
        expanded = expand_recipe_configs(raw)
        assert expanded["resource_overrides"] == {
            "time": "0:15:00",
            "partition": "gpudebug",
        }

    def test_missing_overrides_default_empty(self):
        raw = {
            "sweeps": [{"model_family": "gat", "stage": "normal", "scale": "small"}],
        }
        expanded = expand_recipe_configs(raw)
        assert expanded["trainer_overrides"] == {}
        assert expanded["resource_overrides"] == {}

    def test_smoke_recipe_expands(self):
        path = CONFIG_DIR / "recipes" / "smoke_test.yaml"
        raw = yaml.safe_load(path.read_text())
        expanded = expand_recipe_configs(raw)
        assert expanded["trainer_overrides"] == {"trainer.max_epochs": "50"}
        assert expanded["resource_overrides"]["partition"] == "gpudebug"
        assert expanded["resource_overrides"]["time"] == "1:00:00"
        assert expanded["configs"], "No configs produced"


class TestKDAuxiliaryExpansion:
    def test_expand_sweep_with_kd_auxiliary(self):
        recipe = {
            "sweeps": [
                {
                    "model_family": "gat",
                    "stage": "normal",
                    "scale": ["small"],
                    "model_overrides": {"init_args": {"loss_fn": ["ce"]}},
                    "kd": {
                        "type": "kd",
                        "alpha": 0.5,
                        "teacher_config": "gat_normal_large",
                        "teacher_scale": "large",
                        "temperature": 2.0,
                    },
                }
            ]
        }
        expanded = expand_recipe_configs(
            recipe,
            valid_scales=frozenset({"small", "large"}),
            valid_fusion_methods=frozenset({"bandit", "dqn"}),
        )
        assert expanded["configs"]
        config = next(iter(expanded["configs"].values()))
        assert "auxiliaries" in config
        assert config["auxiliaries"][0]["type"] == "kd"
        assert config["auxiliaries"][0]["teacher_config"] == "gat_normal_large"
        assert config["auxiliaries"][0]["teacher_scale"] == "large"
