"""Tests for recipe expansion (Jsonnet-backed)."""

from __future__ import annotations

from graphids.config.constants import CONFIG_DIR
from graphids.config.jsonnet import render
from graphids.orchestrate.planning import expand_recipe_configs


class TestRecipeOverrideExpansion:
    def test_trainer_overrides_flattened(self):
        raw = {
            "sweeps": [{"model_family": "supervised", "stage": "supervised", "scale": "small"}],
            "trainer_overrides": {"trainer.max_epochs": 2},
        }
        expanded = expand_recipe_configs(raw)
        assert expanded["trainer_overrides"] == {"trainer.max_epochs": "2"}

    def test_resource_overrides_passthrough(self):
        raw = {
            "sweeps": [{"model_family": "supervised", "stage": "supervised", "scale": "small"}],
            "resource_overrides": {"time": "0:15:00", "partition": "gpudebug"},
        }
        expanded = expand_recipe_configs(raw)
        assert expanded["resource_overrides"] == {
            "time": "0:15:00",
            "partition": "gpudebug",
        }

    def test_missing_overrides_default_empty(self):
        raw = {
            "sweeps": [{"model_family": "supervised", "stage": "supervised", "scale": "small"}],
        }
        expanded = expand_recipe_configs(raw)
        assert expanded["trainer_overrides"] == {}
        assert expanded["resource_overrides"] == {}

    def test_smoke_recipe_expands(self):
        path = CONFIG_DIR / "recipes" / "smoke_test.jsonnet"
        raw = render(path)
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
                    "model_family": "supervised",
                    "stage": "supervised",
                    "scale": ["small"],
                    "model_overrides": {"init_args": {"loss_fn": ["ce"]}},
                    "kd": {
                        "type": "kd",
                        "alpha": 0.5,
                        "teacher_config": "gat_supervised_large",
                        "teacher_scale": "large",
                        "temperature": 2.0,
                    },
                }
            ]
        }
        # Public facade reads VALID_SCALES / VALID_FUSION_METHODS from
        # graphids.config.topology at import time — no kwargs.
        expanded = expand_recipe_configs(recipe)
        assert expanded["configs"]
        config = next(iter(expanded["configs"].values()))
        assert "auxiliaries" in config
        assert config["auxiliaries"][0]["type"] == "kd"
        assert config["auxiliaries"][0]["teacher_config"] == "gat_supervised_large"
        assert config["auxiliaries"][0]["teacher_scale"] == "large"
