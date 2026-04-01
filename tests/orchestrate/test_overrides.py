"""Tests for recipe-level trainer and resource override flow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graphids.config import CONFIG_DIR, expand_recipe_configs  # public API (passes valid_scales)
from graphids.config.recipe_expand import _flatten_dict  # internal, tested directly
from graphids.core.contracts import TrainingContract, TrainingSpec
from graphids.orchestrate.execution import training_spec
from graphids.orchestrate.planning import StageConfig
from graphids.slurm import (
    ResourceSpec,
    apply_resource_overrides,
    scale_resources,
)


# ---------------------------------------------------------------------------
# _flatten_dict
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Recipe expansion
# ---------------------------------------------------------------------------


class TestRecipeOverrideExpansion:
    def test_trainer_overrides_flattened(self):
        raw = {
            "sweeps": [{"model_family": "gat", "stage": "normal", "scale": "small"}],
            "trainer_overrides": {"max_epochs": 2},
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
        assert expanded["trainer_overrides"] == {"trainer.max_epochs": "2"}
        assert expanded["resource_overrides"]["partition"] == "gpudebug"
        assert expanded["resource_overrides"]["time"] == "0:15:00"
        assert expanded["configs"], "No configs produced"


# ---------------------------------------------------------------------------
# apply_resource_overrides
# ---------------------------------------------------------------------------


class TestApplyResourceOverrides:
    @pytest.fixture()
    def base_spec(self) -> ResourceSpec:
        return ResourceSpec(
            partition="gpu", time="4:00:00", mem="36G",
            cpus_per_task=4, num_workers=3, gres="gpu:1",
        )

    def test_changes_partition(self, base_spec):
        patched = apply_resource_overrides(base_spec, {"partition": "gpudebug"})
        assert patched.partition == "gpudebug"
        assert patched.time == "4:00:00"

    def test_changes_time(self, base_spec):
        patched = apply_resource_overrides(base_spec, {"time": "0:15:00"})
        assert patched.time == "0:15:00"

    def test_unknown_key_raises(self, base_spec):
        with pytest.raises(ValueError, match="Unknown resource override"):
            apply_resource_overrides(base_spec, {"partiton": "gpudebug"})

    def test_empty_is_noop(self, base_spec):
        assert apply_resource_overrides(base_spec, {}) is base_spec

    def test_override_then_scale(self, base_spec):
        """Recipe overrides apply before retry scaling."""
        patched = apply_resource_overrides(base_spec, {"time": "0:15:00"})
        scaled = scale_resources(patched, "TIMEOUT")
        assert scaled.time_minutes > 15


# ---------------------------------------------------------------------------
# Trainer overrides → runtime_overrides → CLI args
# ---------------------------------------------------------------------------


class TestTrainerOverrideFlow:
    def test_trainer_overrides_in_runtime(self):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        spec = training_spec(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            run_directory="/tmp/test",
            run_directory_path=Path("/tmp/nonexistent"),
            upstream_ckpts={},
        )
        assert spec.runtime_overrides["trainer.max_epochs"] == "2"

    def test_trainer_overrides_become_cli_args(self):
        spec = TrainingSpec(
            stage="autoencoder",
            model_family="vgae",
            scale="small",
            dataset="hcrl_sa",
            seed=42,
            run_dir="/tmp/test",
            config_files=(),
            runtime_overrides={"trainer.max_epochs": "2"},
        )
        cli_args = TrainingContract.to_cli_overrides(spec)
        assert "--trainer.max_epochs=2" in cli_args

    def test_empty_trainer_overrides_noop(self):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
        )
        spec = training_spec(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            run_directory="/tmp/test",
            run_directory_path=Path("/tmp/nonexistent"),
            upstream_ckpts={},
        )
        assert "trainer.max_epochs" not in spec.runtime_overrides
