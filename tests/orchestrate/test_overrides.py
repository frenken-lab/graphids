"""Tests for recipe-level trainer and resource override flow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graphids.config import CONFIG_DIR, expand_recipe_configs  # public API (passes valid_scales)
from graphids.config.recipe_expand import _flatten_dict  # internal, tested directly
from graphids.core.contracts import TrainingContract, TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import ConfigResolver, _deep_merge, _apply_dotted_overrides
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
    @pytest.fixture()
    def resolver(self, tmp_path) -> ConfigResolver:
        return ConfigResolver(lake_root=str(tmp_path), user="test")

    def test_trainer_overrides_in_runtime(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        resolved = resolver.resolve(
            cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
        )
        assert resolved.spec.runtime_overrides["trainer.max_epochs"] == "2"

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

    def test_empty_trainer_overrides_noop(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
        )
        resolved = resolver.resolve(
            cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
        )
        assert "trainer.max_epochs" not in resolved.spec.runtime_overrides

    def test_audit_records_trainer_overrides(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        resolved = resolver.resolve(
            cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
        )
        sources = {r.source for r in resolved.audit}
        assert "recipe_trainer" in sources
        keys = {r.key for r in resolved.audit}
        assert "trainer.max_epochs" in keys

    def test_audit_records_resource_overrides(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            resource_overrides={"time": "0:15:00", "partition": "gpudebug"},
        )
        resolved = resolver.resolve(
            cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
        )
        sources = {r.source for r in resolved.audit}
        assert "recipe_resource" in sources
        keys = {r.key for r in resolved.audit}
        assert "time" in keys
        assert "partition" in keys

    def test_cross_field_workers_exceeds_cpus_raises(self, resolver):
        """num_workers > cpus_per_task - 1 should fail validation."""
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            resource_overrides={"cpus_per_task": 2, "num_workers": 4},
        )
        with pytest.raises(ValueError, match="num_workers.*exceeds"):
            resolver.resolve(
                cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
            )


# ---------------------------------------------------------------------------
# YAML-aware validation
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_override(self):
        base = {"trainer": {"max_epochs": 300, "precision": "16-mixed"}}
        overlay = {"trainer": {"max_epochs": 2}}
        result = _deep_merge(base, overlay)
        assert result["trainer"]["max_epochs"] == 2
        assert result["trainer"]["precision"] == "16-mixed"

    def test_disjoint_keys(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_overlay(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}

    def test_non_dict_replaces_dict(self):
        assert _deep_merge({"a": {"b": 1}}, {"a": 42}) == {"a": 42}


class TestApplyDottedOverrides:
    def test_simple_dotted_key(self):
        merged = {"trainer": {"max_epochs": 300}}
        result = _apply_dotted_overrides(merged, {"trainer.max_epochs": "2"})
        assert result["trainer"]["max_epochs"] == "2"

    def test_creates_intermediate_dicts(self):
        result = _apply_dotted_overrides({}, {"a.b.c": "val"})
        assert result["a"]["b"]["c"] == "val"

    def test_no_overrides_is_noop(self):
        merged = {"x": 1}
        assert _apply_dotted_overrides(merged, {}) == {"x": 1}


class TestYAMLAwareValidation:
    @pytest.fixture()
    def resolver(self, tmp_path) -> ConfigResolver:
        return ConfigResolver(lake_root=str(tmp_path), user="test")

    def _write_yaml(self, tmp_path: Path, name: str, content: dict) -> str:
        p = tmp_path / name
        p.write_text(yaml.dump(content))
        return str(p)

    def test_merge_yaml_chain(self, resolver, tmp_path):
        f1 = self._write_yaml(tmp_path, "base.yaml", {
            "trainer": {"max_epochs": 300, "precision": "16-mixed"},
            "data": {"init_args": {"num_workers": 3}},
        })
        f2 = self._write_yaml(tmp_path, "stage.yaml", {
            "trainer": {"max_epochs": 100},
            "data": {"init_args": {"batch_size": 64}},
        })
        merged = resolver._merge_yaml_chain(
            (f1, f2), {"trainer.max_epochs": "2"},
        )
        assert merged["trainer"]["max_epochs"] == "2"
        assert merged["trainer"]["precision"] == "16-mixed"
        assert merged["data"]["init_args"]["num_workers"] == 3
        assert merged["data"]["init_args"]["batch_size"] == 64

    def test_curriculum_epoch_mismatch_raises(self, resolver, tmp_path):
        f1 = self._write_yaml(tmp_path, "trainer.yaml", {
            "trainer": {"max_epochs": 300},
        })
        f2 = self._write_yaml(tmp_path, "curriculum.yaml", {
            "data": {"init_args": {"max_epochs": 300, "num_workers": 2}},
        })
        cfg = StageConfig(
            asset_name="test_curriculum",
            stage="curriculum",
            model_type="gat",
            scale="small",
            config_files=(f1, f2),
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        with pytest.raises(ValueError, match="CurriculumDataModule.max_epochs.*!=.*trainer"):
            resolver.resolve(cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={})

    def test_curriculum_epoch_match_passes(self, resolver, tmp_path):
        f1 = self._write_yaml(tmp_path, "trainer.yaml", {
            "trainer": {"max_epochs": 300},
        })
        f2 = self._write_yaml(tmp_path, "curriculum.yaml", {
            "data": {"init_args": {"max_epochs": 300, "num_workers": 2}},
        })
        cfg = StageConfig(
            asset_name="test_curriculum",
            stage="curriculum",
            model_type="gat",
            scale="small",
            config_files=(f1, f2),
        )
        resolved = resolver.resolve(
            cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
        )
        assert resolved.spec is not None

    def test_yaml_num_workers_exceeds_cpus_raises(self, resolver, tmp_path):
        f1 = self._write_yaml(tmp_path, "stage.yaml", {
            "data": {"init_args": {"num_workers": 8}},
        })
        cfg = StageConfig(
            asset_name="test_workers",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            config_files=(f1,),
            resource_overrides={"cpus_per_task": 4, "num_workers": 3},
        )
        with pytest.raises(ValueError, match="num_workers.*in YAML exceeds"):
            resolver.resolve(cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={})

    def test_missing_config_files_skipped(self, resolver):
        """Missing YAML files are skipped, not crashed on."""
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            config_files=("/nonexistent/path.yaml",),
        )
        resolved = resolver.resolve(
            cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
        )
        assert resolved.spec is not None
