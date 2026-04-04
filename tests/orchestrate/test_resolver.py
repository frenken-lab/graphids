"""ConfigResolver end-to-end tests: resource overrides, trainer overrides,
stage overrides, YAML-aware cross-field validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graphids.core.contracts import TrainingContract, TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import ConfigResolver
from graphids.slurm import (
    ResourceSpec,
    apply_resource_overrides,
    scale_resources,
)


def _write_yaml(tmp_path: Path, name: str, content: dict) -> str:
    p = tmp_path / name
    p.write_text(yaml.dump(content))
    return str(p)


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
# Trainer / recipe override → runtime_overrides → CLI args
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
        overrides = TrainingContract.to_override_dict(spec)
        assert overrides["trainer.max_epochs"] == "2"

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
# YAML-aware cross-field validation via resolver
# ---------------------------------------------------------------------------


class TestYAMLAwareValidation:
    @pytest.fixture()
    def resolver(self, tmp_path) -> ConfigResolver:
        return ConfigResolver(lake_root=str(tmp_path), user="test")

    def test_stage_overrides_applied(self, resolver, tmp_path):
        """stage_overrides are merged into runtime_overrides for matching stage."""
        f1 = _write_yaml(tmp_path, "trainer.yaml", {
            "trainer": {"max_epochs": 300},
        })
        f2 = _write_yaml(tmp_path, "curriculum.yaml", {
            "data": {"init_args": {"max_epochs": 300, "num_workers": 2}},
        })
        cfg = StageConfig(
            asset_name="test_curriculum",
            stage="curriculum",
            model_type="gat",
            scale="small",
            config_files=(f1, f2),
            trainer_overrides={"trainer.max_epochs": "2"},
            stage_overrides={"data.init_args.max_epochs": "2"},
        )
        resolved = resolver.resolve(cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={})
        assert resolved.spec.runtime_overrides["data.init_args.max_epochs"] == "2"
        sources = {r.source for r in resolved.audit}
        assert "stage_override" in sources
        stages = {r.stage for r in resolved.audit if r.source == "stage_override"}
        assert "curriculum" in stages

    def test_curriculum_epoch_match_passes(self, resolver, tmp_path):
        f1 = _write_yaml(tmp_path, "trainer.yaml", {
            "trainer": {"max_epochs": 300},
        })
        f2 = _write_yaml(tmp_path, "curriculum.yaml", {
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
        f1 = _write_yaml(tmp_path, "stage.yaml", {
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

    def test_missing_config_files_raises(self, resolver):
        """Missing YAML files raise FileNotFoundError (session 7 hardening)."""
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            config_files=("/nonexistent/path.yaml",),
        )
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            resolver.resolve(
                cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={},
            )
