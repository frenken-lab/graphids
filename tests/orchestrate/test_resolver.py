"""ConfigResolver end-to-end tests: resource overrides, trainer overrides,
stage overrides, cross-field validation over the rendered jsonnet config."""

from __future__ import annotations

import pytest

from graphids.orchestrate.contracts import TrainingContract, TrainingSpec
from graphids.orchestrate.resolve import ConfigResolver
from graphids.orchestrate.shared import StageConfig
from graphids.slurm import apply_resource_overrides, scale_resources
from graphids.slurm.resources import ResourceSpec


def _jpath(stage: str) -> str:
    return TrainingContract.resolve_jsonnet_path(stage)


# ---------------------------------------------------------------------------
# apply_resource_overrides
# ---------------------------------------------------------------------------


class TestApplyResourceOverrides:
    @pytest.fixture()
    def base_spec(self) -> ResourceSpec:
        return ResourceSpec(
            partition="gpu",
            time="4:00:00",
            mem="36G",
            cpus_per_task=4,
            num_workers=3,
            gres="gpu:1",
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
# Trainer / recipe override → jsonnet_tla → rendered dict
# ---------------------------------------------------------------------------


class TestTrainerOverrideFlow:
    @pytest.fixture()
    def resolver(self, tmp_path) -> ConfigResolver:
        return ConfigResolver(lake_root=str(tmp_path), user="test")

    def test_trainer_overrides_in_tla(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            jsonnet_path=_jpath("autoencoder"),
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        resolved = resolver.resolve(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            upstream_ckpts={},
        )
        assert resolved.spec.jsonnet_tla["trainer_overrides"]["trainer.max_epochs"] == "2"

    def test_empty_trainer_overrides_noop(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            jsonnet_path=_jpath("autoencoder"),
        )
        resolved = resolver.resolve(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            upstream_ckpts={},
        )
        assert resolved.spec.jsonnet_tla["trainer_overrides"] == {}

    def test_audit_records_trainer_overrides(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            jsonnet_path=_jpath("autoencoder"),
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        resolved = resolver.resolve(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            upstream_ckpts={},
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
            jsonnet_path=_jpath("autoencoder"),
            resource_overrides={"time": "0:15:00", "partition": "gpudebug"},
        )
        resolved = resolver.resolve(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            upstream_ckpts={},
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
            jsonnet_path=_jpath("autoencoder"),
            resource_overrides={"cpus_per_task": 2, "num_workers": 4},
        )
        with pytest.raises(ValueError, match="num_workers.*exceeds"):
            resolver.resolve(
                cfg,
                dataset="hcrl_sa",
                seed=42,
                upstream_ckpts={},
            )


# ---------------------------------------------------------------------------
# Rendered-config cross-field validation via resolver
# ---------------------------------------------------------------------------


class TestRenderedConfigValidation:
    @pytest.fixture()
    def resolver(self, tmp_path) -> ConfigResolver:
        return ConfigResolver(lake_root=str(tmp_path), user="test")

    def test_stage_overrides_applied(self, resolver):
        """stage_overrides propagate into jsonnet_tla under stage_overrides."""
        cfg = StageConfig(
            asset_name="test_supervised",
            stage="supervised",
            model_type="gat",
            scale="small",
            jsonnet_path=_jpath("supervised"),
            trainer_overrides={"trainer.max_epochs": "300"},
            stage_overrides={"data.init_args.max_epochs": "300"},
        )
        resolved = resolver.resolve(cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={})
        assert resolved.spec.jsonnet_tla["stage_overrides"]["data.init_args.max_epochs"] == "300"
        sources = {r.source for r in resolved.audit}
        assert "stage_override" in sources
        stages = {r.stage for r in resolved.audit if r.source == "stage_override"}
        assert "supervised" in stages

    def test_supervised_epoch_match_passes(self, resolver):
        cfg = StageConfig(
            asset_name="test_supervised",
            stage="supervised",
            model_type="gat",
            scale="small",
            jsonnet_path=_jpath("supervised"),
        )
        resolved = resolver.resolve(
            cfg,
            dataset="hcrl_sa",
            seed=42,
            upstream_ckpts={},
        )
        assert resolved.spec is not None

    def test_missing_jsonnet_path_raises(self, resolver):
        cfg = StageConfig(
            asset_name="test",
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            jsonnet_path="/nonexistent/path.jsonnet",
        )
        from graphids.config.jsonnet import JsonnetError

        with pytest.raises(JsonnetError):
            resolver.resolve(
                cfg,
                dataset="hcrl_sa",
                seed=42,
                upstream_ckpts={},
            )
