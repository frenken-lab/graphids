"""resolve_config end-to-end tests: overrides flow through to rendered config,
cross-field validation catches resource/config mismatches."""

from __future__ import annotations

import pytest

from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import ResolvedConfig
from graphids.slurm import apply_resource_overrides, scale_resources
from graphids.slurm.resources import ResourceSpec

# ---------------------------------------------------------------------------
# apply_resource_overrides (unit tests — no resolver needed)
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
# resolve_config end-to-end
# ---------------------------------------------------------------------------


def _resolve(tmp_path, **stage_kwargs):
    cfg = StageConfig(**stage_kwargs)
    return ResolvedConfig.resolve(
        cfg,
        lake_root=str(tmp_path),
        user="test",
        dataset="hcrl_sa",
        seed=42,
    )


class TestResolveConfig:
    def test_trainer_overrides_applied(self, tmp_path):
        """trainer_overrides flow through to rendered config."""
        resolved = _resolve(
            tmp_path,
            stage="autoencoder",
            model_type="vgae",
            scale="small",
            trainer_overrides={"trainer.max_epochs": "2"},
        )
        assert resolved.rendered["trainer"]["max_epochs"] == 2

    def test_empty_overrides_use_defaults(self, tmp_path):
        resolved = _resolve(
            tmp_path,
            stage="autoencoder",
            model_type="vgae",
            scale="small",
        )
        assert resolved.rendered["trainer"]["max_epochs"] > 0

    def test_stage_overrides_applied(self, tmp_path):
        """stage_overrides propagate into rendered config."""
        resolved = _resolve(
            tmp_path,
            stage="supervised",
            model_type="gat",
            scale="small",
            trainer_overrides={"trainer.max_epochs": "300"},
            stage_overrides={"data.init_args.max_epochs": "300"},
        )
        assert resolved.rendered["data"]["init_args"]["max_epochs"] == 300

    def test_cross_field_workers_exceeds_cpus_raises(self, tmp_path):
        """num_workers > cpus_per_task - 1 should fail validation."""
        with pytest.raises(ValueError, match="num_workers.*exceeds"):
            _resolve(
                tmp_path,
                stage="autoencoder",
                model_type="vgae",
                scale="small",
                resource_overrides={"cpus_per_task": 2, "num_workers": 4},
            )

    def test_supervised_resolves(self, tmp_path):
        resolved = _resolve(
            tmp_path,
            stage="supervised",
            model_type="gat",
            scale="small",
        )
        assert resolved.validated is not None
        assert resolved.rendered is not None
