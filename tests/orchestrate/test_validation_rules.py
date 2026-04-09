"""Cross-field validation tests — exercises _validate_cross_fields directly."""

from __future__ import annotations

import pytest

from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import _validate_cross_fields
from graphids.slurm.resources import ResourceSpec


def _res(cpus_per_task=4, num_workers=3, **kw) -> ResourceSpec:
    defaults = {"partition": "gpu", "time": "4:00:00", "mem": "36G", "gres": "gpu:1"}
    return ResourceSpec(cpus_per_task=cpus_per_task, num_workers=num_workers, **{**defaults, **kw})


def _cfg(stage="autoencoder", model_type="vgae") -> StageConfig:
    return StageConfig(stage=stage, model_type=model_type, scale="small")


# ---------------------------------------------------------------------------
# num_workers within cpus
# ---------------------------------------------------------------------------


class TestNumWorkersWithinCpus:
    def test_pass(self):
        _validate_cross_fields(_cfg(), _res(cpus_per_task=4, num_workers=3), {})

    def test_profile_exceeds(self):
        with pytest.raises(ValueError, match=r"num_workers=4.*cpus_per_task-1=1"):
            _validate_cross_fields(_cfg(), _res(cpus_per_task=2, num_workers=4), {})

    def test_rendered_exceeds(self):
        rendered = {"data": {"init_args": {"num_workers": 8}}}
        with pytest.raises(ValueError, match=r"num_workers=8.*in rendered config"):
            _validate_cross_fields(_cfg(), _res(cpus_per_task=4), rendered)

    def test_rendered_absent_passes(self):
        _validate_cross_fields(_cfg(), _res(cpus_per_task=4), {})


# ---------------------------------------------------------------------------
# Datamodule epoch sync (supervised only)
# ---------------------------------------------------------------------------


class TestDatamoduleEpochSync:
    def test_match_passes(self):
        rendered = {"trainer": {"max_epochs": 300}, "data": {"init_args": {"max_epochs": 300}}}
        _validate_cross_fields(_cfg(stage="supervised"), _res(), rendered)

    def test_mismatch_raises(self):
        rendered = {"trainer": {"max_epochs": 2}, "data": {"init_args": {"max_epochs": 300}}}
        with pytest.raises(ValueError, match=r"max_epochs=300.*max_epochs=2"):
            _validate_cross_fields(_cfg(stage="supervised"), _res(), rendered)

    def test_missing_key_passes(self):
        """If either side is absent, no mismatch to flag."""
        _validate_cross_fields(_cfg(stage="supervised"), _res(), {"trainer": {"max_epochs": 2}})
        _validate_cross_fields(
            _cfg(stage="supervised"), _res(), {"data": {"init_args": {"max_epochs": 300}}}
        )

    def test_autoencoder_skips_check(self):
        """Epoch sync only applies to supervised stage."""
        rendered = {"trainer": {"max_epochs": 2}, "data": {"init_args": {"max_epochs": 300}}}
        _validate_cross_fields(_cfg(stage="autoencoder"), _res(), rendered)
