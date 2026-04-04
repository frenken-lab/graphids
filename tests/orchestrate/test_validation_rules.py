"""Per-rule unit tests for ConfigResolver validation rules.

Each rule is exercised in isolation without going through the full
ConfigResolver.resolve() path — a new rule can land alongside a focused
test instead of a StageConfig fixture round-trip.
"""

from __future__ import annotations

from graphids.core.contracts import TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.resolve import (
    _RULES,
    _check_curriculum_epoch_sync,
    _check_fusion_rl_batch_size_override,
    _check_fusion_rl_batch_size_yaml,
    _check_gpu_partition_consistency,
    _check_num_workers_within_cpus,
    _check_yaml_num_workers_within_cpus,
    _is_curriculum,
    _is_fusion_rl,
    _is_gpu_stage,
)
from graphids.slurm import ResourceSpec

# ---------------------------------------------------------------------------
# Minimal fixture builders — flat helpers so each rule test reads tight
# ---------------------------------------------------------------------------


def _spec(stage="autoencoder", runtime_overrides=None) -> TrainingSpec:
    return TrainingSpec(
        stage=stage,
        model_family="vgae",
        scale="small",
        dataset="hcrl_sa",
        seed=42,
        run_dir="/tmp/test",
        config_files=(),
        runtime_overrides=runtime_overrides or {},
    )


def _res(cpus_per_task=4, num_workers=3, partition="gpu", gres="gpu:1") -> ResourceSpec:
    return ResourceSpec(
        partition=partition,
        time="4:00:00",
        mem="36G",
        cpus_per_task=cpus_per_task,
        num_workers=num_workers,
        gres=gres,
    )


def _cfg(stage="autoencoder", model_type="vgae") -> StageConfig:
    return StageConfig(
        asset_name="test",
        stage=stage,
        model_type=model_type,
        scale="small",
    )


# ---------------------------------------------------------------------------
# num_workers within cpus
# ---------------------------------------------------------------------------


class TestNumWorkersWithinCpus:
    def test_pass(self):
        assert _check_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4, num_workers=3), _cfg(), {},
        ) == []

    def test_fail(self):
        msgs = _check_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=2, num_workers=4), _cfg(), {},
        )
        assert len(msgs) == 1
        assert "num_workers=4" in msgs[0]
        assert "cpus_per_task-1=1" in msgs[0]


class TestYamlNumWorkersWithinCpus:
    def test_pass(self):
        merged = {"data": {"init_args": {"num_workers": 3}}}
        assert _check_yaml_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4), _cfg(), merged,
        ) == []

    def test_fail(self):
        merged = {"data": {"init_args": {"num_workers": 8}}}
        msgs = _check_yaml_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4), _cfg(), merged,
        )
        assert len(msgs) == 1
        assert "data.init_args.num_workers=8" in msgs[0]
        assert "in YAML exceeds" in msgs[0]

    def test_absent_is_pass(self):
        """Missing data.init_args.num_workers short-circuits — not an error."""
        assert _check_yaml_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4), _cfg(), {},
        ) == []


# ---------------------------------------------------------------------------
# GPU partition consistency + is_gpu_stage gate
# ---------------------------------------------------------------------------


class TestGpuPartitionConsistency:
    def test_pass(self):
        assert _check_gpu_partition_consistency(
            _spec(), _res(partition="gpu", gres="gpu:1"), _cfg(), {},
        ) == []

    def test_fail(self):
        msgs = _check_gpu_partition_consistency(
            _spec(), _res(partition="cpu", gres="gpu:1"), _cfg(), {},
        )
        assert len(msgs) == 1
        assert "not a GPU partition" in msgs[0]

    def test_is_gpu_stage_gates_evaluation(self):
        """Evaluation stage skips the GPU partition check (runs on CPU)."""
        assert _is_gpu_stage(_spec(), _res(gres="gpu:1"), _cfg(stage="evaluation"), {}) is False
        assert _is_gpu_stage(_spec(), _res(gres="gpu:1"), _cfg(stage="autoencoder"), {}) is True

    def test_is_gpu_stage_gates_no_gres(self):
        """A stage with no GRES doesn't need a GPU partition."""
        assert _is_gpu_stage(_spec(), _res(gres=""), _cfg(), {}) is False


# ---------------------------------------------------------------------------
# Curriculum epoch sync
# ---------------------------------------------------------------------------


class TestCurriculumEpochSync:
    def test_pass(self):
        merged = {
            "trainer": {"max_epochs": 300},
            "data": {"init_args": {"max_epochs": 300}},
        }
        assert _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"), merged,
        ) == []

    def test_mismatch(self):
        merged = {
            "trainer": {"max_epochs": 2},
            "data": {"init_args": {"max_epochs": 300}},
        }
        msgs = _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"), merged,
        )
        assert len(msgs) == 1
        assert "CurriculumDataModule.max_epochs=300" in msgs[0]
        assert "trainer.max_epochs=2" in msgs[0]

    def test_tolerates_missing_keys(self):
        """If either side is absent, no mismatch to flag."""
        assert _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"),
            {"trainer": {"max_epochs": 2}},
        ) == []
        assert _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"),
            {"data": {"init_args": {"max_epochs": 300}}},
        ) == []

    def test_is_curriculum_gate(self):
        assert _is_curriculum(_spec(), _res(), _cfg(stage="curriculum"), {}) is True
        assert _is_curriculum(_spec(), _res(), _cfg(stage="normal"), {}) is False


# ---------------------------------------------------------------------------
# Fusion RL batch_size rules
# ---------------------------------------------------------------------------


class TestFusionRlBatchSize:
    def test_override_pass(self):
        assert _check_fusion_rl_batch_size_override(
            _spec(stage="fusion"), _res(), _cfg(stage="fusion", model_type="dqn"), {},
        ) == []

    def test_override_fail(self):
        spec = _spec(
            stage="fusion",
            runtime_overrides={"data.init_args.batch_size": "64"},
        )
        msgs = _check_fusion_rl_batch_size_override(
            spec, _res(), _cfg(stage="fusion", model_type="dqn"), {},
        )
        assert len(msgs) == 1
        assert "batch_size override" in msgs[0]
        assert "'dqn'" in msgs[0]

    def test_yaml_warning(self):
        """YAML batch_size on RL fusion returns a message (severity=warning)."""
        merged = {"data": {"init_args": {"batch_size": 64}}}
        msgs = _check_fusion_rl_batch_size_yaml(
            _spec(stage="fusion"), _res(), _cfg(stage="fusion", model_type="bandit"), merged,
        )
        assert len(msgs) == 1
        assert "batch_size=64" in msgs[0]
        assert "'bandit'" in msgs[0]

    def test_is_fusion_rl_gate(self):
        """Non-RL fusion methods (mlp, weighted_avg) don't trigger RL rules."""
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="fusion", model_type="dqn"), {}) is True
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="fusion", model_type="bandit"), {}) is True
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="fusion", model_type="mlp"), {}) is False
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="normal", model_type="dqn"), {}) is False


# ---------------------------------------------------------------------------
# Rule registry severity contract
# ---------------------------------------------------------------------------


def test_fusion_rl_yaml_rule_is_warning_severity():
    """RL YAML batch_size is warning, not error — resolution still succeeds."""
    by_name = {r.name: r for r in _RULES}
    assert by_name["fusion_rl_batch_size_yaml"].severity == "warning"
    assert by_name["fusion_rl_batch_size_override"].severity == "error"
