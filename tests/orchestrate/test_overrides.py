"""Tests for recipe-level trainer and resource override flow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graphids.config import CONFIG_DIR, expand_recipe_configs  # public API (passes valid_scales)
from graphids.config.recipe_expand import _flatten_dict  # internal, tested directly
from graphids.core.contracts import TrainingContract, TrainingSpec
from graphids.orchestrate.planning import StageConfig
from graphids.config.yaml_utils import apply_dotted_overrides, deep_merge
from graphids.orchestrate.resolve import (
    ConfigResolver,
    ValidationRule,
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
# YAML-aware validation
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_override(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_override(self):
        base = {"trainer": {"max_epochs": 300, "precision": "16-mixed"}}
        overlay = {"trainer": {"max_epochs": 2}}
        result = deep_merge(base, overlay)
        assert result["trainer"]["max_epochs"] == 2
        assert result["trainer"]["precision"] == "16-mixed"

    def test_disjoint_keys(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_overlay(self):
        base = {"a": 1}
        assert deep_merge(base, {}) == {"a": 1}

    def test_non_dict_replaces_dict(self):
        assert deep_merge({"a": {"b": 1}}, {"a": 42}) == {"a": 42}


class TestApplyDottedOverrides:
    def test_simple_dotted_key(self):
        merged = {"trainer": {"max_epochs": 300}}
        result = apply_dotted_overrides(merged, {"trainer.max_epochs": "2"})
        assert result["trainer"]["max_epochs"] == "2"

    def test_creates_intermediate_dicts(self):
        result = apply_dotted_overrides({}, {"a.b.c": "val"})
        assert result["a"]["b"]["c"] == "val"

    def test_no_overrides_is_noop(self):
        merged = {"x": 1}
        assert apply_dotted_overrides(merged, {}) == {"x": 1}


class TestYAMLAwareValidation:
    @pytest.fixture()
    def resolver(self, tmp_path) -> ConfigResolver:
        return ConfigResolver(lake_root=str(tmp_path), user="test")

    def _write_yaml(self, tmp_path: Path, name: str, content: dict) -> str:
        p = tmp_path / name
        p.write_text(yaml.dump(content))
        return str(p)

    def test_merge_yaml_chain(self, tmp_path):
        from graphids.config.yaml_utils import merge_yaml_chain

        f1 = self._write_yaml(tmp_path, "base.yaml", {
            "trainer": {"max_epochs": 300, "precision": "16-mixed"},
            "data": {"init_args": {"num_workers": 3}},
        })
        f2 = self._write_yaml(tmp_path, "stage.yaml", {
            "trainer": {"max_epochs": 100},
            "data": {"init_args": {"batch_size": 64}},
        })
        merged = merge_yaml_chain(
            (f1, f2), {"trainer.max_epochs": "2"},
        )
        assert merged["trainer"]["max_epochs"] == "2"
        assert merged["trainer"]["precision"] == "16-mixed"
        assert merged["data"]["init_args"]["num_workers"] == 3
        assert merged["data"]["init_args"]["batch_size"] == 64

    def test_stage_overrides_applied(self, resolver, tmp_path):
        """stage_overrides are merged into runtime_overrides for matching stage."""
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
            stage_overrides={"data.init_args.max_epochs": "2"},
        )
        resolved = resolver.resolve(cfg, dataset="hcrl_sa", seed=42, upstream_ckpts={})
        assert resolved.spec.runtime_overrides["data.init_args.max_epochs"] == "2"
        sources = {r.source for r in resolved.audit}
        assert "stage_override" in sources
        stages = {r.stage for r in resolved.audit if r.source == "stage_override"}
        assert "curriculum" in stages

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


# ---------------------------------------------------------------------------
# Per-rule unit tests — exercise each ValidationRule in isolation without
# going through the full ConfigResolver.resolve() path. Lets a new rule land
# alongside a focused test instead of a full StageConfig fixture round-trip.
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


class TestValidationRules:
    def test_rules_list_has_unique_names(self):
        """_RULES entries must have unique names (registry integrity)."""
        names = [r.name for r in _RULES]
        assert len(names) == len(set(names)), f"Duplicate rule names: {names}"

    def test_num_workers_within_cpus_pass(self):
        assert _check_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4, num_workers=3), _cfg(), {},
        ) == []

    def test_num_workers_within_cpus_fail(self):
        msgs = _check_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=2, num_workers=4), _cfg(), {},
        )
        assert len(msgs) == 1
        assert "num_workers=4" in msgs[0]
        assert "cpus_per_task-1=1" in msgs[0]

    def test_yaml_num_workers_within_cpus_pass(self):
        merged = {"data": {"init_args": {"num_workers": 3}}}
        assert _check_yaml_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4), _cfg(), merged,
        ) == []

    def test_yaml_num_workers_within_cpus_fail(self):
        merged = {"data": {"init_args": {"num_workers": 8}}}
        msgs = _check_yaml_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4), _cfg(), merged,
        )
        assert len(msgs) == 1
        assert "data.init_args.num_workers=8" in msgs[0]
        assert "in YAML exceeds" in msgs[0]

    def test_yaml_num_workers_absent_is_pass(self):
        """Missing data.init_args.num_workers short-circuits — not an error."""
        assert _check_yaml_num_workers_within_cpus(
            _spec(), _res(cpus_per_task=4), _cfg(), {},
        ) == []

    def test_gpu_partition_consistency_pass(self):
        assert _check_gpu_partition_consistency(
            _spec(), _res(partition="gpu", gres="gpu:1"), _cfg(), {},
        ) == []

    def test_gpu_partition_consistency_fail(self):
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

    def test_curriculum_epoch_sync_pass(self):
        merged = {
            "trainer": {"max_epochs": 300},
            "data": {"init_args": {"max_epochs": 300}},
        }
        assert _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"), merged,
        ) == []

    def test_curriculum_epoch_sync_mismatch(self):
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

    def test_curriculum_epoch_sync_tolerates_missing_keys(self):
        """If either side is absent, no mismatch to flag."""
        assert _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"),
            {"trainer": {"max_epochs": 2}},
        ) == []
        assert _check_curriculum_epoch_sync(
            _spec(stage="curriculum"), _res(), _cfg(stage="curriculum"),
            {"data": {"init_args": {"max_epochs": 300}}},
        ) == []

    def test_is_curriculum_only_matches_curriculum_stage(self):
        assert _is_curriculum(_spec(), _res(), _cfg(stage="curriculum"), {}) is True
        assert _is_curriculum(_spec(), _res(), _cfg(stage="normal"), {}) is False

    def test_fusion_rl_batch_size_override_pass(self):
        assert _check_fusion_rl_batch_size_override(
            _spec(stage="fusion"), _res(), _cfg(stage="fusion", model_type="dqn"), {},
        ) == []

    def test_fusion_rl_batch_size_override_fail(self):
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

    def test_fusion_rl_batch_size_yaml_warning(self):
        """YAML batch_size on RL fusion returns a message (severity=warning)."""
        merged = {"data": {"init_args": {"batch_size": 64}}}
        msgs = _check_fusion_rl_batch_size_yaml(
            _spec(stage="fusion"), _res(), _cfg(stage="fusion", model_type="bandit"), merged,
        )
        assert len(msgs) == 1
        assert "batch_size=64" in msgs[0]
        assert "'bandit'" in msgs[0]

    def test_is_fusion_rl_gates_non_rl_fusion(self):
        """Non-RL fusion methods (mlp, weighted_avg) don't trigger RL rules."""
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="fusion", model_type="dqn"), {}) is True
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="fusion", model_type="bandit"), {}) is True
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="fusion", model_type="mlp"), {}) is False
        assert _is_fusion_rl(_spec(), _res(), _cfg(stage="normal", model_type="dqn"), {}) is False

    def test_validation_rule_is_frozen(self):
        """ValidationRule is a frozen dataclass — mutation should raise."""
        rule = _RULES[0]
        with pytest.raises((AttributeError, Exception)):
            rule.name = "mutated"  # type: ignore[misc]

    def test_fusion_rl_yaml_rule_is_warning_severity(self):
        """RL YAML batch_size is warning, not error — resolution still succeeds."""
        by_name = {r.name: r for r in _RULES}
        assert by_name["fusion_rl_batch_size_yaml"].severity == "warning"
        assert by_name["fusion_rl_batch_size_override"].severity == "error"


# ---------------------------------------------------------------------------
# KD teacher resolution — explicit teacher_config (replaces silent rewiring)
# ---------------------------------------------------------------------------


class TestKDTeacherResolution:
    """Covers _resolve_kd_teachers via enumerate_assets. Replaces the old
    ``teacher_scale + first-match`` inference with explicit teacher_config
    naming. See docs/reference/orchestration-risks.md item #2."""

    @pytest.fixture()
    def pipeline(self):
        from graphids.config import PIPELINE_YAML
        return PIPELINE_YAML

    def _recipe(self, configs: dict) -> dict:
        """Build a minimal expanded-recipe dict around a set of configs."""
        return {
            "defaults": {},
            "configs": configs,
            "sweep": {"seeds": [42]},
            "trainer_overrides": {},
            "stage_overrides": {},
            "resource_overrides": {},
        }

    def _student(self, teacher_config: str | None, scale: str = "small") -> dict:
        kd: dict = {"type": "kd", "alpha": 0.5, "teacher_scale": "large"}
        if teacher_config is not None:
            kd["teacher_config"] = teacher_config
        return {
            "stages": ["normal"],
            "scale": scale,
            "model_type": "gat",
            "auxiliaries": [kd],
        }

    def _teacher(self, scale: str = "large", stages=None) -> dict:
        return {
            "stages": list(stages or ["normal"]),
            "scale": scale,
            "model_type": "gat",
        }

    def test_explicit_teacher_config_wires_upstream(self, pipeline):
        from graphids.orchestrate.planning import enumerate_assets

        recipe = self._recipe({
            "teacher": self._teacher(scale="large"),
            "student": self._student(teacher_config="teacher", scale="small"),
        })
        specs = enumerate_assets(pipeline, recipe)
        student_specs = [s for s in specs if s.kd_tag == "_kd"]
        assert student_specs, "no KD student asset produced"
        student = student_specs[0]
        assert len(student.upstream_asset_names) >= 1
        # teacher asset name starts with "normal" (same stage as student)
        teacher_assets = [
            a for a in student.upstream_asset_names
            if a.startswith("normal") and "_kd" not in a
        ]
        assert teacher_assets, (
            f"teacher upstream not wired; got {student.upstream_asset_names}"
        )

    def test_missing_teacher_config_raises(self, pipeline):
        from graphids.orchestrate.planning import enumerate_assets

        recipe = self._recipe({
            "teacher_large": self._teacher(scale="large"),
            "student": self._student(teacher_config=None, scale="small"),
        })
        with pytest.raises(ValueError, match="missing teacher_config"):
            enumerate_assets(pipeline, recipe)

    def test_teacher_config_not_found_raises(self, pipeline):
        from graphids.orchestrate.planning import enumerate_assets

        recipe = self._recipe({
            "teacher_large": self._teacher(scale="large"),
            "student": self._student(teacher_config="nonexistent", scale="small"),
        })
        with pytest.raises(ValueError, match="does not name a config"):
            enumerate_assets(pipeline, recipe)

    def test_teacher_with_own_auxiliaries_raises(self, pipeline):
        """A config used as teacher must not itself have KD auxiliaries.

        The bad teacher must differ from the student in identity keys so the
        two configs don't dedup into the same asset (which would bypass the
        teacher-resolution pass entirely). Using scale=large for the teacher
        vs scale=small for the student is enough — scale is in the normal
        stage's identity_keys.
        """
        from graphids.orchestrate.planning import enumerate_assets

        recipe = self._recipe({
            "grandteacher": self._teacher(scale="large"),
            "bad_teacher_has_aux": self._student(
                teacher_config="grandteacher", scale="large",
            ),
            "student": self._student(
                teacher_config="bad_teacher_has_aux", scale="small",
            ),
        })
        with pytest.raises(ValueError, match="has its own auxiliaries"):
            enumerate_assets(pipeline, recipe)

    def test_teacher_missing_student_stage_raises(self, pipeline):
        """Teacher must produce the same stage the student needs."""
        from graphids.orchestrate.planning import enumerate_assets

        # Teacher only trains autoencoder; student needs teacher at "normal"
        recipe = self._recipe({
            "teacher_ae_only": {
                "stages": ["autoencoder"], "scale": "large", "model_type": "vgae",
            },
            "student": self._student(teacher_config="teacher_ae_only", scale="small"),
        })
        with pytest.raises(ValueError, match="does not produce a 'normal' asset"):
            enumerate_assets(pipeline, recipe)

    def test_key_order_insensitive(self, pipeline):
        """Renaming/reordering configs must not rewire the student teacher.

        This is the specific failure mode the old scale-based inference had:
        the "first match wins" loop depended on dict insertion order.
        """
        from graphids.orchestrate.planning import enumerate_assets

        # Two candidate teachers at the same scale; student names one explicitly.
        recipe_a = self._recipe({
            "alpha_large": self._teacher(scale="large"),
            "beta_large": self._teacher(scale="large", stages=["normal"]),
            "student": self._student(teacher_config="beta_large", scale="small"),
        })
        recipe_b = self._recipe({
            "beta_large": self._teacher(scale="large", stages=["normal"]),
            "alpha_large": self._teacher(scale="large"),
            "student": self._student(teacher_config="beta_large", scale="small"),
        })
        specs_a = enumerate_assets(pipeline, recipe_a)
        specs_b = enumerate_assets(pipeline, recipe_b)

        def student_upstream(specs):
            kd = next(s for s in specs if s.kd_tag == "_kd")
            return sorted(kd.upstream_asset_names)

        # Same student → same upstream set, regardless of config iteration order.
        assert student_upstream(specs_a) == student_upstream(specs_b)
