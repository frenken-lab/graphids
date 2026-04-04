"""KD teacher resolution — explicit teacher_config wiring through
enumerate_assets. Replaces the old ``teacher_scale + first-match`` inference
with explicit teacher_config naming. See docs/reference/orchestration-risks.md
item #2."""

from __future__ import annotations

import pytest

from graphids.config import PIPELINE_YAML
from graphids.orchestrate.planning import enumerate_assets


def _recipe(configs: dict) -> dict:
    """Build a minimal expanded-recipe dict around a set of configs."""
    return {
        "defaults": {},
        "configs": configs,
        "sweep": {"seeds": [42]},
        "trainer_overrides": {},
        "stage_overrides": {},
        "resource_overrides": {},
    }


def _student(teacher_config: str | None, scale: str = "small") -> dict:
    kd: dict = {"type": "kd", "alpha": 0.5, "teacher_scale": "large"}
    if teacher_config is not None:
        kd["teacher_config"] = teacher_config
    return {
        "stages": ["normal"],
        "scale": scale,
        "model_type": "gat",
        "auxiliaries": [kd],
    }


def _teacher(scale: str = "large", stages=None) -> dict:
    return {
        "stages": list(stages or ["normal"]),
        "scale": scale,
        "model_type": "gat",
    }


class TestKDTeacherResolution:
    def test_explicit_teacher_config_wires_upstream(self):
        recipe = _recipe({
            "teacher": _teacher(scale="large"),
            "student": _student(teacher_config="teacher", scale="small"),
        })
        specs = enumerate_assets(PIPELINE_YAML, recipe)
        student_specs = [s for s in specs if s.kd_tag == "_kd"]
        assert student_specs, "no KD student asset produced"
        student = student_specs[0]
        assert len(student.upstream_asset_names) >= 1
        teacher_assets = [
            a for a in student.upstream_asset_names
            if a.startswith("normal") and "_kd" not in a
        ]
        assert teacher_assets, (
            f"teacher upstream not wired; got {student.upstream_asset_names}"
        )

    def test_missing_teacher_config_raises(self):
        recipe = _recipe({
            "teacher_large": _teacher(scale="large"),
            "student": _student(teacher_config=None, scale="small"),
        })
        with pytest.raises(ValueError, match="missing teacher_config"):
            enumerate_assets(PIPELINE_YAML, recipe)

    def test_teacher_config_not_found_raises(self):
        recipe = _recipe({
            "teacher_large": _teacher(scale="large"),
            "student": _student(teacher_config="nonexistent", scale="small"),
        })
        with pytest.raises(ValueError, match="does not name a config"):
            enumerate_assets(PIPELINE_YAML, recipe)

    def test_teacher_with_own_auxiliaries_raises(self):
        """A config used as teacher must not itself have KD auxiliaries.

        The bad teacher must differ from the student in identity keys so the
        two configs don't dedup into the same asset (which would bypass the
        teacher-resolution pass entirely). Using scale=large for the teacher
        vs scale=small for the student is enough — scale is in the normal
        stage's identity_keys.
        """
        recipe = _recipe({
            "grandteacher": _teacher(scale="large"),
            "bad_teacher_has_aux": _student(
                teacher_config="grandteacher", scale="large",
            ),
            "student": _student(
                teacher_config="bad_teacher_has_aux", scale="small",
            ),
        })
        with pytest.raises(ValueError, match="has its own auxiliaries"):
            enumerate_assets(PIPELINE_YAML, recipe)

    def test_teacher_missing_student_stage_raises(self):
        """Teacher must produce the same stage the student needs."""
        recipe = _recipe({
            "teacher_ae_only": {
                "stages": ["autoencoder"], "scale": "large", "model_type": "vgae",
            },
            "student": _student(teacher_config="teacher_ae_only", scale="small"),
        })
        with pytest.raises(ValueError, match="does not produce a 'normal' asset"):
            enumerate_assets(PIPELINE_YAML, recipe)

    def test_key_order_insensitive(self):
        """Renaming/reordering configs must not rewire the student teacher.

        This is the specific failure mode the old scale-based inference had:
        the "first match wins" loop depended on dict insertion order.
        """
        recipe_a = _recipe({
            "alpha_large": _teacher(scale="large"),
            "beta_large": _teacher(scale="large", stages=["normal"]),
            "student": _student(teacher_config="beta_large", scale="small"),
        })
        recipe_b = _recipe({
            "beta_large": _teacher(scale="large", stages=["normal"]),
            "alpha_large": _teacher(scale="large"),
            "student": _student(teacher_config="beta_large", scale="small"),
        })
        specs_a = enumerate_assets(PIPELINE_YAML, recipe_a)
        specs_b = enumerate_assets(PIPELINE_YAML, recipe_b)

        def student_upstream(specs):
            kd = next(s for s in specs if s.kd_tag == "_kd")
            return sorted(kd.upstream_asset_names)

        # Same student → same upstream set, regardless of config iteration order.
        assert student_upstream(specs_a) == student_upstream(specs_b)
