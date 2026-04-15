"""Tests for ``graphids.config.schemas`` validation.

After the callback-layout refactor (checkpoint/early_stopping moved
from top-level ``ValidatedConfig`` fields into ``trainer.callbacks``
entries), the old ``valid_dict`` fixture and every test that read
``vc.checkpoint`` / ``vc.early_stopping`` / ``vc.ckpt_path`` referenced
properties that no longer exist. Those 14 tests were deleted 2026-04-13.

What remains: happy-path tests that render real stage jsonnets and
assert the result validates. These don't assume a schema layout —
they just confirm every stage variant ships validates cleanly.
"""

from __future__ import annotations

import pytest

from graphids.config.constants import PROJECT_ROOT
from graphids.config.jsonnet import render
from graphids.config.schemas import validate_config

_STAGES_DIR = PROJECT_ROOT / "configs" / "stages"


def _stage_path(stage: str) -> str:
    return str(_STAGES_DIR / f"{stage}.jsonnet")


class TestRenderedStagesValidate:
    """Every stage variant this repo ships must validate without error."""

    @pytest.mark.parametrize("stage", ["autoencoder", "supervised"])
    def test_default_stage_validates(self, stage: str) -> None:
        rendered = render(_stage_path(stage))
        vc = validate_config(rendered)
        assert vc.data.class_path.startswith("graphids.core.data.")
        assert vc.model.class_path.startswith("graphids.core.models.")

    @pytest.mark.parametrize("scale", ["small", "large"])
    def test_scale_variants_validate(self, scale: str) -> None:
        rendered = render(
            _stage_path("autoencoder"),
            tla={"scale": scale},
        )
        vc = validate_config(rendered)
        # Differential: scale flows into model.init_args.scale without
        # touching top-level identity. Invariant, not a formula mirror.
        assert vc.model.init_args["scale"] == scale

    def test_vgae_with_kd_auxiliary_validates(self) -> None:
        """VGAE student stage with a KD auxiliary must round-trip."""
        rendered = render(
            _stage_path("autoencoder"),
            tla={
                "distillation_config": {
                    "type": "kd",
                    "alpha": 0.7,
                    "vgae_latent_weight": 0.5,
                    "vgae_recon_weight": 0.5,
                },
            },
        )
        vc = validate_config(rendered)
        assert vc.model.init_args.get("distillation_config", {}).get("type") == "kd"
