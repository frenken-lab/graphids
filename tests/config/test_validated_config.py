"""Tests for ``graphids.config.schemas`` validation."""

Phase 2 validation layer. Each test cites the specific invariant it guards
per ``.claude/rules/test-writing.md``; no formula mirrors, no pytest.raises
against framework-level errors (Pydantic does its own testing upstream).

The tests are split into two groups:

- ``TestRenderedStagesValidate`` — integration-level: render the real
  jsonnet stages and assert that every variant this repo ships produces a
  valid ``ValidatedConfig``. These tests cover the happy path end-to-end.

- ``TestConventionRulesEnforced`` — focused regression tests for the rules
  migrated from ``resolve._convention_errors`` into Pydantic
  ``@model_validator`` methods. Each one builds a minimal ``valid_dict``
  fixture, mutates the single field under test, and asserts the validator
  fires.
"""

from __future__ import annotations

import copy

import pytest

from graphids.config.jsonnet import render
from graphids.config.schemas import (
    ConfigValidationError,
    ValidatedConfig,
    validate_config,
)
from graphids.orchestrate.contracts import TrainingContract


# ---------------------------------------------------------------------------
# Happy path — every stage variant this repo ships validates
# ---------------------------------------------------------------------------


class TestRenderedStagesValidate:
    """Phase 2 must not reject any stage this repo currently ships.

    CONTRACT: each tuple in ``_STAGE_VARIANTS`` represents a stage + TLA
    combination that the planner is allowed to produce. Any Pydantic rule
    that makes one of these fail is a regression.
    """

    @pytest.mark.parametrize("stage", ["autoencoder", "supervised"])
    def test_default_stage_validates(self, stage: str) -> None:
        rendered = render(TrainingContract.resolve_jsonnet_path(stage))
        vc = validate_config(rendered)
        assert vc.data.class_path.startswith("graphids.core.preprocessing.")
        assert vc.model.class_path.startswith("graphids.core.models.")

    @pytest.mark.parametrize("scale", ["small", "large"])
    def test_scale_variants_validate(self, scale: str) -> None:
        rendered = render(
            TrainingContract.resolve_jsonnet_path("autoencoder"),
            tla={"scale": scale},
        )
        vc = validate_config(rendered)
        # Differential: scale flows into model.init_args.scale without
        # touching top-level identity. Invariant, not a formula mirror.
        assert vc.model.init_args["scale"] == scale

    @pytest.mark.parametrize(
        "method", ["bandit", "dqn", "mlp", "weighted_avg"]
    )
    def test_fusion_method_dispatch_validates(self, method: str) -> None:
        rendered = render(
            TrainingContract.resolve_jsonnet_path("fusion"),
            tla={"fusion_method": method},
        )
        vc = validate_config(rendered)
        # Every fusion stage agrees on val_acc/max — the stage-archetype
        # the planner warns about when violated.
        assert vc.checkpoint.monitor == "val_acc"
        assert vc.checkpoint.mode == "max"
        assert vc.early_stopping.monitor == "val_acc"

    def test_vgae_with_kd_auxiliary_validates(self) -> None:
        """VGAE student stage with a KD auxiliary must round-trip."""
        rendered = render(
            TrainingContract.resolve_jsonnet_path("autoencoder"),
            tla={
                "auxiliaries": [
                    {
                        "type": "kd",
                        "alpha": 0.7,
                        "vgae_latent_weight": 0.5,
                        "vgae_recon_weight": 0.5,
                    }
                ]
            },
        )
        vc = validate_config(rendered)
        aux = vc.model.init_args["auxiliaries"]
        assert isinstance(aux, list) and len(aux) == 1
        assert aux[0]["type"] == "kd"

    def test_ckpt_path_preserved_when_set(self) -> None:
        """Auto-resume ``ckpt_path`` TLA surfaces as a top-level field."""
        rendered = render(
            TrainingContract.resolve_jsonnet_path("autoencoder"),
            tla={"ckpt_path": "/tmp/resume.ckpt"},
        )
        vc = validate_config(rendered)
        assert vc.ckpt_path == "/tmp/resume.ckpt"


# ---------------------------------------------------------------------------
# Focused regression tests for convention rules
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_dict() -> dict:
    """A minimal rendered dict that validates cleanly.

    Tests mutate a copy to flip exactly one field and assert the matching
    validator fires. Matches the shape emitted by ``configs/stages/*.jsonnet``
    so the rules stay coupled to the real rendered output.
    """
    return {
        "seed_everything": 42,
        "trainer": {
            "accelerator": "auto",
            "devices": "auto",
            "precision": "16-mixed",
            "max_epochs": 300,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 50,
            "default_root_dir": "",
        },
        "data": {
            "class_path": "graphids.core.preprocessing.datamodule.CANBusDataModule",
            "init_args": {"batch_size": 8192, "dataset": "hcrl_ch"},
        },
        "model": {
            "class_path": "graphids.core.models.autoencoder.vgae.VGAEModule",
            "init_args": {
                "conv_type": "gatv2",
                "hidden_dims": [80, 40, 16],
                "pool_aggrs": ["mean"],
                "auxiliaries": [],
            },
        },
        "checkpoint": {
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": True,
            "filename": "best_model",
        },
        "early_stopping": {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 100,
        },
    }


class TestConventionRulesEnforced:
    """Regression tests for rules migrated from resolve._convention_errors.

    Each test mutates ``valid_dict`` to violate exactly one rule and
    asserts the matching validator catches it.
    """

    def test_baseline_fixture_validates(self, valid_dict: dict) -> None:
        """INVARIANT: the fixture itself is a valid input.

        Without this, a rule regression could make every other test in
        this class silently pass for the wrong reason.
        """
        assert isinstance(validate_config(valid_dict), ValidatedConfig)

    @pytest.mark.parametrize(
        "field", ["pool_aggrs", "hidden_dims", "auxiliaries"]
    )
    def test_null_model_list_field_rejected(
        self, valid_dict: dict, field: str
    ) -> None:
        """REGRESSION: null list fields in model.init_args must die early.

        Source: ``_convention_errors`` in resolve.py pre-Phase-2. jsonargparse
        rejects these at instantiation time but with a cryptic error; Phase 2
        fails loudly at planning time instead.
        """
        bad = copy.deepcopy(valid_dict)
        bad["model"]["init_args"][field] = None
        with pytest.raises(ConfigValidationError, match="null"):
            validate_config(bad)

    def test_monitor_checkpoint_vs_early_stopping_mismatch_rejected(
        self, valid_dict: dict
    ) -> None:
        """CONTRACT: checkpoint and early_stopping track the same metric.

        Previously a warning against a hardcoded stage table; Phase 2
        promotes it to an error because every stage this repo ships
        already satisfies the stricter rule and a divergence is almost
        always a typo in the stage libsonnet.
        """
        bad = copy.deepcopy(valid_dict)
        bad["early_stopping"]["monitor"] = "val_acc"
        bad["early_stopping"]["mode"] = "max"
        with pytest.raises(ConfigValidationError, match="same metric"):
            validate_config(bad)

    def test_lr_monitor_without_logger_rejected(self, valid_dict: dict) -> None:
        """REGRESSION: LearningRateMonitor callback needs trainer.logger on.

        Source: ``_convention_errors`` in resolve.py pre-Phase-2.
        """
        bad = copy.deepcopy(valid_dict)
        bad["trainer"]["logger"] = False
        bad["trainer"]["callbacks"] = [
            {"class_path": "pytorch_lightning.callbacks.LearningRateMonitor"}
        ]
        with pytest.raises(ConfigValidationError, match="LearningRateMonitor"):
            validate_config(bad)

    def test_lr_monitor_with_logger_on_accepted(self, valid_dict: dict) -> None:
        """Differential: same callback passes when logger is on (default)."""
        ok = copy.deepcopy(valid_dict)
        ok["trainer"]["callbacks"] = [
            {"class_path": "pytorch_lightning.callbacks.LearningRateMonitor"}
        ]
        validate_config(ok)  # no raise

    def test_class_path_outside_graphids_namespace_rejected(
        self, valid_dict: dict
    ) -> None:
        """CONTRACT: every model/data class_path must be namespaced.

        Guards against relative imports and stray top-level modules
        sneaking into a stage libsonnet.
        """
        bad = copy.deepcopy(valid_dict)
        bad["model"]["class_path"] = "totally.external.module.Model"
        with pytest.raises(ConfigValidationError, match="must start with"):
            validate_config(bad)

    def test_extra_top_level_key_rejected(self, valid_dict: dict) -> None:
        """CONTRACT: unknown top-level keys indicate a recipe/stage typo.

        extra="forbid" on ValidatedConfig is the whole point of the Phase 2
        layer — it's how we'll catch typos that jsonargparse silently
        absorbed before (e.g. ``trinaer`` instead of ``trainer``).
        """
        bad = copy.deepcopy(valid_dict)
        bad["trinaer"] = {"max_epochs": 1}
        with pytest.raises(ConfigValidationError):
            validate_config(bad)

    def test_checkpoint_mode_must_be_min_or_max(self, valid_dict: dict) -> None:
        """CONTRACT: ModelCheckpoint.mode literal enum is enforced."""
        bad = copy.deepcopy(valid_dict)
        bad["checkpoint"]["mode"] = "middle"
        with pytest.raises(ConfigValidationError):
            validate_config(bad)

    def test_trainer_extra_kwargs_allowed(self, valid_dict: dict) -> None:
        """CONTRACT: Trainer accepts ~50 kwargs; TrainerSection must allow extras.

        Phase 2 is deliberately lenient on ``trainer``. Phase 3 narrows
        this to an explicit typed subset as part of the LightningCLI
        strip.
        """
        ok = copy.deepcopy(valid_dict)
        ok["trainer"]["detect_anomaly"] = True
        ok["trainer"]["val_check_interval"] = 0.25
        vc = validate_config(ok)
        # Pydantic with extra="allow" exposes extras via model_extra.
        dumped = vc.trainer.model_dump()
        assert dumped["detect_anomaly"] is True
        assert dumped["val_check_interval"] == 0.25
