"""Direct instantiator smoke tests.

``graphids.orchestrate.instantiate.build_run`` replaces the old
``build_cli``/``GraphIDSCLI`` path. These tests exercise the full chain
for every stage + fusion method variant the repo ships:

    render(jsonnet) → validate_config → build_run(...)

so the CI surface catches signature filtering, KD auxiliary coercion,
forced callback wiring, and class_path imports without having to launch
a SLURM job. No ``trainer.fit`` — these are structural tests.

REGRESSION: fusion models (Bandit/DQN/MLP/WeightedAvg) do NOT accept
``dataset``/``conv_type``/``heads`` in ``__init__``. ``filter_kwargs``
must drop unaccepted keys or every fusion instantiation blows up with a
TypeError.
"""

from __future__ import annotations

import inspect

import pytest

from graphids.config.jsonnet import render
from graphids.orchestrate.instantiate import build_run, filter_kwargs

_STAGE_CASES: list[tuple[str, dict]] = [
    ("autoencoder", {}),
    ("autoencoder", {"scale": "large"}),
    ("supervised", {"scale": "small"}),
    ("fusion", {"fusion_method": "bandit"}),
    ("fusion", {"fusion_method": "dqn"}),
    ("fusion", {"fusion_method": "mlp"}),
    ("fusion", {"fusion_method": "weighted_avg"}),
]


def _id(stage_tla):
    stage, tla = stage_tla
    label = stage
    if "fusion_method" in tla:
        label += f"_{tla['fusion_method']}"
    if "distillation_config" in tla:
        label += "_kd"
    if tla.get("scale") == "large":
        label += "_large"
    return label


@pytest.mark.parametrize("stage_tla", _STAGE_CASES, ids=_id)
def test_stage_instantiates(stage_tla):
    """Every shipping stage renders → validates → instantiates without error."""
    stage, tla = stage_tla
    merged = render(
        f"configs/stages/{stage}.jsonnet",
        tla=tla or None,
    )
    run = build_run(merged, seed_all=False)
    assert run.trainer is not None
    assert run.model is not None
    assert run.datamodule is not None


class TestForcedCallbacks:
    """The callback set wired by defaults.libsonnet must land on every trainer.

    INVARIANT: ModelCheckpoint, EarlyStopping, OTelTrainingCallback, and
    CurriculumEpochCallback are present for every stage.
    """

    def test_autoencoder_has_full_callback_set(self):
        from graphids.core.callbacks import EarlyStopping, ModelCheckpoint
        from graphids.core.data.curriculum import CurriculumEpochCallback
        from graphids.core.monitoring import OTelTrainingCallback

        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        run = build_run(merged, seed_all=False)
        cbs = run.trainer.callbacks
        cb_types = {type(cb) for cb in cbs}
        for required in (
            ModelCheckpoint,
            EarlyStopping,
            OTelTrainingCallback,
            CurriculumEpochCallback,
        ):
            assert required in cb_types, f"missing callback {required.__name__}"


class TestSignatureFiltering:
    """INVARIANT: ``filter_kwargs`` drops unaccepted init kwargs.

    REGRESSION: fusion models don't accept ``dataset``/``conv_type``;
    ``filter_kwargs`` must drop these or fusion instantiation raises
    TypeError at build_run time.
    """

    def test_vgae_receives_linked_dataset_and_seed(self):
        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        assert merged["model"]["init_args"]["dataset"] == "hcrl_ch"
        assert merged["model"]["init_args"]["seed"] == 42
        # build_run instantiates without crashing, which exercises that
        # VGAEModule accepts these linked kwargs.
        run = build_run(merged, seed_all=False)
        assert run.model is not None

    def test_fusion_model_skips_unaccepted_links(self):
        from graphids.core.models.fusion.bandit import BanditFusionModule

        # BanditFusionModule.__init__ does NOT accept `dataset` — verify
        # that filter_kwargs drops it rather than passing it through.
        accepted = set(inspect.signature(BanditFusionModule.__init__).parameters) - {"self"}
        assert "dataset" not in accepted
        stripped = filter_kwargs(BanditFusionModule, {"dataset": "x", "num_models": 3})
        assert "dataset" not in stripped

        # And build_run succeeds on the fusion stage (end-to-end proof).
        merged = render("configs/stages/fusion.jsonnet", tla={"fusion_method": "bandit"})
        run = build_run(merged, seed_all=False)
        assert run.model is not None


class TestKDAuxiliariesCoercion:
    """CONTRACT: loss_fn wiring for KD auxiliaries."""

    def test_default_loss_fn_built(self):
        """CONTRACT: VGAE stage builds a wrapped loss_fn (no distillation)."""
        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        run = build_run(merged, seed_all=False)
        assert type(run.model.loss_fn).__name__ == "VGAETaskLoss"

    def test_no_distillation_config_default(self):
        """CONTRACT: null distillation_config still produces a valid loss."""
        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        assert merged["model"]["init_args"].get("distillation_config") is None
        run = build_run(merged, seed_all=False)
        assert run.model.loss_fn is not None


class TestCheckpointDirpathConvention:
    """REGRESSION: ModelCheckpoint writes to ``{default_root_dir}/checkpoints``.

    The ``/checkpoints`` subdir convention is owned by ``ModelCheckpoint``
    itself (``_resolve_dirpath``), not by the instantiator or jsonnet.
    """

    def test_default_dirpath_tracks_trainer_root(self, tmp_path):
        from unittest.mock import MagicMock

        from graphids.core.callbacks import ModelCheckpoint

        cb = ModelCheckpoint()
        trainer = MagicMock(default_root_dir=str(tmp_path))
        assert cb._resolve_dirpath(trainer) == tmp_path / "checkpoints"

    def test_explicit_dirpath_overrides_default(self, tmp_path):
        from unittest.mock import MagicMock

        from graphids.core.callbacks import ModelCheckpoint

        cb = ModelCheckpoint(dirpath=str(tmp_path / "custom"))
        trainer = MagicMock(default_root_dir="/ignored")
        assert cb._resolve_dirpath(trainer) == tmp_path / "custom"
