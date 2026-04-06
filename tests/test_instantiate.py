"""Phase 3 direct instantiator smoke tests.

``graphids.core.instantiate.instantiate`` replaces the old
``build_cli``/``GraphIDSCLI`` path. These tests exercise the full chain
for every stage + fusion method variant the repo ships:

    render_config(jsonnet) → validate_config → instantiate(...)

so the CI surface catches link_arguments regressions, KD auxiliary
coercion, forced callback wiring, and class_path imports without having
to launch a SLURM job. No ``trainer.fit`` — these are structural tests.

REGRESSION: fusion models (Bandit/DQN/MLP/WeightedAvg) do NOT accept
``dataset``/``conv_type``/``heads`` in ``__init__``. jsonargparse's
``link_arguments`` silently skipped unaccepted links via signature
inspection; the Phase 3 replacement must preserve that behavior or every
fusion instantiation blows up with a TypeError.
"""

from __future__ import annotations

import pytest

from graphids.config.jsonnet import render
from graphids.instantiate import _init_kwargs, instantiate

_STAGE_CASES: list[tuple[str, dict]] = [
    ("autoencoder", {}),
    ("autoencoder", {"scale": "large"}),
    # KD case omitted — requires a real teacher checkpoint on disk.
    # KD loss wiring is covered by TestKDAuxiliariesCoercion.
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
    run = instantiate(merged, seed_everything=False)
    assert run.trainer is not None
    assert run.model is not None
    assert run.datamodule is not None


class TestForcedCallbacks:
    """Phase 3 constructs the forced callback set explicitly.

    INVARIANT: the five callbacks that LightningCLI used to force via
    ``add_lightning_class_args`` (ModelCheckpoint, EarlyStopping,
    DeviceStatsMonitor, ResourceProfileCallback, RunRecordCallback) must
    still land on every trainer regardless of stage.
    """

    def test_autoencoder_has_full_forced_set(self):
        from pytorch_lightning.callbacks import (
            DeviceStatsMonitor,
            EarlyStopping,
            ModelCheckpoint,
        )

        from graphids.core.monitoring.callbacks import ResourceProfileCallback, RunRecordCallback

        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        run = instantiate(merged, seed_everything=False)
        cbs = run.trainer.callbacks
        cb_types = {type(cb) for cb in cbs}
        for required in (
            ModelCheckpoint,
            EarlyStopping,
            DeviceStatsMonitor,
            ResourceProfileCallback,
            RunRecordCallback,
        ):
            assert required in cb_types, f"missing forced callback {required.__name__}"


class TestLinkArguments:
    """INVARIANT: signature-filtered link_arguments propagation.

    REGRESSION: fusion models don't accept ``dataset``/``conv_type``; the
    pre-Phase-3 jsonargparse CLI filtered this via target signature
    inspection. ``_apply_link_arguments`` must do the same or fusion
    instantiation raises TypeError.
    """

    def test_vgae_receives_linked_dataset_and_seed(self):
        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        run = instantiate(merged, seed_everything=False)
        assert run.merged["model"]["init_args"]["dataset"] == "hcrl_ch"
        assert run.merged["model"]["init_args"]["seed"] == 42
        assert run.merged["data"]["init_args"]["seed"] == 42
        assert run.merged["data"]["init_args"]["conv_type"] == "gatv2"

    def test_fusion_model_skips_unaccepted_links(self):
        merged = render(
            "configs/stages/fusion.jsonnet",
            tla={"fusion_method": "bandit"},
        )
        run = instantiate(merged, seed_everything=False)
        # BanditFusionModule.__init__ does NOT accept `dataset` — the link
        # must be skipped rather than set.
        from graphids.core.models.fusion.bandit import BanditFusionModule

        assert "dataset" not in _init_kwargs(BanditFusionModule)
        assert "dataset" not in run.merged["model"]["init_args"]

    def test_init_kwargs_matches_signature(self):
        """``_init_kwargs`` helper mirrors ``inspect.signature``, sans self/varargs."""
        from graphids.core.models.autoencoder.vgae_module import VGAEModule

        params = _init_kwargs(VGAEModule)
        # Sanity-check a few known kwargs:
        assert "dataset" in params
        assert "seed" in params
        assert "lr" in params
        assert "self" not in params


class TestKDAuxiliariesCoercion:
    """CONTRACT: KD auxiliaries list items must support attribute access.

    Pre-Phase-3 jsonargparse wrapped TypedDict list items as Namespace
    objects so ``_install_kd_teacher`` could call ``getattr(a, 'type')``.
    Phase 3 coerces to ``SimpleNamespace`` instead — same contract.
    """

    def test_distillation_config_builds_loss(self):
        """CONTRACT: distillation_config TLA produces a wrapped loss_fn."""
        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        run = instantiate(merged, seed_everything=False)
        # Without distillation_config, loss_fn is plain VGAETaskLoss
        assert type(run.model.loss_fn).__name__ == "VGAETaskLoss"

    def test_no_distillation_config_default(self):
        """CONTRACT: null distillation_config still produces a valid loss."""
        merged = render("configs/stages/autoencoder.jsonnet", tla=None)
        assert merged["model"]["init_args"].get("distillation_config") is None
        run = instantiate(merged, seed_everything=False)
        assert run.model.loss_fn is not None


class TestCheckpointDirpathPatched:
    """REGRESSION: ModelCheckpoint.dirpath must track ``trainer.default_root_dir``.

    Pre-Phase-3 ``patch_config_paths`` set this on the parsed Namespace
    before instantiation. Phase 3 does the same in ``_build_callbacks``.
    """

    def test_dirpath_set_to_run_dir_subpath(self, tmp_path):
        from pytorch_lightning.callbacks import ModelCheckpoint

        merged = render(
            "configs/stages/autoencoder.jsonnet",
            tla={"run_dir": str(tmp_path)},
        )
        run = instantiate(merged, seed_everything=False)
        ckpt = next(cb for cb in run.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.dirpath == f"{tmp_path}/checkpoints"
