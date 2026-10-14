"""Source-only epoch history and calibrated best-checkpoint integration."""

import json

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger

from graphids.core.callbacks import (
    GraphMambaLearningCurveCallback,
    Sha256ModelCheckpoint,
)
from graphids.core.data.snapshots.datamodule import CANSnapshotDataModule
from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from scripts.graph_mamba_gate import fixture_manifest

pytestmark = pytest.mark.slurm


def test_two_epoch_learning_curve_and_source_selected_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIDS_RUN_DIR", str(tmp_path))
    manifest = fixture_manifest(tmp_path / "fixtures")
    dm = CANSnapshotDataModule(
        manifest_path=str(manifest),
        datasets=["set_01", "hcrl_sa"],
        sequence_bins=32,
        warmup_bins=16,
        eval_chunk_bins=37,
        cache_root=str(tmp_path / "cache"),
    )
    dm.setup()
    net = VehicleGraphMambaJoint(width=8, dropout=0, backend="reference")
    net.prepare_from_datamodule(dm)
    checkpoint = Sha256ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="best",
        monitor="val/zero-shot/classification_ap_macro",
        mode="max",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    curve = GraphMambaLearningCurveCallback(str(tmp_path / "learning-curve.json"))
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=CSVLogger(str(tmp_path), name="metrics"),
        callbacks=[curve, checkpoint],
    )
    trainer.fit(net, datamodule=dm)
    history = json.loads((tmp_path / "learning-curve.json").read_text())
    assert [epoch["epoch"] for epoch in history["epochs"]] == [1, 2]
    assert [epoch["optimizer_steps"] for epoch in history["epochs"]] == [2, 4]
    assert all(epoch["training_steps_this_rank"] == 2 for epoch in history["epochs"])
    assert all(
        epoch["available_training_batches"] == len(dm.train_sampler)
        for epoch in history["epochs"]
    )
    assert all(
        "val/zero-shot/classification_ap_macro" in epoch["validation"]
        for epoch in history["epochs"]
    )
    assert all(
        set(epoch["train_mean"]) == set(curve.names) for epoch in history["epochs"]
    )
    assert (tmp_path / "checkpoints/best.ckpt.sha256").is_file()
    assert (tmp_path / "checkpoints/last.ckpt").is_file()
    selected = VehicleGraphMambaJoint.load_from_checkpoint(checkpoint.best_model_path)
    assert selected.calibrated
    saved = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    assert saved["graphids_schema"]["data_manifest"] == dm.manifest.digest


@pytest.mark.parametrize("decrease", [0, 1])
def test_unmonitored_final_checkpoint_retains_final_weights_when_best_is_earlier(
    tmp_path,
    decrease,
):
    """The future preset must retain last weights even when monitored AP declines."""

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("selection", float(1 - decrease * self.current_epoch))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    from torch.utils.data import DataLoader

    best = Sha256ModelCheckpoint(
        dirpath=tmp_path,
        filename="best",
        monitor="selection",
        mode="max",
        save_last=False,
        enable_version_counter=False,
    )
    last = Sha256ModelCheckpoint(
        dirpath=tmp_path,
        filename="last",
        monitor=None,
        save_last=False,
        save_top_k=1,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        enable_version_counter=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[best, last],
    )
    loader = DataLoader(torch.ones(2, 1), batch_size=1)
    trainer.fit(Tiny(), train_dataloaders=loader, val_dataloaders=loader)
    best_saved = torch.load(tmp_path / "best.ckpt", weights_only=True)
    last_saved = torch.load(tmp_path / "last.ckpt", weights_only=True)
    assert best_saved["epoch"] == 0 and last_saved["epoch"] == 1
    assert last_saved["global_step"] == 4
    assert best_saved["state_dict"]["weight"] != last_saved["state_dict"]["weight"]
