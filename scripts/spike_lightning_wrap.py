"""Step 1 of the Lightning-migration spike: wrapper-shim.

Question: does ``pl.Trainer.fit()`` drive our existing ``GraphModuleBase``
contract end-to-end without modifying the model? If yes, full
inheritance migration is mechanical. If no, we surface the exact
contract break.

Strategy: do NOT touch ``graphids.core.models.base``. Wrap an existing
VGAE in a thin ``pl.LightningModule`` that delegates ``training_step`` /
``validation_step`` / ``configure_optimizers`` to the inner model.

Smoke = ``pl.Trainer(fast_dev_run=True, accelerator='cpu')`` with a
tiny synthetic DataLoader of ``make_batch()``. fast_dev_run runs 1
train + 1 val batch then exits — exercises every Lightning hook
(setup, training_step, optimizer.step, validation_step, on_*_end)
without committing to a real fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse the test conftest's tiny batch fixture without pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, make_batch  # noqa: E402

import lightning as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from graphids.core.losses.autoencoder import VGAETaskLoss  # noqa: E402
from graphids.core.models.autoencoder.vgae import VGAE  # noqa: E402


class _BatchRepeater(Dataset):
    """Tiny dataset that yields the same Batch ``n`` times — Lightning's
    DataLoader contract just needs ``__len__`` and ``__getitem__``."""

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.batch = make_batch(2)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _idx: int):
        return self.batch


def _identity_collate(items):
    # Items are already batched (PyG Batch); take the first.
    return items[0]


class LightningWrap(pl.LightningModule):
    """Thin shim — delegates to inner ``GraphModuleBase`` without
    touching it. Tests whether Lightning's training engine accepts
    our existing contract."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def training_step(self, batch, batch_idx):
        return self.inner.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.inner.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Inner is now a LightningModule itself post-migration; just
        # delegate to its own configure_optimizers.
        return self.inner.configure_optimizers()


def _make_vgae() -> VGAE:
    """Match tests/core/models/test_vgae.py::_make_vgae."""
    return VGAE(
        loss_fn=VGAETaskLoss(),
        hidden_dims=[32, 16],
        latent_dim=16,
        heads=2,
        embedding_dim=4,
        dropout=0.0,
        conv_type="gatv2",
        edge_dim=EDGE_DIM,
        proj_dim=0,
        num_ids=NUM_IDS,
        in_channels=IN_CHANNELS,
        gradient_checkpointing=False,
        compile_model=False,
    )


def main() -> int:
    # Post-migration VGAE IS a LightningModule; the wrapper-shim is no longer
    # needed and would in fact break self.log() (the inner has no trainer
    # attached when wrapped). Drive Lightning directly.
    model = _make_vgae()

    train_loader = DataLoader(
        _BatchRepeater(n=4), batch_size=1, collate_fn=_identity_collate
    )
    val_loader = DataLoader(
        _BatchRepeater(n=2), batch_size=1, collate_fn=_identity_collate
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n=== Spike result ===")
    print("pl.Trainer.fit completed without raising.")
    print(f"global_step = {trainer.global_step}")
    print(f"current_epoch = {trainer.current_epoch}")
    cm = trainer.callback_metrics
    print(f"callback_metrics keys = {sorted(cm.keys())}")
    # Post-migration the inner IS a LightningModule, so cm should now be
    # populated by self.log(...) calls inside training_step/validation_step.
    return 0


if __name__ == "__main__":
    sys.exit(main())
