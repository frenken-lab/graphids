"""graphids-specific Lightning callbacks."""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl

from graphids._fs import _sha256_file


class Sha256ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """``ModelCheckpoint`` + ``<ckpt>.sha256`` sidecar after every save.

    GPFS truncation surprises happen on OSC; the sidecar is the load-time
    integrity check used by ``_fs.atomic_load``.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self._ensure_last_epoch(trainer, train_epoch_end=True)

    def on_validation_end(self, trainer, pl_module) -> None:
        super().on_validation_end(trainer, pl_module)
        self._ensure_last_epoch(trainer, train_epoch_end=False)

    def _ensure_last_epoch(self, trainer, *, train_epoch_end: bool) -> None:
        # Lightning can skip last.ckpt when this epoch did not enter top-k.
        if (
            self.save_last is True
            and not self._should_skip_saving_checkpoint(trainer)
            and self._should_save_on_train_epoch_end(trainer) == train_epoch_end
            and self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        ):
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:  # type: ignore[override]
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            p = Path(filepath)
            p.with_suffix(p.suffix + ".sha256").write_text(_sha256_file(p) + "\n")
