"""Fusion DataModule — serves a pre-extracted TensorDict of fusion features.

Loads the cache produced by an ``ExtractRow`` (see ``configs/plans/fusion.jsonnet``).
Yields ``(td_batch, labels)`` per step where ``td_batch`` is a nested
TensorDict keyed by upstream model name (e.g. ``td["vgae", "errors"]``).
"""

from __future__ import annotations

import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from structlog import get_logger
from tensordict import TensorDict

from graphids.core.data.fusion_states import (
    CACHE_VERSION,
    FUSION_STATES_DIR,
    TRAIN_FILENAME,
    VAL_FILENAME,
)

log = get_logger(__name__)


class FusionDataModule(pl.LightningDataModule):
    """Loads a pre-extracted fusion TensorDict and serves batches."""

    def __init__(
        self,
        cached_states_dir: str = "",
        method: str = "bandit",
        batch_size: int = 128,
        episode_sample_size: int = 20000,
    ):
        super().__init__()
        # See GraphDataModule for why this is ``_hp``, not ``hparams``.
        self._hp = {k: v for k, v in locals().items() if k != "self"}
        is_rl = method in ("dqn", "bandit")
        self._batch_size = episode_sample_size if is_rl else batch_size
        self.train_td: TensorDict | None = None
        self.val_td: TensorDict | None = None

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.train_td.batch_size[0] / self._batch_size)

    def _load_one(self, path: Path, which: str) -> TensorDict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        version = blob.get("version") if isinstance(blob, dict) else None
        if version != CACHE_VERSION:
            raise RuntimeError(
                f"Fusion {which} cache at {path} has version={version}, expected "
                f"{CACHE_VERSION}. Re-run the ExtractRow in your fusion plan "
                "(cache format changed: flat states → TensorDict)."
            )
        return TensorDict(blob["td"], batch_size=[blob["td"]["labels"].size(0)])

    def setup(self, stage=None):
        if self.train_td is not None:
            return

        hp = self._hp
        if not hp["cached_states_dir"]:
            raise ValueError(
                "cached_states_dir is required — submit the ExtractRow from "
                "configs/plans/fusion.jsonnet first (or chain via "
                "'graphids submit --depends-on-afterok <extract_jid>')"
            )

        states_dir = Path(hp["cached_states_dir"])
        if not (states_dir / TRAIN_FILENAME).exists():
            states_dir = states_dir / FUSION_STATES_DIR
        train_path = states_dir / TRAIN_FILENAME
        val_path = states_dir / VAL_FILENAME
        if not train_path.exists():
            raise FileNotFoundError(f"Cached train states not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Cached val states not found: {val_path}")

        self.train_td = self._load_one(train_path, "train")
        self.val_td = self._load_one(val_path, "val")
        log.info(
            "loaded_cached_states",
            dir=str(states_dir),
            train_n=self.train_td.batch_size[0],
            val_n=self.val_td.batch_size[0],
            keys=list(self.train_td.keys(include_nested=True, leaves_only=True)),
        )

    def _batches(self, td: TensorDict, shuffle: bool):
        n = td.batch_size[0]
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, self._batch_size):
            sel = idx[start : start + self._batch_size]
            sub = td[sel]
            labels = sub["labels"]
            features = sub.exclude("labels")
            yield features, labels

    def train_dataloader(self):
        return self._batches(self.train_td, shuffle=True)

    def val_dataloader(self):
        return self._batches(self.val_td, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()
