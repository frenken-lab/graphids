"""Fusion DataModule (v2): pre-extracted TensorDict → batched (features, labels).

Loads the cache produced by an ``ExtractRow`` (``configs/plans/fusion.jsonnet``).
Yields ``(features_td, labels)`` per step where ``features_td`` is the
nested TensorDict minus the ``labels`` leaf — keyed by upstream model
name (e.g. ``td["vgae", "errors"]``).

Composition: TensorDict's native indexing + ``td.exclude("labels")``.
Bandit/DQN modes use a single big "batch" of size ``episode_sample_size``
(RL methods consume an episode at a time, not gradient steps).
"""

from __future__ import annotations

import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from structlog import get_logger
from tensordict import TensorDict

from graphids.core.data.extract import (
    CACHE_VERSION,
    FUSION_STATES_DIR,
    TRAIN_FILENAME,
    VAL_FILENAME,
)

log = get_logger(__name__)


def _load_td(path: Path, which: str) -> TensorDict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    version = blob.get("version") if isinstance(blob, dict) else None
    if version != CACHE_VERSION:
        raise RuntimeError(
            f"fusion {which} cache at {path} has version={version!r}, "
            f"expected {CACHE_VERSION}; re-run the ExtractRow"
        )
    return TensorDict(blob["td"], batch_size=[blob["td"]["labels"].size(0)])


class FusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cached_states_dir: str = "",
        method: str = "bandit",
        batch_size: int = 128,
        episode_sample_size: int = 20000,
    ):
        super().__init__()
        self.cached_states_dir = cached_states_dir
        self.method = method
        # RL methods consume one big episode-sized chunk; gradient methods step.
        self._batch_size = episode_sample_size if method in ("dqn", "bandit") else batch_size
        self.train_td: TensorDict | None = None
        self.val_td: TensorDict | None = None

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.train_td.batch_size[0] / self._batch_size)

    def setup(self, stage: str | None = None) -> None:
        if self.train_td is not None:
            return
        if not self.cached_states_dir:
            raise ValueError(
                "cached_states_dir is required — submit the ExtractRow first "
                "or chain via 'graphids submit --depends-on-afterok <jid>'"
            )
        d = Path(self.cached_states_dir)
        if not (d / TRAIN_FILENAME).exists():
            d = d / FUSION_STATES_DIR
        train_path, val_path = d / TRAIN_FILENAME, d / VAL_FILENAME
        for p in (train_path, val_path):
            if not p.exists():
                raise FileNotFoundError(f"cached fusion states not found: {p}")

        self.train_td = _load_td(train_path, "train")
        self.val_td = _load_td(val_path, "val")
        log.info(
            "loaded_cached_states",
            dir=str(d),
            train_n=self.train_td.batch_size[0],
            val_n=self.val_td.batch_size[0],
            keys=list(self.train_td.keys(include_nested=True, leaves_only=True)),
        )

    def _batches(self, td: TensorDict, *, shuffle: bool):
        n = td.batch_size[0]
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, self._batch_size):
            sub = td[idx[start : start + self._batch_size]]
            yield sub.exclude("labels"), sub["labels"]

    def train_dataloader(self):
        return self._batches(self.train_td, shuffle=True)

    def val_dataloader(self):
        return self._batches(self.val_td, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()
