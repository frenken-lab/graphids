"""Fusion data module for pre-extracted TensorDict caches."""

from __future__ import annotations

import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from structlog import get_logger
from tensordict import TensorDict

from graphids.core.data.extract import (
    CACHE_VERSION,
    TRAIN_FILENAME,
    VAL_FILENAME,
)

log = get_logger(__name__)


def _load_td(path: Path, which: str) -> tuple[TensorDict, dict[int, str]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    version = blob.get("version") if isinstance(blob, dict) else None
    if version != CACHE_VERSION:
        raise RuntimeError(
            f"fusion {which} cache at {path} has version={version!r}, "
            f"expected {CACHE_VERSION}; re-run extraction"
        )
    td = TensorDict(blob["td"], batch_size=[blob["td"]["labels"].size(0)])
    return td, dict(blob.get("attack_type_names") or {0: "benign"})


class _TensorDictBatches:
    def __init__(self, td: TensorDict, batch_size: int, *, shuffle: bool):
        self._td = td
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return math.ceil(self._td.batch_size[0] / self._batch_size)

    def __iter__(self):
        n = self._td.batch_size[0]
        idx = torch.randperm(n) if self._shuffle else torch.arange(n)
        for start in range(0, n, self._batch_size):
            sub = self._td[idx[start : start + self._batch_size]]
            yield sub.exclude("labels"), sub["labels"]


class FusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cached_states_dir: Path | None = None,
        method: str = "bandit",
        batch_size: int = 128,
        episode_sample_size: int = 20000,
    ):
        super().__init__()
        self.cached_states_dir = (
            Path(cached_states_dir) if cached_states_dir is not None else None
        )
        self.method = method
        self._batch_size = (
            episode_sample_size if method in ("dqn", "bandit") else batch_size
        )
        self.train_td: TensorDict | None = None
        self.val_td: TensorDict | None = None
        self._test_tds: dict[str, TensorDict] = {}
        self.attack_type_names: dict[int, str] = {0: "benign"}

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.train_td.batch_size[0] / self._batch_size)

    def setup(self, stage: str | None = None) -> None:
        if self.train_td is not None:
            return
        if self.cached_states_dir is None:
            raise ValueError(
                "cached_states_dir is required — run extraction first"
            )
        d = self.cached_states_dir
        train_path, val_path = d / TRAIN_FILENAME, d / VAL_FILENAME
        for p in (train_path, val_path):
            if not p.exists():
                raise FileNotFoundError(f"cached fusion states not found: {p}")

        self.train_td, _ = _load_td(train_path, "train")
        self.val_td, self.attack_type_names = _load_td(val_path, "val")

        for p in sorted(d.glob("*_states.pt")):
            if p.name in (TRAIN_FILENAME, VAL_FILENAME):
                continue
            name = p.stem.replace("_states", "")
            self._test_tds[name], _ = _load_td(p, name)

        log.info(
            "loaded_cached_states",
            dir=str(d),
            train_n=self.train_td.batch_size[0],
            val_n=self.val_td.batch_size[0],
            test_splits=list(self._test_tds.keys()),
            keys=list(self.train_td.keys(include_nested=True, leaves_only=True)),
        )

    def train_dataloader(self):
        return _TensorDictBatches(self.train_td, self._batch_size, shuffle=True)

    def val_dataloader(self):
        return _TensorDictBatches(self.val_td, self._batch_size, shuffle=False)

    @property
    def test_datasets(self) -> dict[str, TensorDict]:
        return self._test_tds

    def test_dataloader(self):
        if self._test_tds:
            return [
                _TensorDictBatches(td, self._batch_size, shuffle=False)
                for td in self._test_tds.values()
            ]
        return self.val_dataloader()
