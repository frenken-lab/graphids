"""Fusion DataModule — serves pre-extracted VGAE+GAT state vectors.

Loads cached state tensors from disk (produced by
``python -m graphids extract-fusion-states``). Independent of the graph
family — serves ``TensorDataset`` batches of dense vectors, not PyG graphs.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids._otel import get_logger
from graphids.core.data.fusion_states import (
    CACHE_VERSION,
    FUSION_STATES_DIR,
    TRAIN_FILENAME,
    VAL_FILENAME,
)

log = get_logger(__name__)


class FusionDataModule:
    """Loads pre-extracted VGAE+GAT state vectors, serves DataLoaders.

    Requires ``cached_states_dir`` pointing to the output of
    ``python -m graphids extract-fusion-states``. If not set, raises
    with instructions.
    """

    def __init__(
        self,
        cached_states_dir: str = "",
        method: str = "bandit",
        batch_size: int = 128,
        episode_sample_size: int = 20000,
    ):
        self.hparams = {k: v for k, v in locals().items() if k != "self"}
        is_rl = method in ("dqn", "bandit")
        self._batch_size = episode_sample_size if is_rl else batch_size
        self.train_cache: dict | None = None
        self.val_cache: dict | None = None

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_cache["states"]) / self._batch_size)

    def _set_device(self, device: torch.device | None) -> None:
        # Fusion batches stay on CPU; fusion modules move tensors inside forward.
        pass

    def _set_model(self, model) -> None:
        pass

    def setup(self, stage=None):
        if self.train_cache is not None:
            return

        hp = self.hparams
        if not hp["cached_states_dir"]:
            raise ValueError(
                "cached_states_dir is required — run "
                "'python -m graphids extract-fusion-states' first"
            )

        states_dir = Path(hp["cached_states_dir"])
        # Support both direct dir and parent dir containing fusion_states/
        if not (states_dir / TRAIN_FILENAME).exists():
            states_dir = states_dir / FUSION_STATES_DIR
        train_path = states_dir / TRAIN_FILENAME
        val_path = states_dir / VAL_FILENAME
        if not train_path.exists():
            raise FileNotFoundError(f"Cached train states not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Cached val states not found: {val_path}")

        self.train_cache = torch.load(train_path, map_location="cpu", weights_only=True)
        self.val_cache = torch.load(val_path, map_location="cpu", weights_only=True)
        for which, cache, path in (
            ("train", self.train_cache, train_path),
            ("val", self.val_cache, val_path),
        ):
            v = cache.get("version")
            if v != CACHE_VERSION:
                raise RuntimeError(
                    f"Fusion {which} cache at {path} has version={v}, expected "
                    f"{CACHE_VERSION}. Re-run "
                    "'python -m graphids extract-fusion-states' to regenerate "
                    "(VGAE 8-D feature columns changed in the mask-recon synthesis)."
                )
        log.info(
            "loaded_cached_states",
            dir=str(states_dir),
            train_shape=list(self.train_cache["states"].shape),
            val_shape=list(self.val_cache["states"].shape),
        )

    def train_dataloader(self):
        ds = TensorDataset(self.train_cache["states"], self.train_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_cache["states"], self.val_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size)

    def test_dataloader(self):
        return self.val_dataloader()
