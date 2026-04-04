"""Frozen VGAE+GAT state caching for fusion training.

Independent of the graph family — serves ``TensorDataset`` batches of
precomputed state vectors, not PyG graph batches. Two paths:

- **Fast path** (``cached_states_dir`` set): load pre-extracted state
  tensors from disk. No GPU needed.
- **Slow path**: load VGAE + GAT checkpoints, run them over CAN data,
  concatenate registered extractor outputs into ``[N, total_dim]`` tensors.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
from graphids.core.preprocessing.sampler import make_graph_loader
from graphids.log import get_logger

from .graph import load_datasets

log = get_logger(__name__)


class FusionDataModule(pl.LightningDataModule):
    """Loads frozen VGAE+GAT, caches state vectors, serves DataLoaders.

    Wraps CAN data loading internally — callers never touch raw graph data.
    """

    def __init__(
        self,
        dataset: str = "",
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
        vgae_ckpt_path: str = "",
        gat_ckpt_path: str = "",
        cached_states_dir: str = "",
        method: str = "bandit",
        batch_size: int = 128,
        episode_sample_size: int = 20000,
        max_samples: int = 150000,
        max_val_samples: int = 30000,
        eval_batch_size: int = 256,
        seed: int = 42,
        window_size: int = 100,
        stride: int = 100,
        val_fraction: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        is_rl = method in ("dqn", "bandit")
        self._batch_size = episode_sample_size if is_rl else batch_size
        self.train_cache: dict | None = None
        self.val_cache: dict | None = None

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_cache["states"]) / self._batch_size)

    @staticmethod
    def cache_predictions(
        models: dict[str, torch.nn.Module],
        data,
        device: torch.device,
        max_samples: int = 150_000,
        batch_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Run registered extractors over data, produce N-D state vectors for fusion."""
        from graphids.core.models.fusion.fusion_features import EXTRACTORS

        active = [(name, ext) for name, ext in EXTRACTORS.items() if name in models]
        for model in models.values():
            model.eval()

        capped = data[:max_samples]
        loader = make_graph_loader(capped, batch_size=batch_size)

        states, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                feats = [ext.extract(models[name], batch, device) for name, ext in active]
                states.append(torch.cat(feats, dim=1))  # [B, total_dim]
                labels.append(batch.y)

        return {"states": torch.cat(states), "labels": torch.cat(labels)}

    def setup(self, stage=None):
        if self.train_cache is not None:
            return

        hp = self.hparams

        # Fast path: load pre-extracted states from disk (no GPU needed)
        if hp.cached_states_dir:
            self._load_cached_states(hp.cached_states_dir)
            return

        # Slow path: load upstream models and extract on GPU
        from graphids.core.models._training import load_inner_model

        train_ds, val_ds, _ = load_datasets(
            dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
            window_size=hp.window_size, stride=hp.stride,
            train_val_split=1.0 - hp.val_fraction,
            dataset_cls=CANBusDataset,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not hp.vgae_ckpt_path:
            raise ValueError("vgae_ckpt_path is empty — upstream VGAE checkpoint not wired")
        if not hp.gat_ckpt_path:
            raise ValueError("gat_ckpt_path is empty — upstream GAT checkpoint not wired")
        vgae, _ = load_inner_model("vgae", Path(hp.vgae_ckpt_path), device)
        gat, _ = load_inner_model("gat", Path(hp.gat_ckpt_path), device)

        # Fusion pre-flight: warn if both models consume > 85% of VRAM
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            usage_pct = allocated / total * 100
            if usage_pct > 85:
                log.warning("fusion_setup_vram_high",
                            allocated_mb=round(allocated / 1e6, 1),
                            total_mb=round(total / 1e6, 1),
                            pct=round(usage_pct, 1))

        models = {"vgae": vgae, "gat": gat}
        self.train_cache = self.cache_predictions(
            models, list(train_ds), device, hp.max_samples, batch_size=hp.eval_batch_size,
        )
        self.val_cache = self.cache_predictions(
            models, list(val_ds), device, hp.max_val_samples, batch_size=hp.eval_batch_size,
        )

        del vgae, gat, models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_cached_states(self, cached_states_dir: str) -> None:
        """Load pre-extracted fusion states from disk. No GPU needed."""
        from graphids.commands.extract_fusion_states import (
            FUSION_STATES_DIR,
            TRAIN_FILENAME,
            VAL_FILENAME,
        )

        states_dir = Path(cached_states_dir)
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
        log.info("loaded_cached_states",
                 dir=str(states_dir),
                 train_shape=list(self.train_cache["states"].shape),
                 val_shape=list(self.val_cache["states"].shape))

    def train_dataloader(self):
        ds = TensorDataset(self.train_cache["states"], self.train_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_cache["states"], self.val_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size)

    def test_dataloader(self):
        return self.val_dataloader()
