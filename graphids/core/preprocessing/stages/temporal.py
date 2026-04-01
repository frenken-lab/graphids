"""Temporal grouping of graph snapshots for sequence-based classification.

Groups N ordered graphs into overlapping windows of size W with stride S.
Each window becomes a GraphSequence with a label: attack (1) if any graph
in the window has an attack label, else normal (0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from torch_geometric.data import Data

log = structlog.get_logger()


@dataclass
class GraphSequence:
    """A temporal sequence of consecutive graph snapshots."""

    graphs: list[Data]
    y: int  # 1 if any graph in sequence has attack label


class TemporalGrouper:
    """Sliding window over ordered graphs to create temporal sequences.

    Args:
        window: Number of consecutive graphs per sequence.
        stride: Step size between windows.
    """

    def __init__(self, window: int = 8, stride: int = 1):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.window = window
        self.stride = stride

    def group(self, graphs: list[Data]) -> list[GraphSequence]:
        """Group ordered graphs into overlapping temporal sequences.

        Args:
            graphs: List of PyG Data objects in temporal order.

        Returns:
            List of GraphSequence objects.
        """
        sequences: list[GraphSequence] = []
        n = len(graphs)

        for start in range(0, n - self.window + 1, self.stride):
            window_graphs = graphs[start : start + self.window]
            # Label: 1 if any graph in window is attack
            label = 0
            for g in window_graphs:
                g_label = g.y.item() if g.y.dim() == 0 else int(g.y[0].item())
                if g_label == 1:
                    label = 1
                    break
            sequences.append(GraphSequence(graphs=window_graphs, y=label))

        return sequences


class TemporalGraphDataset(Dataset):
    """PyTorch Dataset wrapping a list of GraphSequence objects."""

    def __init__(self, sequences: list[GraphSequence], device: torch.device):
        self.sequences = sequences
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq.graphs, seq.y


def collate_temporal(batch):
    """Custom collate for temporal graph sequences.

    Returns:
        graph_sequences: list of lists of Data objects
        labels: tensor of labels
    """
    graph_sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return graph_sequences, labels


# ---------------------------------------------------------------------------
# Temporal data module: wraps CANBusDataModule with temporal sequence grouping
# ---------------------------------------------------------------------------


class TemporalDataModule(pl.LightningDataModule):
    """Loads graph data, groups into temporal sequences, serves DataLoaders.

    During setup, loads a pretrained GAT to probe the spatial embedding dim.
    The GAT reference is stored for build_module to consume (avoids loading twice).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.spatial_dim: int | None = None
        self.gat: torch.nn.Module | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def setup(self, stage=None):
        from pathlib import Path

        from graphids.core.models._training import load_inner_model
        from .datamodule import load_datasets

        train_ds, val_ds, _ = load_datasets(self.cfg)
        tc = self.cfg.temporal

        # Load pretrained GAT and probe spatial embedding dim
        self.gat, _ = load_inner_model("gat", Path(self.cfg.checkpoints["gat"]), self._device)
        with torch.no_grad():
            probe = train_ds[0].clone().to(self._device, non_blocking=True)
            _, emb = self.gat(probe, return_embedding=True)
            self.spatial_dim = emb.shape[-1]
        log.info("spatial_embedding_dim", dim=self.spatial_dim)

        # Contiguous time split: first train_split% train, rest val
        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        all_graphs = list(train_ds) + list(val_ds)
        split_idx = int(len(all_graphs) * tc.train_split)

        self._train_sequences = grouper.group(all_graphs[:split_idx])
        self._val_sequences = grouper.group(all_graphs[split_idx:])

        log.info(
            "temporal_sequences",
            train=len(self._train_sequences),
            val=len(self._val_sequences),
            total_graphs=len(all_graphs),
        )

        if not self._train_sequences or not self._val_sequences:
            raise ValueError("Not enough graphs for temporal windowing")

        self._batch_size = (
            tc.batch_size if tc.batch_size > 0
            else max(1, min(32, len(self._train_sequences) // 10))
        )

    def train_dataloader(self):
        return DataLoader(
            TemporalGraphDataset(self._train_sequences, self._device),
            batch_size=self._batch_size, shuffle=True,
            collate_fn=collate_temporal, num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            TemporalGraphDataset(self._val_sequences, self._device),
            batch_size=self._batch_size, shuffle=False,
            collate_fn=collate_temporal, num_workers=0,
        )
