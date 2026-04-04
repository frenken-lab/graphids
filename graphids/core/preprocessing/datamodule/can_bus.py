"""CAN bus domain binding for ``GraphDataModule``.

Adding a new graph dataset (Ethernet, water-treatment, etc.) is a single
new file alongside this one, subclassing ``GraphDataModule`` with the
concrete ``dataset_cls`` — no other file needs to change.
"""

from __future__ import annotations

from graphids.core.preprocessing.datasets.can_bus import CANBusDataset

from .graph import GraphDataModule


class CANBusDataModule(GraphDataModule):
    """CAN bus graph data — one DataModule for all 6 catalog datasets."""

    dataset_cls = CANBusDataset
