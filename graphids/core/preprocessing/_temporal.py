"""Temporal grouping of graph snapshots for sequence-based classification.

Groups N ordered graphs into overlapping windows of size W with stride S.
Each window becomes a GraphSequence with a label: attack (1) if any graph
in the window has an attack label, else normal (0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data


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
