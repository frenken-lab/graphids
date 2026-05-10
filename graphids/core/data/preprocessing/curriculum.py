"""Curriculum difficulty scorers used by the graph datamodule."""

from __future__ import annotations

import gc
from pathlib import Path

import torch


@torch.no_grad()
def score_vgae(graphs: list, ckpt_path: str) -> torch.Tensor:
    """Per-graph reconstruction MSE from a trained VGAE checkpoint.

    Higher = harder. Loads the VGAE on CPU, computes per-graph mean MSE
    via ``torch_geometric.utils.scatter``, releases the model.
    """
    if not ckpt_path:
        raise ValueError("score_vgae requires a non-empty ckpt_path")

    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.utils import scatter

    from graphids.core.models.base import safe_load_checkpoint

    vgae = safe_load_checkpoint("vgae", Path(ckpt_path), map_location="cpu")
    try:
        device = next(vgae.parameters()).device
        was_training = vgae.training
        vgae.eval()
        try:
            scores: list[float] = []
            for batch in PyGDataLoader(graphs, batch_size=500):
                batch = batch.clone().to(device, non_blocking=True)
                cont, _canid, _nbr, _z, _kl, _edge = vgae(batch)
                node_mse = (cont - batch.x).pow(2).mean(dim=1)
                graph_mse = scatter(node_mse, batch.batch, reduce="mean")
                scores.extend(graph_mse.tolist())
        finally:
            vgae.train(was_training)
    finally:
        del vgae
        gc.collect()
    return torch.tensor(scores, dtype=torch.float)


def score_random(graphs: list, seed: int = 0) -> torch.Tensor:
    """Uniform random per-graph difficulty for curriculum control runs."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(len(graphs), generator=g)
