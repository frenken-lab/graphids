"""Curriculum difficulty scorers — preprocessing only.

Per-graph scalar scores produced **once** at ``dm.setup`` and attached
as ``Data.difficulty`` on each train graph; PyG ``Batch.from_data_list``
collates them automatically. The schedule + loss wrapper that consume
``batch.difficulty`` live in :mod:`graphids.core.losses.curriculum`.

New scorers are new free functions: ``f(graphs, **kwargs) -> Tensor``.
"""

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
    """Uniform random per-graph difficulty — control for the curriculum
    *mechanism*, not "no curriculum".

    Tests whether per-epoch reweighting/hiding contributes signal
    independent of any learned difficulty ordering. Compare against the
    no-wrapper baseline (regular CE/focal) and ``score_vgae``.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(len(graphs), generator=g)


class ScoreRandom:
    """Callable wrapper for ``score_random`` — lets the plan system instantiate
    via ``spec(SCORE_RANDOM, seed=N)`` without passing ``graphs`` at construct time."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def __call__(self, graphs: list) -> torch.Tensor:
        return score_random(graphs, seed=self.seed)


class ScoreVGAE:
    """Callable wrapper for ``score_vgae`` — lets the plan system instantiate
    via ``spec(SCORE_VGAE, ckpt_path=...)`` without passing ``graphs`` at construct time."""

    def __init__(self, ckpt_path: str) -> None:
        self.ckpt_path = ckpt_path

    def __call__(self, graphs: list) -> torch.Tensor:
        return score_vgae(graphs, ckpt_path=self.ckpt_path)
