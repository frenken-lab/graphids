"""Curriculum learning — annotators + schedule + loss wrapper.

One feature, one file. Three primitives:

1. **Annotators** — ``score_vgae`` / ``score_random``: per-graph
   ``Tensor[float]`` produced once at ``dm.setup``, attached as
   ``Data.difficulty``. NOT preprocessing — they run after the cache is
   built and depend on a trained upstream model (or no model, for
   ``score_random``). Annotation, not data transformation.

2. **Schedule** — ``LinearRampSchedule``: pure callable
   ``(epoch, difficulty, in_scope) -> weights``. Called at every
   training step. Zero data dependencies.

3. **Loss wrapper** — ``CurriculumWeightedLoss``: wraps any per-example
   base loss (``reduction='none'``); reduces ``per_ex * weights`` with
   schedule-derived weights.

Data-side integration: ``GraphDataModule._attach_curriculum`` calls the
chosen annotator at ``setup`` and writes ``g.difficulty`` + ``g.in_scope``
on each train graph. PyG's ``Batch.from_data_list`` collates them so the
loss wrapper reads ``batch.difficulty`` / ``batch.in_scope`` directly.

Eval: construct a non-wrapped loss, OR call ``set_epoch`` past
``schedule.max_epochs - 1`` so all weights are 1.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
import torch.nn as nn

# ─── §1. Annotators (per-graph difficulty scorers) ───────────────────


@torch.no_grad()
def score_vgae(graphs: list, ckpt_path: str) -> torch.Tensor:
    """Per-graph reconstruction MSE from a trained VGAE checkpoint.

    Higher = harder. Loads the VGAE on CPU, forwards each graph,
    aggregates node MSE per graph via ``torch_geometric.utils.scatter``,
    releases the model.
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
    independent of any learned difficulty ordering. Compare against:
    no-wrapper baseline (regular CE/focal) and ``score_vgae``.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(len(graphs), generator=g)


# ─── §2. Schedule (pacing) ───────────────────────────────────────────


class LinearRampSchedule:
    """Pure callable: ``(epoch, difficulty, in_scope) -> binary weights``.

    - Out-of-scope (``in_scope == False``): always weight 1.
    - In-scope, in active set: weight 1. Active set = ``fraction(epoch) * n_in_scope``
      easiest in-scope examples.
    - In-scope, not yet unlocked: weight 0.

    Defaults match the prior tier schedule: 10% in-scope visible at
    start, 100% by ``max_epochs - 1``.
    """

    def __init__(
        self,
        start_ratio: float = 1.0,
        end_ratio: float = 10.0,
        max_epochs: int = 300,
    ):
        if end_ratio <= 0:
            raise ValueError(f"end_ratio must be positive, got {end_ratio}")
        if start_ratio <= 0 or start_ratio > end_ratio:
            raise ValueError(
                f"start_ratio must be in (0, end_ratio]; got start={start_ratio}, end={end_ratio}"
            )
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)
        self.max_epochs = int(max_epochs)

    def fraction(self, epoch: int) -> float:
        progress = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        return ratio / self.end_ratio

    def __call__(
        self,
        epoch: int,
        difficulty: torch.Tensor,
        in_scope: torch.Tensor,
    ) -> torch.Tensor:
        if difficulty.shape != in_scope.shape:
            raise ValueError(
                f"difficulty {tuple(difficulty.shape)} and in_scope "
                f"{tuple(in_scope.shape)} must have the same shape"
            )
        in_scope_b = in_scope.bool()
        weights = (~in_scope_b).to(torch.float)  # out-of-scope → 1, in-scope → 0
        in_idx = in_scope_b.nonzero(as_tuple=True)[0]
        n_in = int(in_idx.numel())
        if n_in == 0:
            return weights

        n_active = max(1, min(n_in, math.ceil(self.fraction(epoch) * n_in)))
        easiest = torch.topk(difficulty[in_idx], n_active, largest=False).indices
        weights[in_idx[easiest]] = 1.0
        return weights


# ─── §3. Loss wrapper ────────────────────────────────────────────────


class CurriculumWeightedLoss(nn.Module):
    """Per-example masking via ``(difficulty, schedule)``.

    ``base_loss`` must produce per-example output (``reduction='none'``);
    this wrapper reduces with schedule weights. ``epoch`` is set
    externally via :meth:`set_epoch` from a Lightning hook (typically
    ``on_train_epoch_start``), so the loss forward signature
    ``(logits, labels, graph)`` stays stable across the codebase.
    """

    def __init__(self, base_loss: nn.Module, schedule):
        super().__init__()
        if hasattr(base_loss, "reduction") and base_loss.reduction != "none":
            raise ValueError(
                f"CurriculumWeightedLoss requires base_loss.reduction='none', "
                f"got {base_loss.reduction!r}"
            )
        self.base_loss = base_loss
        self.schedule = schedule
        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        if graph is None:
            raise ValueError(
                "CurriculumWeightedLoss requires the `graph` (Batch) arg "
                "to read batch.difficulty and batch.in_scope"
            )
        if not hasattr(graph, "difficulty") or not hasattr(graph, "in_scope"):
            raise ValueError(
                "graph is missing curriculum attributes — datamodule must be "
                "configured with a `difficulty` spec so Batch carries "
                "difficulty + in_scope"
            )
        per_ex = self.base_loss(logits, labels)
        if per_ex.dim() != 1:
            raise ValueError(
                f"base_loss must return per-example loss (1-D); got shape {tuple(per_ex.shape)}"
            )
        weights = self.schedule(self._epoch, graph.difficulty, graph.in_scope).to(
            per_ex.device, per_ex.dtype
        )
        return (per_ex * weights).sum() / weights.sum().clamp_min(1.0)
