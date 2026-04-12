"""Curriculum learning: pluggable difficulty scoring + tier bucketing + epoch callback.

Decoupled into three concerns:

- **Scoring strategy** — any class with ``score(graphs) -> Tensor``.
  Built-ins: :class:`VGAEScorer` (reconstruction difficulty via a VGAE
  checkpoint), :class:`RandomScorer` (uniform baseline).
- **Tier bucketing** — :func:`build_curriculum_tiers` is scorer-agnostic;
  pass any :class:`DifficultyScorer` and it handles normals/attacks split,
  argsort, and equal-width bucket construction.
- **Epoch gating** — :class:`CurriculumEpochCallback` advances the
  datamodule's active tier set each epoch.

Config format (consistent with the rest of the repo): scorer specs use
``{class_path: "pkg.Cls", init_args: {...}}`` and resolve via
:func:`make_scorer`.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from graphids._reflect import import_class
from graphids.core.callbacks import CallbackBase

# ---------------------------------------------------------------------------
# Scoring strategy — protocol + built-in implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class DifficultyScorer(Protocol):
    """Assigns a per-graph difficulty score; higher = harder for the curriculum."""

    def score(self, graphs: list) -> torch.Tensor:
        """Return a 1-D tensor of length ``len(graphs)`` with per-graph scores."""
        ...


class VGAEScorer:
    """Reconstruction-difficulty scoring via a trained VGAE checkpoint.

    Loads the VGAE on CPU, scores via ``model.score_difficulty``, then
    releases the VGAE immediately (curriculum scoring runs once at setup
    and we don't want to hold the checkpoint in RAM for the whole run).
    """

    def __init__(self, ckpt_path: str, canid_weight: float = 0.1) -> None:
        if not ckpt_path:
            raise ValueError("VGAEScorer requires a non-empty ckpt_path")
        self.ckpt_path = ckpt_path
        self.canid_weight = canid_weight

    def score(self, graphs: list) -> torch.Tensor:
        from graphids.core.models.base import load_inner_model

        vgae, _ = load_inner_model("vgae", Path(self.ckpt_path), torch.device("cpu"))
        try:
            raw = vgae.score_difficulty(graphs, canid_weight=self.canid_weight)
        finally:
            del vgae
            gc.collect()
        return raw if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float)


class RandomScorer:
    """Uniform random difficulty scores — reference baseline for curriculum."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def score(self, graphs: list) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        return torch.rand(len(graphs), generator=g)


# ---------------------------------------------------------------------------
# Scorer spec resolution
# ---------------------------------------------------------------------------


def make_scorer(spec: Any) -> DifficultyScorer:
    """Resolve a scorer spec into a :class:`DifficultyScorer` instance.

    Accepted forms:

    * A scorer instance (anything with ``score(graphs)`` — returned as-is)
    * ``{"class_path": "pkg.mod.Cls", "init_args": {...}}`` — imported and instantiated

    Raises ``ValueError`` for ``None`` or malformed specs.
    """
    if spec is None:
        raise ValueError(
            "curriculum sampler requires a scorer spec; "
            "set data.init_args.scorer to a {class_path, init_args} dict "
            "(e.g. graphids.core.data.curriculum.VGAEScorer)"
        )
    if isinstance(spec, DifficultyScorer):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(
            f"scorer spec must be a dict or DifficultyScorer, got {type(spec).__name__}"
        )
    if "class_path" not in spec:
        raise ValueError(f"scorer spec missing 'class_path': {spec!r}")
    cls = import_class(spec["class_path"])
    return cls(**spec.get("init_args", {}))


# ---------------------------------------------------------------------------
# Bucketing — pure index math over a score vector
# ---------------------------------------------------------------------------


def bucket_by_score(scores: torch.Tensor, num_tiers: int) -> list[list[int]]:
    """Sort indices by score (ascending) and split into ``num_tiers`` equal-size bins.

    Pure function over ``scores``: returns a list of index lists where tier 0
    holds the lowest-scoring indices and tier K-1 holds the highest. The last
    tier may be smaller when ``len(scores)`` is not divisible by ``num_tiers``.
    """
    n = scores.numel()
    if n == 0:
        raise ValueError("bucket_by_score: empty score tensor")
    if num_tiers < 1:
        raise ValueError(f"bucket_by_score: num_tiers must be >= 1, got {num_tiers}")
    order = torch.argsort(scores).tolist()
    step = max(1, math.ceil(n / num_tiers))
    return [order[i:i + step] for i in range(0, n, step)]


# ---------------------------------------------------------------------------
# Gating — pure epoch-to-active-tier-count schedule
# ---------------------------------------------------------------------------


def active_tier_count(
    epoch: int, num_tiers: int, *,
    start_ratio: float, end_ratio: float, max_epochs: int,
) -> int:
    """How many curriculum tiers should be active at ``epoch``.

    Linear ramp on an opaque "ratio" parameter that scales to tier count:

        progress ∈ [0, 1]      = min(epoch / (max_epochs - 1), 1)
        ratio                  = start_ratio + (end_ratio - start_ratio) * progress
        count                  = ceil(ratio * num_tiers / end_ratio)

    Clamped to ``[1, num_tiers]``. With defaults (start=1.0, end=10.0,
    num_tiers=10), one new tier unlocks every ``max_epochs / num_tiers``
    epochs; once unlocked a tier stays active for the rest of training.
    """
    progress = min(epoch / max(max_epochs - 1, 1), 1.0)
    ratio = start_ratio + (end_ratio - start_ratio) * progress
    count = math.ceil(ratio * num_tiers / end_ratio)
    return max(1, min(num_tiers, count))


# ---------------------------------------------------------------------------
# Composer — split labels, score, bucket, attach attacks
# ---------------------------------------------------------------------------


def build_curriculum_tiers(
    train_ds, scorer: DifficultyScorer, *, num_tiers: int = 10,
) -> tuple[torch.Tensor, list[list[int]], list[int], list, torch.Tensor]:
    """Wire the three curriculum pieces into a single setup-time call.

    Composes, in order:

    1. Label split — ``normals`` (``y == 0``) vs ``attacks`` (``y == 1``)
    2. ``scorer.score(normals)`` → 1-D score tensor
    3. :func:`bucket_by_score` → per-tier normal indices

    Attacks are not bucketed — they come back as a flat ``attack_indices``
    list that the datamodule concatenates onto every active tier set.

    Returns ``(scores, normal_tier_indices, attack_indices, full_dataset, dataset_sizes)``.
    """
    normals = [g for g in train_ds if int(g.y[0]) == 0]
    attacks = [g for g in train_ds if int(g.y[0]) == 1]

    scores = scorer.score(normals)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float)
    if scores.numel() != len(normals):
        raise ValueError(
            f"scorer returned {scores.numel()} scores for {len(normals)} graphs"
        )

    normal_tier_indices = bucket_by_score(scores, num_tiers)

    full_dataset = normals + attacks
    dataset_sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)
    attack_indices = list(range(len(normals), len(full_dataset)))

    return scores, normal_tier_indices, attack_indices, full_dataset, dataset_sizes


# ---------------------------------------------------------------------------
# Epoch callback — thin wrapper around the datamodule's gate
# ---------------------------------------------------------------------------


class CurriculumEpochCallback(CallbackBase):
    """Advance active tiers each epoch. No-op when datamodule isn't tier-batched."""

    def on_train_epoch_start(self, trainer, model):
        dm = trainer.datamodule
        if getattr(dm, "_tier_batches", None) is not None:
            dm._select_active_tiers(trainer.current_epoch)
