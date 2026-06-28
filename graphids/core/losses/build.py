"""Loss reconstruction for checkpoint loading.

Single consumer: :func:`graphids.core.models.base.safe_load_checkpoint`.
``loss_fn: nn.Module`` is excluded from saved hparams (it's a Module, not a
config value), so when a checkpoint is reloaded for inference the loss has to
be rebuilt from saved hparam keys.

For new (post-class_path-lift) ckpts, hparams won't carry loss-shaping keys
(libsonnets emit a ``loss_fn`` class_path block consumed at instantiate, not
re-saved). The reloaded loss falls back to defaults — fine for the
inference-only use sites listed above.
"""

from __future__ import annotations

from typing import Any

_LOSS_MODEL_TYPES = frozenset({"temporal_event_classifier", "temporal_gat"})


def build_loss(
    model_type: str | None,
    loss_config: dict[str, Any] | None,
):
    """Reconstruct an ``nn.Module`` loss for a reloaded checkpoint.

    Returns ``None`` for model types that own their loss internally.
    """
    if model_type not in _LOSS_MODEL_TYPES:
        return None

    from graphids.core.losses import (
        CrossEntropyLoss,
        FocalLoss,
        WeightedCrossEntropyLoss,
    )

    cfg = dict(loss_config or {})

    if model_type in {"temporal_event_classifier", "temporal_gat"}:
        loss_type = cfg.pop("type", "ce")
        if loss_type == "focal":
            return FocalLoss(gamma=cfg.get("gamma", 2.0))
        if loss_type == "weighted_ce":
            return WeightedCrossEntropyLoss(weights=cfg["weights"])
        if loss_type == "ce":
            return CrossEntropyLoss()
        raise ValueError(
            f"Unknown loss type {loss_type!r} for temporal classifier. Expected: ce, focal, weighted_ce."
        )
