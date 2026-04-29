"""Loss factory — builds base + optional KD wrapper from config dicts.

Dispatches on ``model_type``:
- ``gat``: classification loss (CE / focal / weighted CE) + optional SoftLabelDistillation
- ``vgae``: VGAETaskLoss + optional FeatureDistillation with projection
- others: returns ``None`` (fusion/dgi handle loss internally)
"""

from __future__ import annotations

from typing import Any

_LOSS_MODEL_TYPES = frozenset({"gat", "vgae"})

# Loss params that live at init_args top-level in jsonnet but belong to VGAETaskLoss
_VGAE_LOSS_KEYS = frozenset({"kl_weight", "canid_weight", "nbr_weight", "k_neg"})

_CLASS_PATH_TO_MODEL_TYPE: dict[str, str] = {
    "VGAEModule": "vgae",
    "GATModule": "gat",
    "DGIModule": "dgi",
}


def build_loss(
    model_type: str | None,
    loss_config: dict[str, Any] | None,
    distillation_config: dict[str, Any] | None,
):
    """Return an ``nn.Module`` loss for ``model_type`` or ``None`` if N/A."""
    if model_type not in _LOSS_MODEL_TYPES:
        return None

    import torch
    import torch.nn as nn

    from graphids.core.losses import (
        CrossEntropyLoss,
        FeatureDistillation,
        FocalLoss,
        SoftLabelDistillation,
        VGAETaskLoss,
        WeightedCrossEntropyLoss,
    )
    from graphids.core.models.base import load_inner_model

    loss_cfg = dict(loss_config or {})

    if model_type == "gat":
        loss_type = loss_cfg.pop("type", "ce")
        if loss_type == "focal":
            base: nn.Module = FocalLoss(gamma=loss_cfg.get("gamma", 2.0))
        elif loss_type == "weighted_ce":
            base = WeightedCrossEntropyLoss(weights=loss_cfg["weights"])
        elif loss_type == "ce":
            base = CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unknown loss type {loss_type!r} for gat. Expected one of: ce, focal, weighted_ce."
            )

        if not distillation_config:
            return base

        teacher_ckpt = distillation_config["teacher_ckpt"]
        teacher, _ = load_inner_model("gat", teacher_ckpt, torch.device("cpu"))
        return SoftLabelDistillation(
            base,
            teacher,
            temperature=distillation_config.get("temperature", 4.0),
            alpha=distillation_config.get("alpha", 0.7),
        )

    # model_type == "vgae"
    base = VGAETaskLoss(
        kl_weight=loss_cfg.get("kl_weight", 0.01),
        canid_weight=loss_cfg.get("canid_weight", 0.1),
        nbr_weight=loss_cfg.get("nbr_weight", 0.05),
        k_neg=loss_cfg.get("k_neg", 32),
    )

    if not distillation_config:
        return base

    teacher_ckpt = distillation_config["teacher_ckpt"]
    teacher, teacher_hparams = load_inner_model("vgae", teacher_ckpt, torch.device("cpu"))

    projection: nn.Linear | None = None
    s_dim = distillation_config.get("student_latent_dim")
    t_dim = getattr(teacher_hparams, "latent_dim", None)
    if s_dim and t_dim and s_dim != t_dim:
        projection = nn.Linear(s_dim, t_dim)

    return FeatureDistillation(
        base,
        teacher,
        latent_weight=distillation_config.get("latent_weight", 1.0),
        recon_weight=distillation_config.get("recon_weight", 1.0),
        alpha=distillation_config.get("alpha", 0.7),
        projection=projection,
    )


def inject_loss_fn(model_init_args: dict[str, Any], *, class_path: str = "") -> dict[str, Any]:
    """Pop loss/distillation config from init_args, build loss, inject as ``loss_fn``.

    Returns a NEW dict — leaves the caller's copy alone.
    """
    init_args = dict(model_init_args)
    init_args.pop("auxiliaries", None)

    loss_cfg = init_args.pop("loss_config", None)
    kd_cfg = init_args.pop("distillation_config", None)
    model_type = init_args.get("model_type")

    if not model_type and class_path:
        cls_name = class_path.rsplit(".", 1)[-1]
        model_type = _CLASS_PATH_TO_MODEL_TYPE.get(cls_name)

    if model_type == "vgae" and loss_cfg is None:
        loss_cfg = {k: init_args.pop(k) for k in _VGAE_LOSS_KEYS if k in init_args}

    if kd_cfg and model_type == "vgae" and "student_latent_dim" not in kd_cfg:
        latent_dim = init_args.get("latent_dim")
        if latent_dim is not None:
            kd_cfg = {**kd_cfg, "student_latent_dim": latent_dim}

    loss_fn = build_loss(model_type, loss_cfg, kd_cfg)
    if loss_fn is not None:
        init_args["loss_fn"] = loss_fn
    return init_args
