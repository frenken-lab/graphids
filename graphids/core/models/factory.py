"""Build a model from ``(model_type, scale)`` for ad-hoc callers.

Used by ``probe-budget`` (and any other code that needs a fresh model
instance without going through the full orchestrate stack). Reads
``configs/models/_expand.jsonnet`` to get the model's class_path +
default init_args, injects the runtime-only shape params
(``num_ids`` / ``in_channels`` / ``conv_type``), builds the loss, and
returns a wired ``nn.Module``.
"""

from __future__ import annotations

import torch.nn as nn

from graphids._reflect import filter_kwargs, import_class
from graphids.config.constants import CONFIG_DIR, FAMILY_FOR_MODEL_TYPE
from graphids.config.jsonnet import render
from graphids.core.losses.build import build_loss


def build_model_from_spec(
    model_type: str,
    scale: str,
    *,
    num_ids: int,
    in_channels: int,
    conv_type: str | None = None,
) -> nn.Module:
    """Render model jsonnet -> inject runtime params -> build."""
    family = FAMILY_FOR_MODEL_TYPE[model_type]
    model_cfg = render(
        CONFIG_DIR / "models" / "_expand.jsonnet",
        tla={"family": family, "model_type": model_type, "scale": scale},
    )
    init_args = dict(model_cfg["model"].get("init_args", {}))
    init_args["num_ids"] = num_ids
    init_args["in_channels"] = in_channels
    if conv_type is not None:
        init_args["conv_type"] = conv_type
    loss_fn = build_loss(model_type, init_args.pop("loss_config", None), distillation_config=None)
    if loss_fn is not None:
        init_args["loss_fn"] = loss_fn

    klass = import_class(model_cfg["model"]["class_path"])
    return klass(**filter_kwargs(klass, init_args))
