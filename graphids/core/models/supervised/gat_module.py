from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from graphids.config.constants import (
    ModelType,  # noqa: F401 (used in __init__ annotation)
)

from ..base import GraphModuleBase, binary_test_metrics
from .gat import GATWithJK

# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class GATModule(GraphModuleBase):
    """GAT supervised classification (normal vs attack).

    Loss selection is decoupled from this module: ``loss_fn`` is an
    ``nn.Module`` built by :func:`graphids.instantiate._build_loss` from
    the config's ``loss_config`` / ``distillation_config`` blocks and
    injected here. When it's a
    :class:`~graphids.core.losses.distillation.SoftLabelDistillation`,
    training automatically becomes a KD run — no branching here, no
    teacher attribute on ``self``, no base-class plumbing.
    """

    def __init__(
        self,
        *,
        loss_fn: nn.Module,
        # --- architecture ---
        hidden: int = 48,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.2,
        fc_layers: int = 3,
        embedding_dim: int = 16,
        conv_type: str = "gatv2",
        edge_dim: int = 11,
        pool_aggrs: list[str] | None = None,
        proj_dim: int = 0,
        # --- training ---
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: ModelType = "gat",
        lake_root: str | None = None,
        dataset: str = "",
        seed: int = 42,
        gat_stage: str = "supervised",
        variational: bool = True,  # upstream VGAE type — identity key for supervised
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        if lake_root is None:
            from graphids.config.settings import get_settings

            lake_root = get_settings().lake_root
        super().__init__()
        if pool_aggrs is None:
            pool_aggrs = ["mean"]
        # ``loss_fn`` is an nn.Module and shouldn't be serialized into the
        # hyperparameter dict — save_hyperparameters ignores it explicitly.
        self.save_hyperparameters(ignore=["loss_fn"])
        self.loss_fn = loss_fn
        self.model = None
        self.test_metrics = binary_test_metrics()
        if num_ids > 0:
            self._build()

    def _build(self):
        hp = self.hparams
        self.model = GATWithJK.from_config(hp, hp.num_ids, hp.in_channels)
        if hp.compile_model:
            from ..base import try_compile

            self.model = try_compile(self.model, conv_type=hp.conv_type, dynamic=True)

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        loss = self.loss_fn(logits, batch.y, graph=batch)
        acc = (logits.argmax(1) == batch.y).float().mean()
        return loss, acc

    def _training_step_inner(self, batch, _idx):
        loss, acc = self._step(batch)
        bs = batch.num_graphs
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        self.log("train_acc", acc, prog_bar=True, batch_size=bs)
        # Log KD components separately when distillation is active. The
        # SoftLabelDistillation wrapper stores the two detached scalars on
        # itself after each forward() call, so no extra passes are needed.
        from graphids.core.losses.distillation import SoftLabelDistillation

        if isinstance(self.loss_fn, SoftLabelDistillation):
            if self.loss_fn.last_hard_loss is not None:
                self.log("train_hard_loss", self.loss_fn.last_hard_loss, batch_size=bs)
            if self.loss_fn.last_soft_loss is not None:
                self.log("train_soft_loss", self.loss_fn.last_soft_loss, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_acc", acc, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, _idx, dataloader_idx=0):
        logits = self(batch)
        scores = F.softmax(logits, dim=1)[:, 1]
        self.test_metrics.update(scores, batch.y)

    def predict_step(self, batch, _idx):
        logits = self(batch)
        scores = F.softmax(logits, dim=1)[:, 1]
        return {"preds": logits.argmax(1), "scores": scores, "labels": batch.y}
