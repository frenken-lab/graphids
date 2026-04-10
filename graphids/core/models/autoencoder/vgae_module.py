from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids.config.constants import (
    ModelType,  # noqa: F401 (used in __init__ annotation)
)

from ..base import GraphModuleBase, binary_test_metrics
from .vgae import GraphAutoencoderNeighborhood

# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------


class VGAEModule(GraphModuleBase):
    """VGAE training: reconstruct node features + CAN IDs + neighborhood.

    Loss selection is decoupled from this module: ``loss_fn`` is an
    ``nn.Module`` built by :func:`graphids.instantiate._build_loss` from
    the config's ``loss_config`` / ``distillation_config`` blocks and
    injected here. The default base loss is
    :class:`~graphids.core.losses.autoencoder.VGAETaskLoss`; when KD is
    active it's wrapped in
    :class:`~graphids.core.losses.distillation.FeatureDistillation`.

    The loss module owns ``canid_weight`` / ``nbr_weight`` / ``kl_weight``
    / ``k_neg`` / ``num_ids``. Because ``_per_graph_errors`` (used at test
    time) needs the same weights to score anomalies consistently with
    training, it reads them back off ``self.loss_fn`` via the
    :meth:`_task_loss_module` helper which unwraps KD if present.
    """

    def __init__(
        self,
        *,
        loss_fn: nn.Module,
        # --- architecture ---
        conv_type: str = "gatv2",
        hidden_dims: list[int] | None = None,
        latent_dim: int = 48,
        heads: int = 4,
        embedding_dim: int = 32,
        dropout: float = 0.15,
        edge_dim: int = 11,
        proj_dim: int = 0,
        variational: bool = True,
        mask_ratio: float = 0.3,
        # --- training ---
        lr: float = 0.003,
        weight_decay: float = 0.0001,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: ModelType = "vgae",
        lake_root: str | None = None,
        dataset: str = "",
        seed: int = 42,
        gat_stage: str = "supervised",
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        if lake_root is None:
            from graphids.config.settings import get_settings

            lake_root = get_settings().lake_root
        super().__init__()
        self.hparams = self._capture_hparams(locals(), ignore=("loss_fn",))
        self.loss_fn = loss_fn
        self._init_threshold_metrics()
        self.model = None
        self.test_metrics = binary_test_metrics()
        if num_ids > 0:
            self._build()

    def _build(self):
        hp = self.hparams
        self.model = GraphAutoencoderNeighborhood.from_config(hp, hp.num_ids, hp.in_channels)
        if hp.compile_model:
            from ..base import try_compile

            self.model = try_compile(self.model, conv_type=hp.conv_type, dynamic=True)
        # The loss module was constructed in ``instantiate._build_loss``
        # before ``setup`` ran, so ``num_ids`` on VGAETaskLoss was 0.
        # Propagate the real value now that the datamodule has populated it.
        task_loss = self._task_loss_module()
        if hasattr(task_loss, "num_ids"):
            task_loss.num_ids = hp.num_ids

    def _task_loss_module(self) -> nn.Module:
        """Return the base VGAETaskLoss, unwrapping FeatureDistillation if present."""
        return getattr(self.loss_fn, "base_loss", self.loss_fn)

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        mask_ratio = self.hparams.mask_ratio if self.training else 0.0
        return self.model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            mask_ratio=mask_ratio,
            node_id=batch.node_id,
        )

    def _step(self, batch):
        outputs = self(batch)
        return self.loss_fn(outputs, batch)

    def extract_features(self, batch, device: torch.device) -> torch.Tensor:
        """8-D fusion features: [recon_err, nbr_err, canid_err, z_mean, z_std, z_max, z_min, confidence]."""
        import torch.nn.functional as F
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None) if getattr(self.model, "_uses_edge_attr", False) else None
        cont, canid_logits, nbr_logits, z, _, _ = self.model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=batch.node_id,
        )
        b = batch.batch
        recon_err = scatter((cont - batch.x).pow(2).mean(1), b, dim=0, reduce="mean")
        canid_err = scatter(F.cross_entropy(canid_logits, batch.node_id, reduction="none"), b, dim=0, reduce="mean")
        nbr_targets = self.model.create_neighborhood_targets(batch.node_id, batch.edge_index, b)
        nbr_err = scatter(F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets, reduction="none").mean(1), b, dim=0, reduce="mean")
        z_mean = scatter(z.mean(1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(1).values, b, dim=0, reduce="min")
        conf = 1.0 / (1.0 + recon_err)
        return torch.stack([recon_err, nbr_err, canid_err, z_mean, z_std, z_max, z_min, conf], dim=1)

    def _training_step_inner(self, batch, _idx):
        loss = self._step(batch)
        bs = batch.num_graphs
        self.log("train_loss", loss, batch_size=bs)
        # Log KD components separately when FeatureDistillation is active.
        from graphids.core.losses.distillation import FeatureDistillation

        if isinstance(self.loss_fn, FeatureDistillation):
            if self.loss_fn.last_task_loss is not None:
                self.log("train_task_loss", self.loss_fn.last_task_loss, batch_size=bs)
            if self.loss_fn.last_kd_loss is not None:
                self.log("train_kd_loss", self.loss_fn.last_kd_loss, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("val_loss", loss, batch_size=batch.num_graphs)

    def _per_graph_errors(self, batch):
        """Compute weighted per-graph anomaly errors from a batch.

        Reads the recon / canid / nbr weighting off ``self.loss_fn`` (via
        ``_task_loss_module``) so training and test scoring share the same
        weighting by construction.
        """
        from torch_geometric.utils import scatter

        task = self._task_loss_module()
        edge_attr = getattr(batch, "edge_attr", None)
        cont, canid_logits, nbr_logits, _, _, _ = self.model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )
        per_node_se = (cont - batch.x).pow(2).mean(dim=1)
        recon = scatter(per_node_se, batch.batch, dim=0, reduce="max")
        canid_err = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
        canid_per_graph = scatter(canid_err, batch.batch, dim=0, reduce="max")
        nbr_targets = self.model.create_neighborhood_targets(
            batch.node_id, batch.edge_index, batch.batch
        )
        nbr_err = F.binary_cross_entropy_with_logits(
            nbr_logits, nbr_targets, reduction="none"
        ).mean(dim=1)
        nbr_per_graph = scatter(nbr_err, batch.batch, dim=0, reduce="max")
        return recon + task.canid_weight * canid_per_graph + task.nbr_weight * nbr_per_graph

    def test_step(self, batch, _idx, dataloader_idx=0):
        errors = self._per_graph_errors(batch)
        self.roc_metric.update(errors.detach(), batch.y.detach())

    def on_test_epoch_end(self):
        self._log_thresholded_metrics()

    def predict_step(self, batch, _idx):
        errors = self._per_graph_errors(batch)
        return {"errors": errors, "labels": batch.y}

    def build_optimizers(self, max_epochs: int):
        params = list(self.model.parameters())
        # FeatureDistillation's optional projection layer — if KD is active
        # and the student/teacher latent dims differ, its weights need to
        # be optimized too.
        if hasattr(self.loss_fn, "projection") and self.loss_fn.projection is not None:
            params += list(self.loss_fn.projection.parameters())
        opt = torch.optim.Adam(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
        return opt, scheduler
