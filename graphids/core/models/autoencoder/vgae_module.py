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
    ``nn.Module`` built by :func:`graphids.core.losses.build.build_loss` from
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

    # Calibration state for benign-val z-normalization (Tier B). Six scalar
    # statistics + a fitted flag; serialized via state_dict so a ckpt's
    # scoring head is reproducible without re-running calibration. Filled
    # by fit_score_norm() at test-start (mirrors OCGIN's calibrate_svdd_center
    # pattern at trainer.test, see trainer.py:262-265 for the rationale —
    # callback-based calibration deadlocked on ckpt-save ordering).
    def _register_score_norm_buffers(self) -> None:
        for name in ("recon", "canid", "nbr"):
            self.register_buffer(f"score_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"score_{name}_std", torch.tensor(1.0))
        self.register_buffer("score_norm_fitted", torch.tensor(False))

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
        id_encoder_class_path: str = "graphids.core.models.id_encoding.LookupIdEncoder",
        id_encoder_kwargs: dict | None = None,
        # --- training ---
        lr: float = 0.003,
        weight_decay: float = 0.0001,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        # --- anomaly scoring (decoupled from training-loss weights) ---
        # Frenken et al. 2025 §8.2 composite_error = α·E_node + β·E_neighbor + γ·E_CANID
        # with α=1.0, β=20.0, γ=0.3 — chosen empirically for discrimination, NOT
        # the same as the training-loss component weights (canid_weight=0.1,
        # nbr_weight=0.05) which were chosen for ELBO gradient balance.
        score_recon_weight: float = 1.0,
        score_canid_weight: float = 0.3,
        score_nbr_weight: float = 20.0,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: ModelType = "vgae",
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        super().__init__()
        self._store_init_kwargs(locals())
        self.id_encoder_kwargs = self.id_encoder_kwargs or {}
        self._init_threshold_metrics()
        self.model = None
        self.test_metrics = binary_test_metrics()
        self._register_score_norm_buffers()
        if num_ids > 0:
            self._build()

    def _build(self):
        from graphids._reflect import import_class

        hp = self.hparams
        encoder_cls = import_class(hp.id_encoder_class_path)
        encoder_kwargs = {"embedding_dim": hp.embedding_dim, **(hp.id_encoder_kwargs or {})}
        id_encoder = encoder_cls.from_vocab_size(num_ids=hp.num_ids, **encoder_kwargs)
        self.model = GraphAutoencoderNeighborhood.from_config(
            hp, id_encoder, hp.num_ids, hp.in_channels
        )
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
        return self.model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )

    def _step(self, batch):
        outputs = self(batch)
        return self.loss_fn(outputs, batch)

    def extract_features(self, batch, device: torch.device) -> torch.Tensor:
        """8-D fusion features: [recon, nbr, canid, z_mean, z_std, z_max, z_min, confidence].

        Single forward via ``_per_component_errors`` — the three component
        errors and ``z`` come from the same pass; the four z-statistics and
        the confidence scalar are derived locally. Output column order is
        preserved for backward-compat with cached fusion-state files.
        """
        from torch_geometric.utils import scatter

        recon, canid, nbr, z = self._per_component_errors(batch)
        b = batch.batch
        z_mean = scatter(z.mean(1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(1).values, b, dim=0, reduce="min")
        conf = 1.0 / (1.0 + recon)
        return torch.stack([recon, nbr, canid, z_mean, z_std, z_max, z_min, conf], dim=1)

    def _training_step_inner(self, batch, _idx):
        loss = self._step(batch)
        bs = batch.num_graphs
        self.log("train_loss", loss, batch_size=bs)
        # Per-component VGAETaskLoss telemetry — recon-dominance after the
        # sigmoid + masking deletions is the diagnostic to watch.
        task_loss = self._task_loss_module()
        if task_loss.last_recon is not None:
            self.log("train_recon", task_loss.last_recon, batch_size=bs)
            self.log("train_canid", task_loss.last_canid, batch_size=bs)
            self.log("train_nbr", task_loss.last_nbr, batch_size=bs)
            self.log("train_kl", task_loss.last_kl, batch_size=bs)
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
        """Single forward + label-mask aggregation.

        Old design ran ``_step(batch)`` three times (full / benign-subset /
        attack-subset) to log the per-class reconstruction split — three
        forward passes for the same diagnostic information. The forward
        already computes predictions for every node regardless of label;
        the subset distinction is purely an aggregation choice on per-graph
        errors. Replaced with one forward via ``_per_component_errors`` and
        ``tensor[mask].mean()`` for the per-class numbers.

        ``val_loss`` here drops the KL term that training loss includes.
        KL is regularization (encoder spread, not held-out fit) and isn't
        cleanly per-graph; per-class signal lives in the recon-class
        components. Old val_loss numbers won't be directly comparable.
        """
        recon, canid, nbr, _ = self._per_component_errors(batch)
        tl = self._task_loss_module()
        per_graph = recon + tl.canid_weight * canid + tl.nbr_weight * nbr
        bs = batch.num_graphs
        self.log("val_loss", per_graph.mean(), batch_size=bs)

        y = batch.y.view(-1)
        sub: dict[str, torch.Tensor] = {}
        for label, mask in (("benign", y == 0), ("attack", y != 0)):
            n = int(mask.sum())
            if not n:
                continue
            v = per_graph[mask].mean()
            self.log(f"val_loss_{label}", v, batch_size=n)
            sub[label] = v
        # gap shrinks monotonically as both losses converge → useless under
        # mode='max'. ratio grows as discrimination strengthens → the right
        # monitor. Gap kept as diagnostic for absolute-error convergence.
        if "benign" in sub and "attack" in sub:
            self.log("val_discrimination_gap", sub["attack"] - sub["benign"], batch_size=bs)
            self.log(
                "val_discrimination_ratio", sub["attack"] / (sub["benign"] + 1e-6), batch_size=bs
            )

    def _per_component_errors(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-graph (recon, canid, nbr) errors plus the per-node latent ``z``.

        ``z`` is returned (not discarded) so ``extract_features`` can compute
        its z-statistics without a second forward pass. Callers that don't
        need it can unpack with ``_``. No weighting / no normalization.

        Per-component errors are MEAN-pooled per graph to align with the
        training objective (``F.mse_loss`` defaults to mean reduction).
        MAX-pooling produced score inversions when benign graphs had outlier
        nodes.
        """
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        cont, canid_logits, nbr_logits, z, _ = self.model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )
        recon = scatter((cont - batch.x).pow(2).mean(dim=1), batch.batch, dim=0, reduce="mean")
        canid = scatter(
            F.cross_entropy(canid_logits, batch.node_id, reduction="none"),
            batch.batch,
            dim=0,
            reduce="mean",
        )
        nbr_targets = self.model.create_neighborhood_targets(
            batch.node_id, batch.edge_index, batch.batch
        )
        nbr = scatter(
            F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets, reduction="none").mean(
                dim=1
            ),
            batch.batch,
            dim=0,
            reduce="mean",
        )
        return recon, canid, nbr, z

    def _per_graph_errors(self, batch):
        """Per-graph anomaly score. Two scoring paths:

        - **Z-norm (Tier B):** when ``fit_score_norm`` has populated the
          benign-val mean/std buffers (``score_norm_fitted=True``), score
          each component in σ-units against benign val and aggregate by
          max — a single component spiking is the attack signal.
        - **Fixed weights (legacy):** ``α·recon + γ·canid + β·nbr`` with
          dataset-tuned coefficients. Used when calibration hasn't run
          (e.g. older ckpts without the buffers) so old runs still score.
        """
        recon, canid, nbr, _ = self._per_component_errors(batch)
        if bool(self.score_norm_fitted):
            eps = 1e-6
            z_recon = (recon - self.score_recon_mean) / (self.score_recon_std + eps)
            z_canid = (canid - self.score_canid_mean) / (self.score_canid_std + eps)
            z_nbr = (nbr - self.score_nbr_mean) / (self.score_nbr_std + eps)
            return torch.stack([z_recon, z_canid, z_nbr], dim=0).amax(dim=0)
        return (
            self.score_recon_weight * recon
            + self.score_canid_weight * canid
            + self.score_nbr_weight * nbr
        )

    @torch.no_grad()
    def fit_score_norm(self, val_loader, device: torch.device) -> None:
        """Compute benign-val per-component mean/std and write buffers.

        Filters each batch's val rows to label==0 (benign) before computing
        component errors, so the calibration is against the in-distribution
        side only — attacks are OOD by construction. Mirrors OCGIN's
        ``calibrate_svdd_center`` lifecycle: fit once at test-start.
        """
        from torch_geometric.data import Batch as PyGBatch

        was_training = self.training
        self.eval()
        all_recon, all_canid, all_nbr = [], [], []
        for batch in val_loader:
            batch = batch.clone().to(device)
            y = batch.y.view(-1)
            benign_idx = (y == 0).nonzero(as_tuple=False).flatten().tolist()
            if not benign_idx:
                continue
            sub = PyGBatch.from_data_list([batch.to_data_list()[i] for i in benign_idx])
            r, c, n, _ = self._per_component_errors(sub)
            all_recon.append(r.cpu())
            all_canid.append(c.cpu())
            all_nbr.append(n.cpu())
        if was_training:
            self.train()
        if not all_recon:
            raise RuntimeError("fit_score_norm: no benign rows in val loader")
        r = torch.cat(all_recon)
        c = torch.cat(all_canid)
        n = torch.cat(all_nbr)
        if len(r) < 100:
            raise RuntimeError(f"fit_score_norm: need >=100 benign val graphs, got {len(r)}")
        self.score_recon_mean.copy_(r.mean())
        self.score_recon_std.copy_(r.std())
        self.score_canid_mean.copy_(c.mean())
        self.score_canid_std.copy_(c.std())
        self.score_nbr_mean.copy_(n.mean())
        self.score_nbr_std.copy_(n.std())
        self.score_norm_fitted.fill_(True)

    def test_step(self, batch, _idx, dataloader_idx=0):
        errors = self._per_graph_errors(batch)
        self.roc_metric.update(errors.detach(), batch.y.detach())
        self._record_test_batch(dataloader_idx, scores=errors, labels=batch.y)

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
        return opt, None
