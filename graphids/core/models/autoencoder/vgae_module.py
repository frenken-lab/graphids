from __future__ import annotations

import torch
import torch.nn as nn

from graphids.config.constants import ModelType

from ..base import GraphModuleBase, binary_test_metrics
from .vgae import GraphAutoencoderNeighborhood

# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------

NUM_ROUNDS = 7  # round-robin masking groups at test time — every node masked exactly once


class VGAEModule(GraphModuleBase):
    """VGAE training: mask-and-reconstruct + KL.

    Loss selection is decoupled from this module: ``loss_fn`` is an
    ``nn.Module`` built by :func:`graphids.core.losses.build.build_loss` from
    the config's ``loss_config`` / ``distillation_config`` blocks and
    injected here. The base loss is
    :class:`~graphids.core.losses.autoencoder.VGAETaskLoss`; when KD is
    active it's wrapped in
    :class:`~graphids.core.losses.distillation.FeatureDistillation`.

    Training applies 15% random node masking before forward; validation
    uses the same regime (single masked forward) for an aligned signal;
    test scoring runs deterministic 7-round round-robin masking + a
    single unmasked forward for ``mu`` (Mahalanobis) and KL.

    Anomaly score = max-σ over three components (recon under round-robin
    masking, Mahalanobis on ``mu``, KL). Calibration buffers are filled
    by :meth:`fit_score_norm` at test-start (mirrors OCGIN's
    ``calibrate_svdd_center`` lifecycle, see ``trainer.py``).
    """

    def _register_score_norm_buffers(self) -> None:
        """Score-norm calibration state — populated by ``fit_score_norm``.

        Per-component scalar mean/std for z-normalized scoring, plus a
        per-dim ``mu_mean`` / ``mu_std`` for Mahalanobis on the latent.
        Serialized via state_dict so a calibrated ckpt is reproducible
        without re-running the fit.
        """
        for name in ("recon", "mahal", "kl"):
            self.register_buffer(f"score_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"score_{name}_std", torch.tensor(1.0))
        latent_dim = int(self.hparams.latent_dim)
        self.register_buffer("mu_mean", torch.zeros(latent_dim))
        self.register_buffer("mu_std", torch.ones(latent_dim))
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
        dropout: float = 0.1,
        edge_dim: int = 11,
        proj_dim: int = 0,
        id_encoder_class_path: str = "graphids.core.models.id_encoding.LookupIdEncoder",
        id_encoder_kwargs: dict | None = None,
        # --- training ---
        lr: float = 0.003,
        weight_decay: float = 0.0001,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        mask_rate: float = 0.15,
        # --- anomaly scoring (z-normed; weights apply only to the legacy
        # fallback that errors out if calibration hasn't run). Kept for
        # config-schema stability; values are unused under the calibrated
        # path which uses max-σ over the three components.
        score_recon_weight: float = 1.0,
        score_mahal_weight: float = 1.0,
        score_kl_weight: float = 1.0,
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

        from .._conv import resolve_edge_dim

        hp = self.hparams
        encoder_cls = import_class(hp.id_encoder_class_path)
        encoder_kwargs = {"embedding_dim": hp.embedding_dim, **(hp.id_encoder_kwargs or {})}
        # +1 vocab slot for the reserved mask_id (= num_ids); see
        # GraphAutoencoderNeighborhood docstring.
        id_encoder = encoder_cls.from_vocab_size(num_ids=hp.num_ids + 1, **encoder_kwargs)
        self.model = GraphAutoencoderNeighborhood(
            id_encoder=id_encoder,
            num_ids=hp.num_ids,
            in_channels=hp.in_channels,
            hidden_dims=list(hp.hidden_dims),
            latent_dim=hp.latent_dim,
            encoder_heads=hp.heads,
            dropout=hp.dropout,
            conv_type=hp.conv_type,
            edge_dim=resolve_edge_dim(hp.conv_type, hp.edge_dim),
            proj_dim=hp.proj_dim,
            use_checkpointing=hp.gradient_checkpointing,
        )
        if hp.compile_model:
            from ..base import try_compile

            self.model = try_compile(self.model, conv_type=hp.conv_type, dynamic=True)

    def _task_loss_module(self) -> nn.Module:
        """Return the base VGAETaskLoss, unwrapping FeatureDistillation if present."""
        return getattr(self.loss_fn, "base_loss", self.loss_fn)

    def _training_step_inner(self, batch, _idx):
        edge_attr = getattr(batch, "edge_attr", None)
        x_m, nid_m, mask = self.model.apply_random_mask(
            batch.x, batch.node_id, mask_rate=self.hparams.mask_rate
        )
        outputs = self.model(
            x_m,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=nid_m,
        )
        loss = self.loss_fn(outputs, batch)
        bs = batch.num_graphs
        self.log("train_loss", loss, batch_size=bs)

        task_loss = self._task_loss_module()
        if task_loss.last_recon is not None:
            self.log("train_recon", task_loss.last_recon, batch_size=bs)
            self.log("train_kl", task_loss.last_kl, batch_size=bs)

        # Sanity: masked-subset recon must exceed unmasked-subset recon —
        # if they converge the encoder is echoing v back via some path
        # other than the masked feature (e.g. node_id leakage).
        cont, _z, _kl = outputs
        node_mse = (cont - batch.x).pow(2).mean(-1)
        if mask.any():
            self.log("train_recon_masked", node_mse[mask].mean(), batch_size=bs)
        if (~mask).any():
            self.log("train_recon_unmasked", node_mse[~mask].mean(), batch_size=bs)

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
        """Single random-masked forward; per-class recon split.

        Validation matches training regime (random masking, single forward)
        rather than the test-time round-robin path — round-robin would 7×
        the per-epoch wall while only marginally tightening the validation
        signal that drives ``val_discrimination_ratio``. KL is omitted from
        ``val_loss`` (regularization, not held-out fit; the per-graph
        signal is in the recon-class components).
        """
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        x_m, nid_m, _mask = self.model.apply_random_mask(
            batch.x, batch.node_id, mask_rate=self.hparams.mask_rate
        )
        cont, _z, _kl = self.model(
            x_m,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=nid_m,
        )
        recon = scatter((cont - batch.x).pow(2).mean(dim=1), batch.batch, dim=0, reduce="mean")

        bs = batch.num_graphs
        self.log("val_loss", recon.mean(), batch_size=bs)

        y = batch.y.view(-1)
        sub: dict[str, torch.Tensor] = {}
        for label, m in (("benign", y == 0), ("attack", y != 0)):
            n = int(m.sum())
            if not n:
                continue
            v = recon[m].mean()
            self.log(f"val_loss_{label}", v, batch_size=n)
            sub[label] = v
        # gap shrinks monotonically as both losses converge → useless under
        # mode='max'. ratio grows as discrimination strengthens → the right
        # monitor. Gap kept as diagnostic for absolute-error convergence.
        if "benign" in sub and "attack" in sub:
            self.log("val_discrimination_gap", sub["attack"] - sub["benign"], batch_size=bs)
            self.log(
                "val_discrimination_ratio",
                sub["attack"] / (sub["benign"] + 1e-6),
                batch_size=bs,
            )

    def _score_via_round_robin(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-graph (recon, mahal, kl) plus per-node ``z``.

        Recon uses 7-round round-robin masking — every node is masked
        exactly once and contributes its "predict-from-neighbors" error.
        Mahalanobis and KL come from a single unmasked forward; using
        unmasked ``mu`` matches the train-time encoder regime for the
        latent-density axis (the masked encode would shift mu's
        distribution and break the calibration).
        """
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        ea = edge_attr if self.model._uses_edge_attr else None

        # Unmasked encode → mu, kl
        _z, kl_per_node, mu = self.model.encode(
            batch.x,
            batch.edge_index,
            edge_attr=ea,
            batch=batch.batch,
            node_id=batch.node_id,
        )
        # Mahalanobis with eps floor on mu_std (KL pulls some latent dims
        # toward zero variance; without the floor those dims dominate
        # with massive distances).
        mahal_per_node = ((mu - self.mu_mean) / self.mu_std.clamp(min=1e-3)).pow(2).sum(-1)

        # Round-robin masked recon — 7 forwards, every node masked once.
        n = batch.x.size(0)
        recon_per_node = torch.zeros(n, device=batch.x.device, dtype=batch.x.dtype)
        # Re-encode for each masked round; reuse the unmasked z for the
        # caller's z-statistics path (extract_features) — it's the
        # benign-regime latent the fusion stage trained on historically.
        for r in range(NUM_ROUNDS):
            x_m, nid_m, mask = self.model.apply_round_robin_mask(
                batch.x, batch.node_id, r, num_rounds=NUM_ROUNDS
            )
            cont, _, _ = self.model(
                x_m,
                batch.edge_index,
                batch.batch,
                edge_attr=edge_attr,
                node_id=nid_m,
            )
            recon_per_node[mask] = (cont[mask] - batch.x[mask]).pow(2).mean(-1)

        recon = scatter(recon_per_node, batch.batch, dim=0, reduce="mean")
        mahal = scatter(mahal_per_node, batch.batch, dim=0, reduce="mean")
        kl = scatter(kl_per_node, batch.batch, dim=0, reduce="mean")
        return recon, mahal, kl, _z

    def extract_features(self, batch, device: torch.device) -> torch.Tensor:
        """8-D fusion features: ``[recon, mahal, kl, z_mean, z_std, z_max, z_min, conf]``.

        Same shape as the pre-mask-recon code path; columns swap from
        ``[recon, nbr, canid, ...]`` to ``[recon, mahal, kl, ...]``. Fusion
        cache version is bumped (see ``core/data/fusion_states.py``) so
        previously-cached fusion-state files are regenerated on next
        access. Mahalanobis here is uncalibrated unless ``fit_score_norm``
        has run on this module instance — the raw squared distance is
        still a usable feature for the fusion learner.
        """
        from torch_geometric.utils import scatter

        recon, mahal, kl, z = self._score_via_round_robin(batch)
        b = batch.batch
        z_mean = scatter(z.mean(1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(1).values, b, dim=0, reduce="min")
        conf = 1.0 / (1.0 + recon)
        return torch.stack([recon, mahal, kl, z_mean, z_std, z_max, z_min, conf], dim=1)

    def _per_graph_errors(self, batch):
        """Per-graph anomaly score — max-σ over (recon, mahal, kl) z-normed components.

        Requires :meth:`fit_score_norm` to have populated calibration
        buffers. There is no fixed-weight fallback: an old ckpt loaded
        without mask parameters will fail at this point with a clear
        message rather than silently producing nonsensical scores.
        """
        if not bool(self.score_norm_fitted):
            raise RuntimeError(
                "VGAE scoring requires fit_score_norm() to have run. "
                "If loading an old ckpt without mask_token, retrain under "
                "the mask-recon code or use the legacy scoring path "
                "from before commit 2."
            )
        recon, mahal, kl, _z = self._score_via_round_robin(batch)
        eps = 1e-6
        z_recon = (recon - self.score_recon_mean) / (self.score_recon_std + eps)
        z_mahal = (mahal - self.score_mahal_mean) / (self.score_mahal_std + eps)
        z_kl = (kl - self.score_kl_mean) / (self.score_kl_std + eps)
        return torch.stack([z_recon, z_mahal, z_kl], dim=0).amax(dim=0)

    @torch.no_grad()
    def fit_score_norm(self, val_loader, device: torch.device) -> None:
        """Two-pass calibration on benign val: (1) mu_mean/std for Mahalanobis,
        (2) per-component score mean/std under the test scoring path.

        Only label==0 (benign) val rows are used — attacks are OOD by
        construction. Pass 1 must complete before pass 2 because
        ``_score_via_round_robin`` reads ``mu_mean``/``mu_std`` to
        compute Mahalanobis.
        """
        from torch_geometric.data import Batch as PyGBatch

        was_training = self.training
        self.eval()

        def _benign_subbatch(batch):
            y = batch.y.view(-1)
            benign_idx = (y == 0).nonzero(as_tuple=False).flatten().tolist()
            if not benign_idx:
                return None
            return PyGBatch.from_data_list([batch.to_data_list()[i] for i in benign_idx])

        # Pass 1: per-node mu over all benign val nodes
        mus: list[torch.Tensor] = []
        for batch in val_loader:
            batch = batch.clone().to(device)
            sub = _benign_subbatch(batch)
            if sub is None:
                continue
            ea = getattr(sub, "edge_attr", None) if self.model._uses_edge_attr else None
            _z, _kl, mu = self.model.encode(
                sub.x,
                sub.edge_index,
                edge_attr=ea,
                batch=sub.batch,
                node_id=sub.node_id,
            )
            mus.append(mu.cpu())
        if not mus:
            raise RuntimeError("fit_score_norm: no benign rows in val loader")
        mu_all = torch.cat(mus, dim=0)
        self.mu_mean.copy_(mu_all.mean(dim=0).to(self.mu_mean.device))
        self.mu_std.copy_(mu_all.std(dim=0).clamp(min=1e-3).to(self.mu_std.device))

        # Pass 2: per-graph (recon, mahal, kl) under the test scoring path
        all_recon, all_mahal, all_kl = [], [], []
        for batch in val_loader:
            batch = batch.clone().to(device)
            sub = _benign_subbatch(batch)
            if sub is None:
                continue
            recon, mahal, kl, _z = self._score_via_round_robin(sub)
            all_recon.append(recon.cpu())
            all_mahal.append(mahal.cpu())
            all_kl.append(kl.cpu())

        if was_training:
            self.train()

        r = torch.cat(all_recon)
        m = torch.cat(all_mahal)
        k = torch.cat(all_kl)
        if len(r) < 100:
            raise RuntimeError(f"fit_score_norm: need >=100 benign val graphs, got {len(r)}")
        self.score_recon_mean.copy_(r.mean())
        self.score_recon_std.copy_(r.std())
        self.score_mahal_mean.copy_(m.mean())
        self.score_mahal_std.copy_(m.std())
        self.score_kl_mean.copy_(k.mean())
        self.score_kl_std.copy_(k.std())
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
