"""Variational graph autoencoder — collapsed arch + trainer-bridge.

The single :class:`VGAE` class is both the architecture (encoder /
decoder / aux heads / mask token / score-norm calibration buffers) and
the trainer-bridge (``training_step``/``validation_step``/``test_step``,
score primitives, fusion-feature extractor). No wrapper module — see
``~/plans/graphids-collapse-model-modules.md`` Phase 1.

Encoder maps node features to ``q(z|x) = N(mu, σ²)``; decoder
reconstructs continuous features from the reparameterized ``z``.
Mask-recon training (15% random node masking) commits the encoder to
"predict v from neighborhood" rather than "echo v back".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from graphids.paths import ModelType

from .._conv import (
    InputEncoder,
    build_conv_stack,
    build_encoder_stack,
    conv_forward,
)
from ..base import ScoreBasedDetectorMixin
from .._score_primitives import rayleigh_quotient, tam_affinity
from ..id_encoding import IdEncodingCfg


class VGAE(ScoreBasedDetectorMixin):
    """Collapsed VGAE — arch + trainer-bridge in one ``nn.Module``.

    Loss selection is decoupled: ``loss_fn`` is an ``nn.Module`` built
    from experiment config.

    Anomaly score = max-σ over four components (masked recon mean,
    masked recon max, TAM affinity, Rayleigh quotient). Calibration
    buffers are filled by :meth:`on_test_setup` at test-start.
    """

    _SCALES: dict[str, dict] = {
        "small": {"latent_dim": 64, "hidden_dims": [64]},
        "large": {"latent_dim": 128, "hidden_dims": [128]},
    }

    def __init__(
        self,
        *,
        loss_fn: nn.Module | None = None,
        # --- architecture (latent_dim/hidden_dims=None → resolve from scale) ---
        conv_type: str = "gatv2",
        hidden_dims: list[int] | None = None,
        latent_dim: int | None = None,
        heads: int = 4,
        embedding_dim: int = 32,
        dropout: float = 0.1,
        edge_dim: int = 11,
        proj_dim: int = 0,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        batch_norm: bool = True,
        mlp_hidden: int | None = None,
        id_encoder_cfg: IdEncodingCfg | None = None,
        id_encoder_class_path: str = "graphids.core.models.id_encoding.LookupIdEncoder",
        id_encoder_kwargs: dict | None = None,
        # --- training ---
        lr: float = 0.003,
        weight_decay: float = 0.0001,
        mask_rate: float = 0.15,
        # --- anomaly scoring (config-schema stability; calibrated max-σ
        # path doesn't read these). ---
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
        s = self._SCALES.get(scale, {})
        if latent_dim is None:
            latent_dim = s.get("latent_dim", 48)
        if hidden_dims is None:
            hidden_dims = s.get("hidden_dims")
        super().__init__()
        self._register_score_norm_buffers(latent_dim)
        self._init_post(locals())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    _SCORE_COMPONENTS: tuple[str, ...] = ("recon", "recon_max", "affinity", "rq")

    def _register_score_norm_buffers(self, latent_dim: int) -> None:
        for name in self._SCORE_COMPONENTS:
            self.register_buffer(f"score_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"score_{name}_std", torch.tensor(1.0))
        self.register_buffer("mu_mean", torch.zeros(latent_dim))
        self.register_buffer("mu_std", torch.ones(latent_dim))
        self.register_buffer("score_norm_fitted", torch.tensor(False))

    def _build(self):
        hp = self.hparams
        # +1 vocab slot for the reserved mask_id (= num_ids).
        id_encoder = self._build_id_encoder(num_ids_offset=1)
        edge_dim = hp.edge_dim if hp.conv_type in {"transformer", "gatv2", "gps"} else None

        self.input_encoder = InputEncoder(
            id_encoder=id_encoder,
            in_channels=hp.in_channels,
            conv_type=hp.conv_type,
            edge_dim=edge_dim,
            proj_dim=hp.proj_dim,
        )
        self.dropout_rate = hp.dropout
        self.batch_norm = hp.batch_norm
        self.use_checkpointing = hp.gradient_checkpointing
        self.conv_type = hp.conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._edge_dim = self.input_encoder._edge_dim
        self._proj_dim = hp.proj_dim

        gat_in_dim = self.input_encoder.out_dim
        self.gat_in_dim = gat_in_dim
        self.encoder_layers, self.encoder_bns, encoder_targets = build_encoder_stack(
            list(hp.hidden_dims) if hp.hidden_dims else None,
            hp.latent_dim,
            gat_in_dim,
            hp.conv_type,
            self._edge_dim,
            encoder_heads=hp.heads,
            batch_norm=hp.batch_norm,
        )
        self.latent_in_dim = encoder_targets[-1]
        self.z_mean = nn.Linear(self.latent_in_dim, hp.latent_dim)
        self.z_logvar = nn.Linear(self.latent_in_dim, hp.latent_dim)

        decoder_targets = list(reversed(encoder_targets))
        decoder_targets[-1] = hp.in_channels
        self.decoder_layers, self.decoder_bns = build_conv_stack(
            hp.conv_type,
            hp.latent_dim,
            decoder_targets,
            self._edge_dim,
            heads_first=hp.heads,
            batch_norm=hp.batch_norm,
        )
        if hp.batch_norm and len(self.decoder_bns) == len(decoder_targets):
            self.decoder_bns = self.decoder_bns[:-1]

        self.canid_classifier = nn.Linear(hp.latent_dim, hp.num_ids)

        # Neighborhood decoder (GAD-NR / Roy et al., WSDM 2024). Predicts the
        # per-node neighbor-latent mean from z. Output dim is latent_dim
        # (NOT num_ids — that was the previous vocabulary-bag predictor that
        # inherited the UNK cliff and the V100 fp32-overflow surface;
        # 2026-05-06-drop-neighborhood-adopt-tam.md). Paired with
        # `kl_neighbor_loss` in VGAETaskLoss for distributional matching.
        mlp_hidden = hp.mlp_hidden if hp.mlp_hidden is not None else hp.latent_dim
        self.neighborhood_decoder = nn.Sequential(
            nn.Linear(hp.latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(mlp_hidden, hp.latent_dim),
        )

        # Edge-attribute reconstruction head — concatenated endpoint embeddings
        # → predicted edge_attr. Only built when the conv stack actually consumes
        # edge_attr (otherwise edge_attr never enters the latent and a head here
        # would learn from nothing).
        if self._uses_edge_attr and self._edge_dim:
            self.edge_decoder = nn.Sequential(
                nn.Linear(2 * hp.latent_dim, hp.latent_dim),
                nn.ReLU(),
                nn.Dropout(hp.dropout),
                nn.Linear(hp.latent_dim, self._edge_dim),
            )
        else:
            self.edge_decoder = None

        from .._masking import RandomNodeMasker

        # mask_id == num_ids indexes the +1 vocab slot reserved by the id encoder.
        self.masker = RandomNodeMasker(
            in_channels=hp.in_channels,
            mask_id=hp.num_ids,
            mask_rate=hp.mask_rate,
        )

        # Propagate true num_ids into the loss module (constructed with
        # placeholder default before datamodule was attached).
        task_loss = self._task_loss_module()
        if hasattr(task_loss, "num_ids"):
            task_loss.num_ids = hp.num_ids

        if hp.compile_model:
            from ..base import try_compile

            try_compile(self, conv_type=hp.conv_type, dynamic=True)

    def _task_loss_module(self) -> nn.Module:
        """Return base VGAETaskLoss, unwrapping FeatureDistillation if present."""
        return getattr(self.loss_fn, "base_loss", self.loss_fn)

    @staticmethod
    def _rebuild_excluded_kwargs(hp: dict) -> dict:
        """Rebuild ``loss_fn`` from saved hp keys (loss_fn isn't pickleable)."""
        from graphids.core.losses.build import _VGAE_LOSS_KEYS, build_loss

        loss_cfg = {k: hp[k] for k in _VGAE_LOSS_KEYS if k in hp}
        return {"loss_fn": build_loss("vgae", loss_cfg)}

    # ------------------------------------------------------------------
    # Architecture primitives
    # ------------------------------------------------------------------

    def encode(self, x, edge_index, edge_attr=None, batch=None, node_id=None):
        """Returns ``(z, kl_per_node, mu)``."""
        x = self.input_encoder(x, node_id)
        for i, conv in enumerate(self.encoder_layers):
            bn = self.encoder_bns[i] if self.batch_norm else None
            x = conv_forward(
                conv,
                x,
                edge_index,
                edge_attr,
                bn=bn,
                batch=batch,
                dropout_p=self.dropout_rate,
                training=self.training,
                use_checkpointing=self.use_checkpointing,
            )
        mu = self.z_mean(x)
        logvar = self.z_logvar(x).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl_per_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)
        return z, kl_per_node, mu

    def decode_node(self, z, edge_index, edge_attr=None, batch=None):
        assert z.size(-1) == self.hparams.latent_dim, (
            f"Expected {self.hparams.latent_dim}D input, got {z.size(-1)}D"
        )
        x = z

        for i, conv in enumerate(self.decoder_layers):
            if i < len(self.decoder_layers) - 1:
                bn = self.decoder_bns[i] if self.batch_norm else None
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    bn=bn,
                    batch=batch,
                    dropout_p=self.dropout_rate,
                    training=self.training,
                    use_checkpointing=self.use_checkpointing,
                )
            else:
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    activation=None,
                    use_checkpointing=self.use_checkpointing,
                )
        return x

    def _forward_tensors(self, x, edge_index, batch_idx, edge_attr=None, node_id=None):
        """Tensor-form forward → 6-tuple. Used by callers with unpacked tensors.

        Returns ``(cont_out, canid_logits, nbr_pred, z, kl_per_node, edge_logits)``.
        ``nbr_pred`` is ``[N, latent_dim]`` — GAD-NR neighbor-mean prediction.
        ``edge_logits`` is ``None`` when the conv stack doesn't consume edge_attr.
        """
        ea = edge_attr if self._uses_edge_attr else None
        z, kl_per_node, _mu = self.encode(
            x, edge_index, edge_attr=ea, batch=batch_idx, node_id=node_id
        )
        cont_out = self.decode_node(z, edge_index, edge_attr=ea, batch=batch_idx)
        canid_logits = self.canid_classifier(z)
        nbr_pred = self.neighborhood_decoder(z)
        if self.edge_decoder is not None:
            edge_logits = self.edge_decoder(torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1))
        else:
            edge_logits = None
        return cont_out, canid_logits, nbr_pred, z, kl_per_node, edge_logits

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        return self._forward_tensors(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )

    def _masked_forward(self, batch):
        """Training-shape random-masked forward. Returns (outputs, mask)."""
        edge_attr = getattr(batch, "edge_attr", None)
        x_m, nid_m, mask = self.masker(batch.x, batch.node_id)
        outputs = self._forward_tensors(
            x_m, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=nid_m
        )
        return outputs, mask

    def _per_graph_masked_recon(self, cont, x, mask, batch_idx, *, return_components: bool = False):
        """Per-graph masked reconstruction error.

        Default returns the per-graph **mean** over masked nodes (the historical
        behavior). With ``return_components=True`` returns a 3-tuple
        ``(recon_mean, recon_max, recon_per_node)``:

        - ``recon_mean`` ``[G]`` — mean per-node MSE over masked nodes.
        - ``recon_max``  ``[G]`` — max per-node MSE over masked nodes.
          Captures spike anomalies (one nuclear-bad node) that ``recon_mean``
          smears across the window. Graphs with zero masked nodes → 0.
        - ``recon_per_node`` ``[N]`` — raw per-node MSE before reduction.
          Exposed for histogram logging / item-level analysis.
        """
        from torch_geometric.utils import scatter

        recon_per_node = (cont - x).pow(2).mean(dim=-1)
        mask_f = mask.to(recon_per_node.dtype)
        recon_sum = scatter(recon_per_node * mask_f, batch_idx, dim=0, reduce="sum")
        mask_count = scatter(mask_f, batch_idx, dim=0, reduce="sum")
        recon_mean = recon_sum / mask_count.clamp(min=1.0)
        if not return_components:
            return recon_mean
        # Per-graph max over MASKED nodes only. Unmasked nodes get 0; since
        # per-node MSE is always non-negative, 0 cannot win the scatter-max
        # against any real masked-node error, except in the degenerate
        # zero-masked-graph case where the max stays 0 (matches recon_mean).
        masked_err = torch.where(mask, recon_per_node, torch.zeros_like(recon_per_node))
        recon_max = scatter(masked_err, batch_idx, dim=0, reduce="max")
        return recon_mean, recon_max, recon_per_node

    # ------------------------------------------------------------------
    # Trainer-bridge hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, _idx):
        outputs, mask = self._masked_forward(batch)
        loss = self.loss_fn(outputs, batch, mask=mask)
        bs = batch.num_graphs
        self.log("train_loss", loss, batch_size=bs)

        task_loss = self._task_loss_module()
        if task_loss.last_recon is not None:
            self.log("train_recon", task_loss.last_recon, batch_size=bs)
            self.log("train_canid", task_loss.last_canid, batch_size=bs)
            self.log("train_kl", task_loss.last_kl, batch_size=bs)
            if task_loss.last_nbr is not None:
                self.log("train_nbr", task_loss.last_nbr, batch_size=bs)

        log_fn = getattr(self.loss_fn, "log_components", None)
        if log_fn is not None:
            log_fn(self, batch_size=bs, prefix="train_")
        return loss

    def validation_step(self, batch, _idx):
        (cont, _canid, _nbr, z, _kl, _e), mask = self._masked_forward(batch)
        recon, recon_max, recon_per_node = self._per_graph_masked_recon(
            cont, batch.x, mask, batch.batch, return_components=True
        )
        affinity_per_node = tam_affinity(z, batch.edge_index)
        rq = rayleigh_quotient(batch.x, batch.edge_index, batch=batch.batch)

        bs = batch.num_graphs
        self.log("val_loss", recon.mean(), batch_size=bs)
        self.log("val_recon_max_mean", recon_max.mean(), batch_size=bs)
        self.log("val_rq_mean", rq.mean(), batch_size=bs)

        y = batch.y.view(-1)
        sub: dict[str, torch.Tensor] = {}
        sub_max: dict[str, torch.Tensor] = {}
        for label, m in (("benign", y == 0), ("attack", y != 0)):
            n = int(m.sum())
            if not n:
                continue
            v = recon[m].mean()
            vmax = recon_max[m].mean()
            self.log(f"val_loss_{label}", v, batch_size=n)
            self.log(f"val_recon_max_{label}", vmax, batch_size=n)
            self.log(f"val_rq_{label}", rq[m].mean(), batch_size=n)
            sub[label] = v
            sub_max[label] = vmax
        if "benign" in sub and "attack" in sub:
            self.log("val_discrimination_gap", sub["attack"] - sub["benign"], batch_size=bs)
            self.log(
                "val_discrimination_ratio",
                sub["attack"] / (sub["benign"] + 1e-6),
                batch_size=bs,
            )
            self.log(
                "val_recon_max_gap",
                sub_max["attack"] - sub_max["benign"],
                batch_size=bs,
            )

        # Per-node histogram (masked nodes only for recon; all nodes for TAM
        # affinity). Reduction is light — a handful of percentile scalars
        # per batch per class.
        if mask.any():
            mn = recon_per_node[mask]
            node_y = y[batch.batch][mask]
            for label, m in (("benign", node_y == 0), ("attack", node_y != 0)):
                n = int(m.sum())
                if not n:
                    continue
                vals = mn[m]
                qs = torch.quantile(vals, vals.new_tensor([0.5, 0.95, 0.99]))
                self.log(f"val_node_recon_p50_{label}", qs[0], batch_size=n)
                self.log(f"val_node_recon_p95_{label}", qs[1], batch_size=n)
                self.log(f"val_node_recon_p99_{label}", qs[2], batch_size=n)
                self.log(f"val_node_recon_max_{label}", vals.max(), batch_size=n)

        node_y_all = y[batch.batch]
        for label, m in (("benign", node_y_all == 0), ("attack", node_y_all != 0)):
            n = int(m.sum())
            if not n:
                continue
            vals = affinity_per_node[m]
            qs = torch.quantile(vals, vals.new_tensor([0.5, 0.95, 0.99]))
            self.log(f"val_node_affinity_p50_{label}", qs[0], batch_size=n)
            self.log(f"val_node_affinity_p95_{label}", qs[1], batch_size=n)
            self.log(f"val_node_affinity_p99_{label}", qs[2], batch_size=n)
            self.log(f"val_node_affinity_max_{label}", vals.max(), batch_size=n)

    def _score(
        self, batch
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        ea = edge_attr if self._uses_edge_attr else None

        z, kl_per_node, mu = self.encode(
            batch.x,
            batch.edge_index,
            edge_attr=ea,
            batch=batch.batch,
            node_id=batch.node_id,
        )
        mahal_per_node = ((mu - self.mu_mean) / self.mu_std.clamp(min=1e-3)).pow(2).sum(-1)
        affinity_per_node = tam_affinity(z, batch.edge_index)
        rq = rayleigh_quotient(batch.x, batch.edge_index, batch=batch.batch)

        (cont, _canid, _nbr, _z2, _kl2, _e), mask = self._masked_forward(batch)
        recon, recon_max, _per_node = self._per_graph_masked_recon(
            cont, batch.x, mask, batch.batch, return_components=True
        )

        mahal = scatter(mahal_per_node, batch.batch, dim=0, reduce="mean")
        kl = scatter(kl_per_node, batch.batch, dim=0, reduce="mean")
        affinity = scatter(affinity_per_node, batch.batch, dim=0, reduce="mean")
        return recon, recon_max, affinity, rq, mahal, kl, z

    def extract_features(self, batch, device: torch.device) -> dict[str, torch.Tensor]:
        """Per-graph fusion features as named tensors.

        - ``errors``   [N, 3] — recon, mahal, kl (anomaly evidence)
        - ``conf``     [N, 1] — 1 / (1 + recon)
        - ``z_stats``  [N, 4] — z_mean, z_std, z_max, z_min
        - ``spike``    [N, 1] — recon_max (per-graph max masked-node MSE)
        - ``affinity`` [N, 1] — TAM per-graph mean affinity
        - ``rq``       [N, 1] — Rayleigh quotient (input-space spectral smoothness)
        """
        from torch_geometric.utils import scatter

        recon, recon_max, affinity, rq, mahal, kl, z = self._score(batch)
        b = batch.batch
        z_mean = scatter(z.mean(1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(1).values, b, dim=0, reduce="min")
        return {
            "errors": torch.stack([recon, mahal, kl], dim=1),
            "conf": (1.0 / (1.0 + recon)).unsqueeze(-1),
            "z_stats": torch.stack([z_mean, z_std, z_max, z_min], dim=1),
            "spike": recon_max.unsqueeze(-1),
            "affinity": affinity.unsqueeze(-1),
            "rq": rq.unsqueeze(-1),
        }

    def score(self, batch) -> torch.Tensor:
        """Per-graph anomaly score: max-σ over (recon, recon_max, TAM affinity, RQ)
        in the calibrated z-norm space."""
        if not bool(self.score_norm_fitted):
            raise RuntimeError(
                "VGAE scoring requires on_test_setup() to have run. "
                "If loading an old ckpt without masker.mask_token, retrain under "
                "the mask-recon code or use the legacy scoring path."
            )
        recon, recon_max, affinity, rq, _mahal, _kl, _z = self._score(batch)
        eps = 1e-6
        zs = []
        for name, v in (
            ("recon", recon),
            ("recon_max", recon_max),
            ("affinity", affinity),
            ("rq", rq),
        ):
            mean = getattr(self, f"score_{name}_mean")
            std = getattr(self, f"score_{name}_std")
            zs.append((v - mean) / (std + eps))
        return torch.stack(zs, dim=0).amax(dim=0)

    def on_test_setup(self, datamodule, device) -> None:
        """Fit z-norm calibration buffers from benign val if not already
        populated. Idempotent: skips if a calibrated ckpt was reloaded."""
        if not bool(self.score_norm_fitted):
            self._fit_score_norm(datamodule.val_dataloader(), device)

    @torch.no_grad()
    def _fit_score_norm(self, val_loader, device: torch.device) -> None:
        """Two-pass calibration on benign val. Full-batch forward + boolean
        mask on the resulting per-node / per-graph tensors — never decompose
        the packed batch (matches the dataset-time benign filter convention
        in :mod:`graphids.core.data.datamodule.graph`)."""
        was_training = self.training
        self.eval()
        try:
            mus: list[torch.Tensor] = []
            for batch in val_loader:
                batch = batch.clone().to(device)
                y_g = batch.y.view(-1)
                if not (y_g == 0).any():
                    continue
                ea = getattr(batch, "edge_attr", None) if self._uses_edge_attr else None
                _z, _kl, mu = self.encode(
                    batch.x,
                    batch.edge_index,
                    edge_attr=ea,
                    batch=batch.batch,
                    node_id=batch.node_id,
                )
                benign_node = y_g[batch.batch] == 0
                mus.append(mu[benign_node].cpu())
            if not mus:
                raise RuntimeError("_fit_score_norm: no benign rows in val loader")
            mu_all = torch.cat(mus, dim=0)
            self.mu_mean.copy_(mu_all.mean(dim=0).to(self.mu_mean.device))
            self.mu_std.copy_(mu_all.std(dim=0).clamp(min=1e-3).to(self.mu_std.device))

            accum: dict[str, list[torch.Tensor]] = {n: [] for n in self._SCORE_COMPONENTS}
            for batch in val_loader:
                batch = batch.clone().to(device)
                y_g = batch.y.view(-1)
                benign_g = y_g == 0
                if not benign_g.any():
                    continue
                recon, recon_max, affinity, rq, _mahal, _kl, _z = self._score(batch)
                accum["recon"].append(recon[benign_g].cpu())
                accum["recon_max"].append(recon_max[benign_g].cpu())
                accum["affinity"].append(affinity[benign_g].cpu())
                accum["rq"].append(rq[benign_g].cpu())

            n_total = sum(len(t) for t in accum["recon"])
            if n_total < 100:
                raise RuntimeError(f"_fit_score_norm: need >=100 benign val graphs, got {n_total}")
            for name in self._SCORE_COMPONENTS:
                cat = torch.cat(accum[name])
                getattr(self, f"score_{name}_mean").copy_(cat.mean())
                getattr(self, f"score_{name}_std").copy_(cat.std())
            self.score_norm_fitted.fill_(True)
        finally:
            if was_training:
                self.train()
