"""Deep Graph Infomax — collapsed arch + trainer-bridge.

Maximizes mutual information between node embeddings and a graph-level
summary via a bilinear discriminator. Uses the same encoder backbone as
VGAE (InputEncoder + conv stack) for fair ablation comparison.

Anomaly scoring at test time: OCGIN-style L2 distance between the pooled
node embedding of a query graph and the centroid of training-normal
pooled embeddings (Zhao & Akoglu 2021, arxiv:2103.04494).

Reference: Veličković et al., "Deep Graph Infomax" (ICLR 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from graphids.paths import ModelType

from .._conv import InputEncoder, build_encoder_stack, conv_forward, resolve_edge_dim
from .._detector import ScoreBasedDetectorMixin


class DGI(ScoreBasedDetectorMixin):
    """Collapsed DGI — arch + trainer-bridge in one ``nn.Module``.

    No ``loss_fn`` kwarg: the contrastive MI loss is intrinsic to the
    architecture (built into the discriminator).
    """

    _SCALES: dict[str, dict[str, int]] = {
        "small": {"latent_dim": 48, "embedding_dim": 32, "heads": 4},
        "large": {"latent_dim": 96, "embedding_dim": 64, "heads": 8},
    }

    def __init__(
        self,
        # --- architecture (latent_dim/embedding_dim/heads=None → scale) ---
        conv_type: str = "gatv2",
        hidden_dims: list[int] | None = None,
        latent_dim: int | None = None,
        heads: int | None = None,
        embedding_dim: int | None = None,
        dropout: float = 0.15,
        edge_dim: int = 11,
        proj_dim: int = 0,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        batch_norm: bool = True,
        id_encoder_class_path: str = "graphids.core.models.id_encoding.LookupIdEncoder",
        id_encoder_kwargs: dict | None = None,
        # --- training ---
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: ModelType = "dgi",
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        s = self._SCALES.get(scale, {})
        if latent_dim is None:
            latent_dim = s.get("latent_dim", 48)
        if embedding_dim is None:
            embedding_dim = s.get("embedding_dim", 32)
        if heads is None:
            heads = s.get("heads", 4)
        super().__init__()
        # OCGIN scoring head: centroid of training-normal pooled embeddings.
        # Re-fit at test-start by ``Trainer.test`` via ``on_test_setup`` —
        # the centroid is a deterministic statistic of (encoder weights,
        # benign train data). Zero init means an uncalibrated forward pass
        # raises in ``score`` rather than returning bogus scores.
        self.register_buffer("svdd_center", torch.zeros(latent_dim))
        self._init_post(locals())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        hp = self.hparams
        id_encoder = self._build_id_encoder()
        edge_dim = resolve_edge_dim(hp.conv_type, hp.edge_dim)

        self.dropout_rate = hp.dropout
        self.batch_norm = hp.batch_norm
        self.use_checkpointing = hp.gradient_checkpointing
        self.conv_type = hp.conv_type

        self.input_encoder = InputEncoder(
            id_encoder=id_encoder,
            in_channels=hp.in_channels,
            conv_type=hp.conv_type,
            edge_dim=edge_dim,
            proj_dim=hp.proj_dim,
        )
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._edge_dim = self.input_encoder._edge_dim

        gat_in_dim = self.input_encoder.out_dim
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
        self.z_proj = nn.Linear(self.latent_in_dim, hp.latent_dim)

        self.discriminator_weight = nn.Parameter(torch.empty(hp.latent_dim, hp.latent_dim))
        nn.init.xavier_uniform_(self.discriminator_weight)

        if hp.compile_model:
            from ..base import try_compile

            try_compile(self, conv_type=hp.conv_type, dynamic=True)

    # ------------------------------------------------------------------
    # Architecture primitives
    # ------------------------------------------------------------------

    def encode(self, x, edge_index, edge_attr=None, batch=None, node_id=None):
        """Encode nodes to latent embeddings (same contract as VGAE minus KL)."""
        x = self.input_encoder(x, node_id)
        ea = edge_attr if self._uses_edge_attr else None

        for i, conv in enumerate(self.encoder_layers):
            bn = self.encoder_bns[i] if self.batch_norm else None
            x = conv_forward(
                conv,
                x,
                edge_index,
                ea,
                bn=bn,
                batch=batch,
                dropout_p=self.dropout_rate,
                training=self.training,
                use_checkpointing=self.use_checkpointing,
            )
        return self.z_proj(x)

    def summarize(self, z, batch):
        """Graph-level summary: sigmoid(mean_pool(z))."""
        return torch.sigmoid(global_mean_pool(z, batch))

    def discriminate(self, z, summary, batch):
        """Bilinear scoring: sigmoid(z^T W s) per node."""
        s = summary[batch]
        return torch.sigmoid((z @ self.discriminator_weight * s).sum(dim=1))

    def _forward_tensors(self, x, edge_index, batch_idx, edge_attr=None, node_id=None):
        """Tensor-form forward → (pos_z, neg_z, summary)."""
        ea = edge_attr if self._uses_edge_attr else None
        pos_z = self.encode(x, edge_index, ea, batch_idx, node_id)
        perm = torch.randperm(x.size(0), device=x.device)
        neg_z = self.encode(x[perm], edge_index, ea, batch_idx, node_id)
        summary = self.summarize(pos_z, batch_idx)
        return pos_z, neg_z, summary

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        return self._forward_tensors(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
        )

    def dgi_loss(self, pos_z, neg_z, summary, batch_idx):
        """Contrastive MI loss: maximize real node–summary agreement."""
        EPS = 1e-6
        pos_score = self.discriminate(pos_z, summary, batch_idx)
        neg_score = self.discriminate(neg_z, summary, batch_idx)
        return -torch.log(pos_score + EPS).mean() - torch.log(1 - neg_score + EPS).mean()

    # ------------------------------------------------------------------
    # Trainer-bridge hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("val_loss", loss, batch_size=batch.num_graphs)

    def _pooled_latent(self, batch) -> torch.Tensor:
        """Per-graph pooled latent."""
        z = self.encode(
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            batch.batch,
            batch.node_id,
        )
        return global_mean_pool(z, batch.batch)

    def score(self, batch) -> torch.Tensor:
        """OCGIN score: L2 distance from SVDD centroid in pooled-latent space."""
        if not torch.any(self.svdd_center):
            raise RuntimeError(
                "DGI.svdd_center is zero. Call "
                "on_test_setup(datamodule, device) before scoring "
                "(Trainer.test does this automatically for the test phase)."
            )
        pooled = self._pooled_latent(batch)
        return (pooled - self.svdd_center).pow(2).sum(dim=1)

    def on_test_setup(self, datamodule, device) -> None:
        """Fit SVDD center from training-normal graphs at test-start.
        Always re-fits (no idempotence flag — center isn't persisted in
        state_dict; see Cardinal jid 8772115 for ckpt-ordering rationale)."""
        self._calibrate_svdd_center(datamodule.train_eval_dataloader(), device)

    @torch.no_grad()
    def _calibrate_svdd_center(self, train_loader, device) -> None:
        """Fit svdd_center = mean of pooled latents over training-normal graphs."""
        was_training = self.training
        self.eval()
        total = torch.zeros(self.hparams.latent_dim, device=device)
        count = 0
        for batch in train_loader:
            batch = batch.clone().to(device)
            pooled = self._pooled_latent(batch)
            total += pooled.sum(dim=0)
            count += pooled.shape[0]
        if count == 0:
            raise RuntimeError("_calibrate_svdd_center: empty train loader")
        self.svdd_center.copy_(total / count)
        if was_training:
            self.train()

    def extract_features(self, batch, device: torch.device) -> dict[str, torch.Tensor]:
        """Per-graph fusion features as named tensors (symmetric to VGAE/GAT).

        - ``pos_stats`` [N, 3] — anomaly, pos_mean, pos_spread (discriminator-derived)
        - ``conf``      [N, 1] — 1 / (1 + anomaly)
        - ``z_stats``   [N, 4] — z_mean, z_std, z_max, z_min (latent-pooled)
        """
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        z = self.encode(
            batch.x,
            batch.edge_index,
            edge_attr,
            batch.batch,
            batch.node_id,
        )
        summary = self.summarize(z, batch.batch)
        pos_scores = self.discriminate(z, summary, batch.batch)

        b = batch.batch
        pos_mean = scatter(pos_scores, b, dim=0, reduce="mean")
        pos_sq_mean = scatter(pos_scores.pow(2), b, dim=0, reduce="mean")
        pos_spread = (pos_sq_mean - pos_mean.pow(2)).clamp(min=0).sqrt()
        anomaly = 1.0 - pos_mean

        z_mean = scatter(z.mean(1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(1).values, b, dim=0, reduce="min")
        return {
            "pos_stats": torch.stack([anomaly, pos_mean, pos_spread], dim=1),
            "conf": (1.0 / (1.0 + anomaly)).unsqueeze(-1),
            "z_stats": torch.stack([z_mean, z_std, z_max, z_min], dim=1),
        }
