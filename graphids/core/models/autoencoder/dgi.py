"""Deep Graph Infomax: contrastive self-supervised graph representation learning.

Maximizes mutual information between node embeddings and a graph-level summary
via a bilinear discriminator. Uses the same encoder backbone as VGAE (InputEncoder
+ conv stack) for fair ablation comparison.

Reference: Veličković et al., "Deep Graph Infomax" (ICLR 2019).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .._conv import InputEncoder, build_encoder_stack, conv_forward, resolve_edge_dim
from .._training import GraphModuleBase, binary_test_metrics


class GraphInfomaxModel(nn.Module):
    """DGI model with shared VGAE encoder backbone.

    Forward returns ``(pos_z, neg_z, summary)`` for contrastive loss.
    ``encode()`` returns node embeddings for downstream use.
    """

    def __init__(
        self,
        num_ids: int,
        in_channels: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 48,
        encoder_heads: int = 4,
        embedding_dim: int = 32,
        dropout: float = 0.15,
        batch_norm: bool = True,
        use_checkpointing: bool = False,
        conv_type: str = "gatv2",
        edge_dim: int | None = 11,
        proj_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type

        # Shared input encoding (same as VGAE)
        self.input_encoder = InputEncoder(
            num_ids=num_ids,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            conv_type=conv_type,
            edge_dim=edge_dim,
            proj_dim=proj_dim,
        )
        self.num_ids = num_ids
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._edge_dim = self.input_encoder._edge_dim

        # Encoder conv stack (same architecture as VGAE encoder)
        gat_in_dim = self.input_encoder.out_dim
        self.encoder_layers, self.encoder_bns, self.latent_in_dim = build_encoder_stack(
            hidden_dims, latent_dim, gat_in_dim, conv_type, self._edge_dim,
            encoder_heads=encoder_heads, batch_norm=batch_norm,
        )
        self.z_proj = nn.Linear(self.latent_in_dim, latent_dim)

        # Bilinear discriminator: scores node–summary pairs
        self.discriminator_weight = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.discriminator_weight)

    def encode(self, x, edge_index, edge_attr=None, batch=None, node_id=None):
        """Encode nodes to latent embeddings (same contract as VGAE minus KL)."""
        x = self.input_encoder(x, node_id)
        ea = edge_attr if self._uses_edge_attr else None

        for i, conv in enumerate(self.encoder_layers):
            bn = self.encoder_bns[i] if self.batch_norm else None
            x = conv_forward(
                conv, x, edge_index, ea,
                bn=bn, batch=batch,
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
        s = summary[batch]  # expand summary to node level
        return torch.sigmoid((z @ self.discriminator_weight * s).sum(dim=1))

    def forward(self, x, edge_index, batch, edge_attr=None, node_id=None, **kwargs):
        """DGI forward: encode positive + corrupted, compute summary.

        Returns:
            pos_z: node embeddings from real graph [num_nodes, latent_dim]
            neg_z: node embeddings from corrupted graph [num_nodes, latent_dim]
            summary: graph-level summary [num_graphs, latent_dim]
        """
        ea = edge_attr if self._uses_edge_attr else None
        pos_z = self.encode(x, edge_index, ea, batch, node_id)
        # Corruption: shuffle node features (standard DGI approach)
        perm = torch.randperm(x.size(0), device=x.device)
        neg_z = self.encode(x[perm], edge_index, ea, batch, node_id)
        summary = self.summarize(pos_z, batch)
        return pos_z, neg_z, summary

    def dgi_loss(self, pos_z, neg_z, summary, batch):
        """Contrastive MI loss: maximize real node–summary agreement."""
        EPS = 1e-6
        pos_score = self.discriminate(pos_z, summary, batch)
        neg_score = self.discriminate(neg_z, summary, batch)
        return -torch.log(pos_score + EPS).mean() - torch.log(1 - neg_score + EPS).mean()

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> GraphInfomaxModel:
        """Construct from config (same interface as VGAE/GAT)."""
        conv_type = cfg.conv_type
        return cls(
            num_ids=num_ids,
            in_channels=in_ch,
            hidden_dims=list(cfg.hidden_dims),
            latent_dim=cfg.latent_dim,
            encoder_heads=cfg.heads,
            embedding_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
            conv_type=conv_type,
            edge_dim=resolve_edge_dim(conv_type, cfg.edge_dim),
            proj_dim=cfg.proj_dim,
            use_checkpointing=cfg.gradient_checkpointing,
        )


# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class DGIModule(GraphModuleBase):
    """DGI contrastive training: maximize node–summary mutual information.

    Anomaly scoring at test time uses discriminator confidence:
    low discriminator agreement → anomalous graph.
    """

    def __init__(
        self,
        # --- architecture ---
        conv_type: str = "gatv2",
        hidden_dims: list[int] | None = None,
        latent_dim: int = 48,
        heads: int = 4,
        embedding_dim: int = 32,
        dropout: float = 0.15,
        edge_dim: int = 11,
        proj_dim: int = 0,
        # --- training ---
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: str = "dgi",
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self._init_threshold_metrics()
        self.test_metrics = binary_test_metrics()
        if num_ids > 0:
            self._build()

    def _build(self):
        hp = self.hparams
        self.model = GraphInfomaxModel.from_config(hp, hp.num_ids, hp.in_channels)
        if hp.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, dynamic=True)

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        return self.model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=edge_attr, node_id=batch.node_id,
        )

    def _training_step_inner(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def _per_graph_scores(self, batch):
        """Compute per-graph anomaly scores (1 - mean discriminator confidence)."""
        from torch_geometric.utils import scatter
        pos_z = self.model.encode(
            batch.x, batch.edge_index, getattr(batch, "edge_attr", None),
            batch.batch, batch.node_id,
        )
        summary = self.model.summarize(pos_z, batch.batch)
        node_scores = self.model.discriminate(pos_z, summary, batch.batch)
        return 1 - scatter(node_scores, batch.batch, dim=0, reduce="mean")

    def test_step(self, batch, _idx, dataloader_idx=0):
        scores = self._per_graph_scores(batch)
        self.roc_metric.update(scores.detach(), batch.y.detach())

    def on_test_epoch_start(self):
        self.test_metrics.reset()
        self.roc_metric.reset()

    def on_test_epoch_end(self):
        if not self.roc_metric.preds:
            return

        scores = torch.cat(self.roc_metric.preds).cpu()
        labels = torch.cat(self.roc_metric.target).cpu().long()

        if self.test_threshold is None:
            threshold = self._find_threshold()
            if threshold is None:
                self.test_threshold = float(scores.median())
            else:
                self.test_threshold = threshold

        preds = (scores >= self.test_threshold).long()
        self.test_metrics.update(preds, labels)
        metrics = self.test_metrics.compute()
        metrics["threshold"] = self.test_threshold
        self.log_dict(metrics)

    def on_save_checkpoint(self, checkpoint):
        if self.test_threshold is not None:
            checkpoint["test_threshold"] = self.test_threshold

    def on_load_checkpoint(self, checkpoint):
        self.test_threshold = checkpoint.get("test_threshold")

    def predict_step(self, batch, _idx):
        scores = self._per_graph_scores(batch)
        return {"scores": scores, "labels": batch.y}
