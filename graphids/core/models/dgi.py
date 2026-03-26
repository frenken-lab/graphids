"""Deep Graph Infomax: contrastive self-supervised graph representation learning.

Maximizes mutual information between node embeddings and a graph-level summary
via a bilinear discriminator. Uses the same encoder backbone as VGAE (InputEncoder
+ conv stack) for fair ablation comparison.

Reference: Veličković et al., "Deep Graph Infomax" (ICLR 2019).
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from ._conv import InputEncoder, build_encoder_stack, conv_forward, resolve_edge_dim
from ._training import OOMSkipMixin, build_optimizer_dict, binary_test_metrics


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
        conv_type = cfg.dgi.conv_type
        return cls(
            num_ids=num_ids,
            in_channels=in_ch,
            hidden_dims=list(cfg.dgi.hidden_dims),
            latent_dim=cfg.dgi.latent_dim,
            encoder_heads=cfg.dgi.heads,
            embedding_dim=cfg.dgi.embedding_dim,
            dropout=cfg.dgi.dropout,
            conv_type=conv_type,
            edge_dim=resolve_edge_dim(conv_type, cfg.dgi.edge_dim),
            proj_dim=cfg.dgi.proj_dim,
            use_checkpointing=cfg.training.gradient_checkpointing,
        )


# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class DGIModule(OOMSkipMixin, pl.LightningModule):
    """DGI contrastive training: maximize node–summary mutual information.

    Anomaly scoring at test time uses discriminator confidence:
    low discriminator agreement → anomalous graph.
    """

    def __init__(self, cfg, teacher=None, projection=None):
        super().__init__()
        from graphids.config import _ns_to_dict, to_namespace
        cfg = to_namespace(cfg)
        self.save_hyperparameters(ignore=["cfg", "teacher", "projection"])
        self.save_hyperparameters({"cfg": _ns_to_dict(cfg)})
        num_ids, in_channels = cfg.num_ids, cfg.in_channels
        self.cfg = cfg
        self.model = GraphInfomaxModel.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, dynamic=True)
        self.test_threshold: float | None = None
        self.test_metrics = binary_test_metrics()

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

    def test_step(self, batch, _idx):
        scores = self._per_graph_scores(batch)
        if self.test_threshold is not None:
            self.test_metrics.update(scores, batch.y)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_threshold is not None:
            self.log_dict(self.test_metrics.compute())

    def predict_step(self, batch, _idx):
        scores = self._per_graph_scores(batch)
        return {"scores": scores, "labels": batch.y}

    def find_threshold(self, data: list, batch_size: int = 256) -> tuple[float, float]:
        """Find optimal anomaly threshold via Youden's J on validation data."""
        from torchmetrics.functional.classification import binary_roc

        from graphids.core.preprocessing.datamodule import make_graph_loader

        trainer = pl.Trainer(
            accelerator="auto", devices="auto",
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
        )
        loader = make_graph_loader(data, batch_size=batch_size)
        preds = trainer.predict(self, dataloaders=loader)

        scores = torch.cat([p["scores"] for p in preds]).cpu()
        labels = torch.cat([p["labels"] for p in preds]).cpu()

        if len(scores) == 0:
            return 0.5, 0.0
        if labels.unique().numel() < 2:
            return float(scores.median()), 0.0

        fpr_v, tpr_v, thresholds_v = binary_roc(scores, labels.long())
        j_scores = tpr_v - fpr_v
        if len(j_scores) == 0 or len(thresholds_v) == 0:
            return float(scores.median()), 0.0
        best_idx = torch.argmax(j_scores).item()
        thresh = float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(scores.median())
        return thresh, float(j_scores[best_idx])

    @classmethod
    def evaluate(cls, cfg, val_data, test_scenarios, device, *, load_model_fn) -> dict:
        """Evaluate DGI: threshold search on val, then test scenarios."""
        from ._training import eval_with_scenarios, gpu_cleanup
        model = load_model_fn(cfg, "dgi", "autoencoder", device)
        module = cls(cfg)
        module.model = model

        bs = cfg.evaluation.batch_size
        threshold, youden_j = module.find_threshold(val_data, batch_size=bs)
        module.test_threshold = threshold
        module.test_metrics.reset()

        val_metrics, scenario_metrics = eval_with_scenarios(
            module, val_data, test_scenarios, bs,
        )
        val_metrics["optimal_threshold"] = threshold
        val_metrics["youden_j"] = youden_j

        gpu_cleanup(model)
        return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": None}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        return build_optimizer_dict(opt, self.cfg)
