from __future__ import annotations

import os

from ..base import GraphModuleBase, binary_test_metrics

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
        if hp.compile_model:
            from ..base import try_compile

            self.model = try_compile(self.model, conv_type=hp.conv_type, dynamic=True)

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        return self.model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_attr,
            node_id=batch.node_id,
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
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            batch.batch,
            batch.node_id,
        )
        summary = self.model.summarize(pos_z, batch.batch)
        node_scores = self.model.discriminate(pos_z, summary, batch.batch)
        return 1 - scatter(node_scores, batch.batch, dim=0, reduce="mean")

    def test_step(self, batch, _idx, dataloader_idx=0):
        scores = self._per_graph_scores(batch)
        self.roc_metric.update(scores.detach(), batch.y.detach())

    def on_test_epoch_end(self):
        self._log_thresholded_metrics()

    def predict_step(self, batch, _idx):
        scores = self._per_graph_scores(batch)
        return {"scores": scores, "labels": batch.y}
