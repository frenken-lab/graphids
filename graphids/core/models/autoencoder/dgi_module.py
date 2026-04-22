from __future__ import annotations

import torch

from graphids.config.constants import (
    ModelType,  # noqa: F401 (used in __init__ annotation)
)

from ..base import GraphModuleBase, binary_test_metrics
from .dgi import GraphInfomaxModel

# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------


class DGIModule(GraphModuleBase):
    """DGI contrastive training: maximize node-summary mutual information.

    Training: DGI contrastive loss on (pos_z, neg_z, summary) triples.
    Anomaly scoring at test time: OCGIN-style L2 distance between the pooled
    node embedding of a query graph and the centroid of training-normal
    pooled embeddings (Zhao & Akoglu 2021, arxiv:2103.04494). The
    discriminator-confidence score used in DGI's original formulation is a
    known-bad signal for graph-level anomaly detection — it saturates to ~1
    on all normal-ish graphs once the contrastive loss converges, collapsing
    the score range for both benign and attack windows. Centroid distance
    in latent space preserves separability because the encoder's contrastive
    objective produces representations where out-of-distribution graphs
    drift away from the training manifold.
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
        id_encoder_class_path: str = "graphids.core.models.id_encoding.LookupIdEncoder",
        id_encoder_kwargs: dict | None = None,
        # --- training ---
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: ModelType = "dgi",
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.proj_dim = proj_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_model = compile_model
        self.scale = scale
        self.model_type = model_type
        self.dataset = dataset
        self.seed = seed
        self.num_ids = num_ids
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.id_encoder_class_path = id_encoder_class_path
        self.id_encoder_kwargs = id_encoder_kwargs or {}
        self.model = None
        self._init_threshold_metrics()
        self.test_metrics = binary_test_metrics()
        # OCGIN scoring head: centroid of training-normal pooled embeddings.
        # Populated by calibrate_svdd_center() at on_fit_end; persists through
        # state_dict so test/analyze paths pick it up automatically.
        self.register_buffer("svdd_center", torch.zeros(latent_dim))
        self.register_buffer("svdd_calibrated", torch.tensor(False))
        if num_ids > 0:
            self._build()

    def _build(self):
        from graphids._reflect import import_class

        hp = self.hparams
        encoder_cls = import_class(hp.id_encoder_class_path)
        encoder_kwargs = {"embedding_dim": hp.embedding_dim, **(hp.id_encoder_kwargs or {})}
        id_encoder = encoder_cls.from_vocab_size(num_ids=hp.num_ids, **encoder_kwargs)
        self.model = GraphInfomaxModel.from_config(hp, id_encoder, hp.in_channels)
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

    def _step(self, batch):
        pos_z, neg_z, summary = self(batch)
        return self.model.dgi_loss(pos_z, neg_z, summary, batch.batch)

    def extract_features(self, batch, device: torch.device) -> torch.Tensor:
        """8-D fusion features: [anomaly, pos_mean, pos_spread, z_mean, z_std, z_max, z_min, conf].

        Shape matches VGAE/GAT (8D / 7D) for concat-friendly fusion state vectors.
        Features are derived from the DGI discriminator (node–summary agreement):
        graphs with low real-vs-summary agreement are anomalous.
        """
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        z = self.model.encode(
            batch.x,
            batch.edge_index,
            edge_attr,
            batch.batch,
            batch.node_id,
        )
        summary = self.model.summarize(z, batch.batch)
        pos_scores = self.model.discriminate(z, summary, batch.batch)  # per-node [0,1]

        b = batch.batch
        pos_mean = scatter(pos_scores, b, dim=0, reduce="mean")
        # per-graph std via E[X²] − E[X]² (scatter has no native "std" reduce)
        pos_sq_mean = scatter(pos_scores.pow(2), b, dim=0, reduce="mean")
        pos_spread = (pos_sq_mean - pos_mean.pow(2)).clamp(min=0).sqrt()
        anomaly = 1.0 - pos_mean

        z_mean = scatter(z.mean(1), b, dim=0, reduce="mean")
        z_std = scatter(z.std(1), b, dim=0, reduce="mean")
        z_max = scatter(z.max(1).values, b, dim=0, reduce="max")
        z_min = scatter(z.min(1).values, b, dim=0, reduce="min")
        conf = 1.0 / (1.0 + anomaly)
        return torch.stack(
            [anomaly, pos_mean, pos_spread, z_mean, z_std, z_max, z_min, conf],
            dim=1,
        )

    def _training_step_inner(self, batch, _idx):
        loss = self._step(batch)
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("val_loss", loss, batch_size=batch.num_graphs)

    def _pooled_latent(self, batch) -> torch.Tensor:
        """Per-graph pooled latent: mean(encode(x)) over nodes in each graph."""
        from torch_geometric.nn import global_mean_pool

        z = self.model.encode(
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            batch.batch,
            batch.node_id,
        )
        return global_mean_pool(z, batch.batch)  # [num_graphs, latent_dim]

    def _per_graph_scores(self, batch):
        """OCGIN score: L2 distance from SVDD centroid in pooled-latent space.

        Requires ``calibrate_svdd_center`` to have run at least once; raises
        if called before calibration (no silent fallback to discriminator
        scoring — that's the failure mode we're fixing).
        """
        if not bool(self.svdd_calibrated.item()):
            raise RuntimeError(
                "DGIModule.svdd_center is uncalibrated. Run "
                "calibrate_svdd_center(train_loader, device) after fit or "
                "load a checkpoint with svdd_calibrated=True."
            )
        pooled = self._pooled_latent(batch)
        return (pooled - self.svdd_center).pow(2).sum(dim=1)

    @torch.no_grad()
    def calibrate_svdd_center(self, train_loader, device) -> None:
        """Fit ``svdd_center`` = mean of pooled latents over training-normal graphs.

        Runs a single pass over ``train_loader`` in eval mode. The caller is
        responsible for passing the benign-only loader (the autoencoder stage
        already filters with ``label_filter='benign'``).
        """
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
            raise RuntimeError("calibrate_svdd_center: empty train loader")
        self.svdd_center.copy_(total / count)
        self.svdd_calibrated.fill_(True)
        if was_training:
            self.train()

    def test_step(self, batch, _idx, dataloader_idx=0):
        scores = self._per_graph_scores(batch)
        self.roc_metric.update(scores.detach(), batch.y.detach())
        self._record_test_batch(dataloader_idx, scores=scores, labels=batch.y)

    def on_test_epoch_end(self):
        self._log_thresholded_metrics()

    def predict_step(self, batch, _idx):
        scores = self._per_graph_scores(batch)
        return {"scores": scores, "labels": batch.y}
