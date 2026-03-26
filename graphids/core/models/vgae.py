from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._conv import InputEncoder, build_conv_stack, build_encoder_stack, _make_conv, conv_forward, resolve_edge_dim
from ._training import (
    OOMSkipMixin, soft_label_kd_loss, _get_kd_config, teacher_on_device,
    build_optimizer_dict, compute_node_budget, NodeBudgetInfo,
    binary_test_metrics,
)


@dataclass(frozen=True)
class VGAEResult:
    """Artifacts from VGAE evaluation: errors, embeddings, loss components."""
    errors: np.ndarray
    labels: np.ndarray
    attack_types: np.ndarray
    embeddings: np.ndarray | None = None
    components: dict[str, np.ndarray] | None = None


class GraphAutoencoderNeighborhood(nn.Module):
    """
    Graph Autoencoder that reconstructs node features and edge list.

    This implementation follows a *progressive compression schedule* defined by
    `hidden_dims`, which is a list like [256, 128, 96, 48]. The last element is
    typically the `latent_dim` and is *not* used as a GAT output size; instead
    the encoder builds GAT layers targeting the preceding entries and the final
    latent `z` is produced via linear heads mapping the last encoder output to
    `latent_dim`.

    Decoder is the mirror of the encoder (reverse of the progressive schedule)
    and the final decoder GAT produces the reconstructed continuous features.

    Args (key):
      - num_ids: number of CAN ID tokens
      - in_channels: input channel count (including CAN ID as first column)
      - hidden_dims: compression schedule, e.g., [256,128,96,48] (last element is latent_dim)
      - latent_dim: dimensionality of latent `z` (if None and hidden_dims provided, inferred as hidden_dims[-1])
      - encoder_heads: number of heads for the first encoder layer (others default to 1)
      - decoder_heads: number of heads for decoder intermediate layers
      - embedding_dim: CAN ID embedding size
      - dropout: dropout probability
      - mlp_hidden: hidden dimension for neighborhood decoder MLP (if None, uses latent_dim)
    """

    def __init__(
        self,
        num_ids,
        in_channels,
        hidden_dims=None,
        latent_dim=32,
        encoder_heads=4,
        decoder_heads=4,
        embedding_dim=8,
        dropout=0.35,
        batch_norm=True,
        mlp_hidden=None,
        use_checkpointing=False,
        conv_type="gat",
        edge_dim=None,
        proj_dim=0,
        variational=True,
    ):
        super().__init__()

        # Shared input encoding (ID embedding + optional projection)
        self.input_encoder = InputEncoder(
            num_ids=num_ids,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            conv_type=conv_type,
            edge_dim=edge_dim,
            proj_dim=proj_dim,
        )
        self.num_ids = num_ids
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._edge_dim = self.input_encoder._edge_dim
        self._proj_dim = proj_dim

        # Encoder conv stack (shared with DGI)
        gat_in_dim = self.input_encoder.out_dim
        self.gat_in_dim = gat_in_dim
        self.encoder_layers, self.encoder_bns, self.latent_in_dim = build_encoder_stack(
            hidden_dims, latent_dim, gat_in_dim, conv_type, self._edge_dim,
            encoder_heads=encoder_heads, batch_norm=batch_norm,
        )
        self.variational = variational
        self.z_mean = nn.Linear(self.latent_in_dim, latent_dim)
        if variational:
            self.z_logvar = nn.Linear(self.latent_in_dim, latent_dim)

        # Decoder: mirror of encoder, final layer outputs continuous features
        # Recompute encoder_targets (same logic as build_encoder_stack)
        if hidden_dims is not None and len(hidden_dims) >= 2 and hidden_dims[-1] == latent_dim:
            encoder_targets = hidden_dims[:-1]
        else:
            encoder_targets = hidden_dims if hidden_dims else [max(128, latent_dim * 2), latent_dim]
        decoder_targets = list(reversed(encoder_targets))
        # Replace last target with in_channels for reconstruction output
        decoder_targets[-1] = in_channels
        self.decoder_layers, self.decoder_bns = build_conv_stack(
            conv_type, latent_dim, decoder_targets, self._edge_dim,
            heads_first=decoder_heads, batch_norm=batch_norm,
        )
        # Remove the batch norm for the last decoder layer (sigmoid output, no BN)
        if batch_norm and len(self.decoder_bns) == len(decoder_targets):
            self.decoder_bns = self.decoder_bns[:-1]

        # CAN ID classifier head
        self.canid_classifier = nn.Linear(latent_dim, num_ids)

        # Neighborhood decoder MLP: use mlp_hidden if provided, else default to latent_dim for parameter efficiency
        if mlp_hidden is None:
            mlp_hidden = latent_dim  # Default to latent_dim for compact models
        self.neighborhood_decoder = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_ids),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.latent_dim = latent_dim

    def encode(self, x, edge_index, edge_attr=None, batch=None, node_id=None):
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
        if self.variational:
            logvar = self.z_logvar(x).clamp(-20, 20)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z = mu
            kl_loss = mu.new_tensor(0.0)
        return z, kl_loss

    def decode_node(self, z, edge_index, edge_attr=None, batch=None):
        assert z.size(-1) == self.latent_dim, f"Expected {self.latent_dim}D input, got {z.size(-1)}D"
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
            else:  # Last decoder layer — sigmoid constrains output to [0,1]
                x = torch.sigmoid(
                    conv_forward(
                        conv,
                        x,
                        edge_index,
                        edge_attr,
                        activation=None,
                        use_checkpointing=self.use_checkpointing,
                    )
                )
        cont_out = x  # shape: [num_nodes, in_channels]
        canid_logits = self.canid_classifier(z)

        return cont_out, canid_logits

    def create_neighborhood_targets(self, node_id, edge_index, batch):
        """Create neighborhood target matrix for training.

        Args:
            node_id: Global CAN ID indices [num_nodes].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment vector.

        Returns:
            Binary target matrix [num_nodes, num_ids].
        """
        num_nodes = node_id.size(0)
        neighbor_targets = torch.zeros(num_nodes, self.num_ids, device=node_id.device)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        dst_can_ids = node_id[dst_nodes]

        valid = (dst_can_ids >= 0) & (dst_can_ids < self.num_ids)
        neighbor_targets[src_nodes[valid], dst_can_ids[valid]] = 1.0

        return neighbor_targets

    @staticmethod
    def neighborhood_loss_negsampled(
        logits: torch.Tensor,
        node_id: torch.Tensor,
        edge_index: torch.Tensor,
        num_ids: int,
        k_neg: int = 32,
    ) -> torch.Tensor:
        """Neighborhood BCE loss with negative sampling.

        Memory: O(num_edges + num_nodes * k_neg) instead of O(num_nodes * num_ids).
        """
        src, dst = edge_index
        dst_ids = node_id[dst]
        valid = (dst_ids >= 0) & (dst_ids < num_ids)
        # Positive: logits at true neighbor IDs
        pos_logits = logits[src[valid], dst_ids[valid]]
        pos_loss = -F.logsigmoid(pos_logits).mean()
        # Negative: random IDs per node, excluding true neighbors (sparse rejection)
        neg_ids = torch.randint(0, num_ids, (logits.size(0), k_neg), device=logits.device)
        # Encode (node, id) as unique keys for sparse collision check
        pos_keys = src[valid].long() * num_ids + dst_ids[valid].long()
        node_range = torch.arange(logits.size(0), device=logits.device)
        neg_keys = (node_range.unsqueeze(1) * num_ids + neg_ids).reshape(-1)
        is_collision = torch.isin(neg_keys, pos_keys)
        neg_logits = logits.gather(1, neg_ids).reshape(-1)[~is_collision]
        neg_loss = -F.logsigmoid(-neg_logits).mean() if neg_logits.numel() > 0 else logits.new_zeros(1)
        return pos_loss + neg_loss

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> "GraphAutoencoderNeighborhood":
        """Construct from a config."""
        conv_type = cfg.vgae.conv_type
        return cls(
            num_ids=num_ids,
            in_channels=in_ch,
            hidden_dims=list(cfg.vgae.hidden_dims),
            latent_dim=cfg.vgae.latent_dim,
            encoder_heads=cfg.vgae.heads,
            embedding_dim=cfg.vgae.embedding_dim,
            dropout=cfg.vgae.dropout,
            conv_type=conv_type,
            edge_dim=resolve_edge_dim(conv_type, cfg.vgae.edge_dim),
            proj_dim=cfg.vgae.proj_dim,
            use_checkpointing=cfg.training.gradient_checkpointing,
            variational=cfg.vgae.get("variational", True),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, mask_ratio: float = 0.0, node_id=None):
        """Forward pass through the GraphAutoencoderNeighborhood.

        Args:
            x: Continuous node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment vector.
            edge_attr: Optional edge features for TransformerConv/GATv2Conv.
            mask_ratio: Fraction of features to mask during training
                (GraphMAE-style). Masked features are zeroed before encoding;
                the returned mask indicates which (node, feature) positions
                were masked for selective reconstruction loss. Set to 0.0 to
                disable (inference, or legacy behavior).
            node_id: Global CAN ID indices [num_nodes] for embedding lookup.

        Returns:
            tuple: (cont_out, canid_logits, neighbor_logits, z, kl_loss, mask).
            mask is a bool tensor [num_nodes, in_channels] or None if mask_ratio=0.
        """
        mask = None
        if mask_ratio > 0.0 and self.training:
            mask = torch.rand_like(x) < mask_ratio
            x = x.clone()
            x[mask] = 0.0

        ea = edge_attr if self._uses_edge_attr else None
        z, kl_loss = self.encode(x, edge_index, edge_attr=ea, batch=batch, node_id=node_id)
        cont_out, canid_logits = self.decode_node(z, edge_index, edge_attr=ea, batch=batch)
        neighbor_logits = self.neighborhood_decoder(z)
        return cont_out, canid_logits, neighbor_logits, z, kl_loss, mask

    @torch.no_grad()
    def score_difficulty(
        self, graphs: list, canid_weight: float = 1.0, chunk_size: int = 500,
    ) -> list[float]:
        """Score reconstruction difficulty for curriculum learning.

        Per-graph score = mean_node_MSE + canid_weight * mean_node_CE.
        Higher score = harder to reconstruct = more difficult sample.

        Args:
            graphs: List of PyG Data objects (normal-class only).
            canid_weight: Weight for CAN ID cross-entropy term.
            chunk_size: Batch size for chunked inference.

        Returns:
            List of float scores, one per graph.
        """
        from torch_geometric.data import Batch
        from torch_geometric.utils import scatter

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            scores: list[float] = []
            for start in range(0, len(graphs), chunk_size):
                chunk = graphs[start : start + chunk_size]
                batch = Batch.from_data_list([g.clone() for g in chunk]).to(device, non_blocking=True)
                edge_attr = getattr(batch, "edge_attr", None)
                cont, canid_logits, _, _, _, _ = self(
                    batch.x, batch.edge_index, batch.batch,
                    edge_attr=edge_attr, node_id=batch.node_id,
                )
                node_mse = (cont - batch.x).pow(2).mean(dim=1)
                graph_mse = scatter(node_mse, batch.batch, reduce="mean")
                node_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
                graph_ce = scatter(node_ce, batch.batch, reduce="mean")
                scores.extend((graph_mse + canid_weight * graph_ce).tolist())
                del batch
                torch.cuda.empty_cache()
            return scores
        finally:
            self.train(was_training)

    @torch.no_grad()
    def capture_artifacts(
        self, data: list, device: torch.device, *,
        embeddings: bool = True, components: bool = True, batch_size: int = 256,
    ) -> VGAEResult:
        """Capture embeddings and component-level loss decomposition.

        Uses batched forward passes with scatter reduction for per-graph losses.
        """
        from torch_geometric.utils import scatter

        from graphids.core.preprocessing.datamodule import make_graph_loader

        errors_all, labels_all, types_all = [], [], []
        embs_all = [] if embeddings else None
        comps: dict[str, list] = {"recon": [], "canid": [], "nbr": [], "kl": []} if components else {}

        was_training = self.training
        self.eval()
        try:
            for batch in make_graph_loader(data, batch_size=batch_size):
                batch = batch.to(device, non_blocking=True)
                edge_attr = getattr(batch, "edge_attr", None)
                cont, canid_logits, nbr_logits, z, kl_loss, _ = self(
                    batch.x, batch.edge_index, batch.batch,
                    edge_attr=edge_attr, node_id=batch.node_id,
                )
                node_mse = (cont - batch.x).pow(2).mean(dim=1)
                graph_mse = scatter(node_mse, batch.batch, reduce="max")
                errors_all.append(graph_mse.cpu())
                labels_all.append(batch.y.cpu())
                types_all.append(batch.attack_type.cpu() if hasattr(batch, "attack_type") and batch.attack_type is not None else torch.full_like(batch.y, -1))

                if embeddings and z is not None:
                    graph_emb = scatter(z, batch.batch, dim=0, reduce="mean")
                    embs_all.append(graph_emb.cpu().numpy())

                if components:
                    comps["recon"].append(graph_mse.cpu())
                    node_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
                    comps["canid"].append(scatter(node_ce, batch.batch, reduce="max").cpu())
                    nbr_targets = self.create_neighborhood_targets(batch.node_id, batch.edge_index, batch.batch)
                    node_nbr = F.binary_cross_entropy_with_logits(
                        nbr_logits, nbr_targets, reduction="none",
                    ).mean(dim=1)
                    comps["nbr"].append(scatter(node_nbr, batch.batch, reduce="max").cpu())
                    kl_val = kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
                    n_graphs = int(batch.batch.max().item()) + 1
                    comps["kl"].extend([kl_val] * n_graphs)
        finally:
            self.train(was_training)

        return VGAEResult(
            errors=torch.cat(errors_all).numpy(),
            labels=torch.cat(labels_all).numpy(),
            attack_types=torch.cat(types_all).numpy(),
            embeddings=np.vstack(embs_all) if embs_all else None,
            components={k: (torch.cat(v).numpy() if isinstance(v[0], torch.Tensor) else np.array(v))
                        for k, v in comps.items()} if components else None,
        )


# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class VGAEModule(OOMSkipMixin, pl.LightningModule):
    """VGAE training: reconstruct node features + CAN IDs + neighborhood.

    When teacher is provided, adds dual-signal KD loss:
      kd_loss = latent_w * MSE(project(z_s), z_t) + recon_w * MSE(recon_s, recon_t)
      total = alpha * kd_loss + (1-alpha) * task_loss
    """

    def __init__(self, cfg, teacher: nn.Module | None = None, projection: nn.Linear | None = None):
        super().__init__()
        if isinstance(cfg, dict):
            from omegaconf import OmegaConf
            cfg = OmegaConf.create(cfg)
        self.save_hyperparameters(ignore=["teacher", "projection"])
        num_ids, in_channels = cfg.num_ids, cfg.in_channels
        self.cfg = cfg
        self.model = GraphAutoencoderNeighborhood.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, dynamic=True)
        self.teacher = teacher
        self.projection = projection
        self._teacher_on_cpu = False
        self.test_threshold: float | None = None
        self.test_metrics = binary_test_metrics()
        self._test_errors: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        mask_ratio = self.cfg.vgae.mask_ratio if self.training else 0.0
        return self.model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=edge_attr, mask_ratio=mask_ratio, node_id=batch.node_id,
        )

    def _task_loss(self, batch):
        cont_out, canid_logits, nbr_logits, z, kl_loss, mask = self(batch)
        target = batch.x
        if mask is not None:
            recon = F.mse_loss(cont_out[mask], target[mask])
        else:
            recon = F.mse_loss(cont_out, target)
        canid = F.cross_entropy(canid_logits, batch.node_id)
        nbr_loss = GraphAutoencoderNeighborhood.neighborhood_loss_negsampled(
            nbr_logits, batch.node_id, batch.edge_index,
            self.cfg.num_ids, k_neg=self.cfg.vgae.k_neg,
        )
        w = self.cfg.vgae
        task_loss = recon + w.canid_weight * canid + w.nbr_weight * nbr_loss + w.kl_weight * kl_loss
        return task_loss, cont_out, z

    def _step(self, batch):
        task_loss, cont_out, z = self._task_loss(batch)
        if self.teacher is not None:
            kd = _get_kd_config(self.cfg)
            with teacher_on_device(self, batch.x.device):
                with torch.no_grad():
                    batch_idx = (
                        batch.batch if batch.batch is not None
                        else torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
                    )
                    t_edge_attr = getattr(batch, "edge_attr", None)
                    t_cont, _, _, t_z, _, _ = self.teacher(
                        batch.x, batch.edge_index, batch_idx, edge_attr=t_edge_attr, node_id=batch.node_id,
                    )
            z_s = self.projection(z) if self.projection is not None else z
            min_n = min(z_s.size(0), t_z.size(0))
            latent_kd = F.mse_loss(z_s[:min_n], t_z[:min_n])
            min_r = min(cont_out.size(0), t_cont.size(0))
            recon_kd = F.mse_loss(cont_out[:min_r], t_cont[:min_r])
            kd_loss = kd.vgae_latent_weight * latent_kd + kd.vgae_recon_weight * recon_kd
            return kd.alpha * kd_loss + (1 - kd.alpha) * task_loss
        return task_loss

    def _training_step_inner(self, batch, _idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, _idx):
        from torch_geometric.utils import scatter
        edge_attr = getattr(batch, "edge_attr", None)
        cont, canid_logits, nbr_logits, _, _, _ = self.model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_id=batch.node_id,
        )
        per_node_se = (cont - batch.x).pow(2).mean(dim=1)
        recon = scatter(per_node_se, batch.batch, dim=0, reduce="max")
        canid_err = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
        canid_per_graph = scatter(canid_err, batch.batch, dim=0, reduce="max")
        nbr_targets = self.model.create_neighborhood_targets(batch.node_id, batch.edge_index, batch.batch)
        nbr_err = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets, reduction="none").mean(dim=1)
        nbr_per_graph = scatter(nbr_err, batch.batch, dim=0, reduce="max")
        w = self.cfg.vgae
        errors = recon + w.canid_weight * canid_per_graph + w.nbr_weight * nbr_per_graph
        self._test_errors.append(errors)
        self._test_labels.append(batch.y)
        if self.test_threshold is not None:
            self.test_metrics.update(errors, batch.y)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_threshold is not None:
            self.log_dict(self.test_metrics.compute())

    def get_test_errors(self) -> tuple:
        """Return accumulated (errors, labels) as numpy arrays after test."""
        if not self._test_errors:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        return (torch.cat(self._test_errors).cpu().numpy(),
                torch.cat(self._test_labels).cpu().numpy())

    def find_threshold(self, data: list, batch_size: int = 256) -> tuple[float, float]:
        """Find optimal anomaly threshold via Youden's J on validation data.

        Runs test without a threshold to accumulate errors, then computes optimal.
        Returns (threshold, youden_j).
        """
        import pytorch_lightning as _pl
        from torchmetrics.functional.classification import binary_roc

        from graphids.core.preprocessing.datamodule import make_graph_loader

        self.test_threshold = None  # accumulate errors only
        self._test_errors.clear()
        self._test_labels.clear()

        trainer = _pl.Trainer(
            accelerator="auto", devices="auto",
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
        )
        loader = make_graph_loader(data, batch_size=batch_size)
        trainer.test(self, dataloaders=loader, verbose=False)

        errors, labels = self.get_test_errors()

        if len(errors) == 0:
            return 0.5, 0.0

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return float(np.median(errors)), 0.0

        fpr_v, tpr_v, thresholds_v = binary_roc(
            torch.as_tensor(errors, dtype=torch.float),
            torch.as_tensor(labels, dtype=torch.long),
        )
        j_scores = tpr_v - fpr_v

        if len(j_scores) == 0 or len(thresholds_v) == 0:
            return float(np.median(errors)), 0.0

        best_idx = torch.argmax(j_scores).item()
        thresh = float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(np.median(errors))
        return thresh, float(j_scores[best_idx])

    @classmethod
    def evaluate(cls, cfg, val_data, test_scenarios, device, *, load_model_fn) -> dict:
        """Evaluate VGAE: threshold search on val, test, capture artifacts."""
        from ._training import eval_with_scenarios, gpu_cleanup
        model = load_model_fn(cfg, "vgae", "autoencoder", device)
        module = cls(cfg)
        module.model = model

        bs = cfg.evaluation.batch_size
        threshold, youden_j = module.find_threshold(val_data, batch_size=bs)
        module.test_threshold = threshold

        def _clear_accumulators():
            for attr in ("_test_errors", "_test_scores", "_test_labels"):
                acc = getattr(module, attr, None)
                if acc is not None:
                    acc.clear()

        _clear_accumulators()
        module.test_metrics.reset()

        val_metrics, scenario_metrics = eval_with_scenarios(
            module, val_data, test_scenarios, bs, reset_fn=_clear_accumulators,
        )
        val_metrics["optimal_threshold"] = threshold
        val_metrics["youden_j"] = youden_j

        artifacts = model.capture_artifacts(val_data, device, batch_size=bs)
        gpu_cleanup(model)
        return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": artifacts}

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.projection is not None:
            params += list(self.projection.parameters())
        opt = torch.optim.Adam(params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        return build_optimizer_dict(opt, self.cfg)
