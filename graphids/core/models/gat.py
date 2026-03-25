import functools
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.nn import (
    JumpingKnowledge,
    global_mean_pool,
)
from torch_geometric.nn.aggr import MultiAggregation

from ._conv import InputEncoder, _make_conv, conv_forward, resolve_edge_dim
from ._training import (
    OOMSkipMixin, soft_label_kd_loss, focal_loss, _get_kd_config,
    teacher_on_device, build_optimizer_dict, binary_test_metrics,
)


@dataclass(frozen=True)
class GATResult:
    """Artifacts from GAT evaluation: predictions, embeddings, attention."""
    preds: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    attack_types: np.ndarray
    embeddings: np.ndarray | None = None
    attention: list[dict] | None = None


class GATWithJK(nn.Module):
    """Graph Attention Network with Jumping Knowledge connections.

    Supports GATConv (default), GATv2Conv, and TransformerConv via conv_type.
    TransformerConv natively uses edge_attr, enabling the 11-D edge features
    (frequency, temporal intervals, bidirectionality, degree products) that
    GATConv ignores.
    """

    def __init__(
        self,
        num_ids,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        heads=4,
        dropout=0.2,
        num_fc_layers=3,
        embedding_dim=8,
        use_checkpointing=False,
        conv_type="gat",
        edge_dim=None,
        pool_aggrs=("mean",),
        proj_dim=0,
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
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._proj_dim = proj_dim

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.input_encoder.out_dim if i == 0 else hidden_channels * heads
            self.convs.append(
                _make_conv(
                    conv_type,
                    in_dim,
                    hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim if self._uses_edge_attr else None,
                )
            )

        self.jk = JumpingKnowledge(
            mode="lstm", channels=hidden_channels * heads, num_layers=num_layers
        )

        # Pooling — "lstm" JK outputs single-layer dim (not concatenated)
        jk_out_dim = hidden_channels * heads
        if len(pool_aggrs) > 1:
            self.pool = MultiAggregation(list(pool_aggrs))
            fc_input_dim = jk_out_dim * len(pool_aggrs)
        else:
            self.pool = None  # use global_mean_pool
            fc_input_dim = jk_out_dim

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=dropout))
        self.fc_layers.append(nn.Linear(fc_input_dim, out_channels))

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> "GATWithJK":
        """Construct from a config."""
        return cls(
            num_ids=num_ids,
            in_channels=in_ch,
            hidden_channels=cfg.gat.hidden,
            out_channels=cfg.num_classes,
            num_layers=cfg.gat.layers,
            heads=cfg.gat.heads,
            dropout=cfg.gat.dropout,
            num_fc_layers=cfg.gat.fc_layers,
            embedding_dim=cfg.gat.embedding_dim,
            conv_type=cfg.gat.conv_type,
            edge_dim=resolve_edge_dim(cfg.gat.conv_type, cfg.gat.edge_dim),
            pool_aggrs=cfg.gat.pool_aggrs,
            proj_dim=cfg.gat.proj_dim,
            use_checkpointing=cfg.training.gradient_checkpointing,
        )

    def _pool(self, x, batch):
        if self.pool is not None:
            return self.pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(
        self,
        data,
        return_intermediate=False,
        return_attention_weights=False,
        return_embedding=False,
    ):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None) if self._uses_edge_attr else None
        node_id = data.node_id

        x = self.input_encoder(x, node_id)

        attention_weights = [] if return_attention_weights else None

        xs = []
        for conv in self.convs:
            if return_attention_weights and self.conv_type == "gat":
                # Attention weight extraction requires direct conv call
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                attention_weights.append(alpha.detach().cpu())
            else:
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    batch=batch,
                    dropout_p=self.dropout,
                    training=self.training,
                    use_checkpointing=self.use_checkpointing,
                )
            xs.append(x)
        if return_attention_weights:
            return xs, attention_weights
        if return_intermediate:
            return xs
        x = self.jk(xs)
        x = self._pool(x, batch)
        if return_embedding:
            emb = x.clone()
            for layer in self.fc_layers:
                x = layer(x)
            return x, emb
        for layer in self.fc_layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def capture_artifacts(
        self, data: list, device: torch.device, *,
        embeddings: bool = True, attention: bool = True,
        batch_size: int = 256, attention_limit: int = 50,
    ) -> GATResult:
        """Capture predictions, embeddings, and attention weights for paper artifacts."""
        from torch_geometric.loader import DataLoader as PyGDataLoader

        preds_all, scores_all, labels_all, types_all, embs_all = [], [], [], [], []
        for g in PyGDataLoader(data, batch_size=batch_size, shuffle=False):
            g = g.to(device, non_blocking=True)
            logits, emb = self(g, return_embedding=True)
            preds_all.append(logits.argmax(1).cpu())
            scores_all.append(F.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(g.y.cpu())
            at = (g.attack_type.cpu()
                  if hasattr(g, "attack_type") and g.attack_type is not None
                  else torch.full((g.num_graphs,), -1))
            types_all.append(at)
            if embeddings:
                embs_all.append(emb.cpu())

        attn_data = None
        if attention:
            attn_data = []
            for idx in range(min(len(data), attention_limit)):
                g = data[idx].clone().to(device, non_blocking=True)
                _, att_weights = self(g, return_attention_weights=True)
                attn_data.append({
                    "graph_idx": idx,
                    "label": g.y.item() if g.y.dim() == 0 else int(g.y[0].item()),
                    "edge_index": g.edge_index.cpu().numpy(),
                    "node_features": g.node_id.cpu().numpy(),
                    "attention_weights": [a.numpy() for a in att_weights],
                })

        return GATResult(
            preds=torch.cat(preds_all).numpy(),
            labels=torch.cat(labels_all).numpy(),
            scores=torch.cat(scores_all).numpy(),
            attack_types=torch.cat(types_all).numpy(),
            embeddings=torch.cat(embs_all).numpy() if embs_all else None,
            attention=attn_data,
        )


# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class GATModule(OOMSkipMixin, pl.LightningModule):
    """GAT supervised classification (normal vs attack).

    When teacher is provided, adds soft-label KD:
      kd_loss = KL_div(student_logits/T, teacher_logits/T) * T^2
      total = alpha * kd_loss + (1-alpha) * task_loss
    """

    def __init__(self, cfg, num_classes: int = 2, teacher: nn.Module | None = None):
        super().__init__()
        num_ids, in_channels = cfg.num_ids, cfg.in_channels
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg), "num_ids": num_ids, "in_channels": in_channels})
        self.cfg = cfg
        self.model = GATWithJK.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, dynamic=True)
        self.teacher = teacher
        self._teacher_on_cpu = False
        self.test_metrics = binary_test_metrics()
        loss_name = cfg.training.loss_fn
        if loss_name == "weighted_ce":
            w = torch.tensor([1.0, cfg.training.loss_weight])
            self.loss_fn = nn.CrossEntropyLoss(weight=w)
        elif loss_name == "focal":
            self.loss_fn = functools.partial(focal_loss, gamma=cfg.training.focal_gamma)
        else:
            self.loss_fn = F.cross_entropy

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = self.loss_fn(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()
        if self.teacher is not None:
            kd = _get_kd_config(self.cfg)
            with teacher_on_device(self, batch.x.device):
                with torch.no_grad():
                    t_logits = self.teacher(batch)
            kd_loss = soft_label_kd_loss(logits, t_logits, kd.temperature)
            loss = kd.alpha * kd_loss + (1 - kd.alpha) * task_loss
        else:
            loss = task_loss
        return loss, acc

    def _training_step_inner(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_acc", acc, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, _idx):
        logits = self(batch)
        scores = F.softmax(logits, dim=1)[:, 1]
        self.test_metrics.update(scores, batch.y)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        return build_optimizer_dict(opt, self.cfg)
