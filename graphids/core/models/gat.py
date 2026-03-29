from __future__ import annotations

import functools
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    JumpingKnowledge,
    global_mean_pool,
)
from torch_geometric.nn.aggr import MultiAggregation

from ._conv import InputEncoder, _make_conv, conv_forward, resolve_edge_dim
from ._training import (
    KDAuxiliary,
    OOMSkipMixin, soft_label_kd_loss, focal_loss,
    teacher_on_device, binary_test_metrics,
)


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
        pool_aggrs = pool_aggrs or ("mean",)
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
            hidden_channels=cfg.hidden,
            out_channels=cfg.num_classes,
            num_layers=cfg.layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
            num_fc_layers=cfg.fc_layers,
            embedding_dim=cfg.embedding_dim,
            conv_type=cfg.conv_type,
            edge_dim=resolve_edge_dim(cfg.conv_type, cfg.edge_dim),
            pool_aggrs=cfg.pool_aggrs,
            proj_dim=cfg.proj_dim,
            use_checkpointing=cfg.gradient_checkpointing,
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

# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class GATModule(OOMSkipMixin, pl.LightningModule):
    """GAT supervised classification (normal vs attack).

    When teacher is provided, adds soft-label KD:
      kd_loss = KL_div(student_logits/T, teacher_logits/T) * T^2
      total = alpha * kd_loss + (1-alpha) * task_loss
    """

    def __init__(
        self,
        # --- architecture ---
        hidden: int = 48,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.2,
        fc_layers: int = 3,
        embedding_dim: int = 16,
        conv_type: str = "gatv2",
        edge_dim: int = 11,
        pool_aggrs: list[str] | None = None,
        proj_dim: int = 0,
        # --- training ---
        lr: float = 0.003,
        weight_decay: float = 0.0001,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        loss_fn: str = "ce",
        focal_gamma: float = 2.0,
        loss_weight: float = 10.0,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: str = "gat",
        lake_root: str = "experimentruns",
        dataset: str = "",
        seed: int = 42,
        gat_stage: str = "curriculum",
        variational: bool = True,  # upstream VGAE type — identity key for curriculum
        auxiliaries: list[KDAuxiliary] | None = None,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        super().__init__()
        if pool_aggrs is None:
            pool_aggrs = ["mean"]
        if auxiliaries is None:
            auxiliaries = []
        self.save_hyperparameters()
        self.model = None
        self.teacher = None
        self._teacher_on_cpu = False
        self.test_metrics = binary_test_metrics()
        if loss_fn == "weighted_ce":
            w = torch.tensor([1.0, loss_weight])
            self.loss_fn = nn.CrossEntropyLoss(weight=w)
        elif loss_fn == "focal":
            self.loss_fn = functools.partial(focal_loss, gamma=focal_gamma)
        else:
            self.loss_fn = F.cross_entropy
        if num_ids > 0:
            self._build()

    def setup(self, stage=None):
        if self.model is None:
            dm = self.trainer.datamodule
            self.hparams.num_ids = dm.num_ids
            self.hparams.in_channels = dm.in_channels
            self.hparams.num_classes = dm.num_classes
            self._build()

    def _build(self):
        from ._training import prepare_kd
        hp = self.hparams
        self.model = GATWithJK.from_config(hp, hp.num_ids, hp.in_channels)
        if hp.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, dynamic=True)
        if self.teacher is None:
            self.teacher, _ = prepare_kd(hp, hp.model_type, torch.device("cpu"))

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = self.loss_fn(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()
        if self.teacher is not None:
            kd = next(a for a in getattr(self.hparams, "auxiliaries", []) if a.type == "kd")
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

    def predict_step(self, batch, _idx):
        logits = self(batch)
        scores = F.softmax(logits, dim=1)[:, 1]
        return {"preds": logits.argmax(1), "scores": scores, "labels": batch.y}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
