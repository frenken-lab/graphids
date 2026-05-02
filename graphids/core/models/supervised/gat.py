"""GAT supervised classifier — collapsed arch + trainer-bridge.

The single :class:`GAT` class is both the architecture (InputEncoder +
conv stack + JK + pool + FC head) and the trainer-bridge
(``training_step``/``validation_step``/``test_step``, fusion-feature
extractor). No wrapper module — see
``~/plans/graphids-collapse-model-modules.md`` Phase 3.

Supports GATConv (default), GATv2Conv, and TransformerConv via conv_type.
TransformerConv natively uses edge_attr, enabling the 11-D edge features
(frequency, temporal intervals, bidirectionality, degree products) that
GATConv ignores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge, global_mean_pool
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.utils import add_self_loops, remove_self_loops

from graphids.config.constants import ModelType

from .._conv import InputEncoder, _make_conv, conv_forward, resolve_edge_dim
from ..base import GraphModuleBase, classification_test_metrics


class GAT(GraphModuleBase):
    """Collapsed GAT — arch + trainer-bridge in one ``nn.Module``.

    Loss selection is decoupled: ``loss_fn`` is an ``nn.Module``
    instantiated by :func:`graphids.orchestrate._instantiate` from the
    rendered_config's ``model.init_args.loss_fn`` class_path block. When
    the block resolves to a
    :class:`~graphids.core.losses.distillation.SoftLabelDistillation`,
    training automatically becomes a KD run — no branching here.
    """

    def __init__(
        self,
        *,
        loss_fn: nn.Module,
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
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        id_encoder_class_path: str = "graphids.core.models.id_encoding.LookupIdEncoder",
        id_encoder_kwargs: dict | None = None,
        # --- training ---
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # --- identity / dynamic ---
        scale: str = "small",
        model_type: ModelType = "gat",
        dataset: str = "",
        seed: int = 42,
        variational: bool = True,  # upstream VGAE type — identity key for supervised
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.test_metrics = classification_test_metrics(num_classes)
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._init_post(locals())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        hp = self.hparams
        id_encoder = self._build_id_encoder()
        edge_dim = resolve_edge_dim(hp.conv_type, hp.edge_dim)
        pool_aggrs = list(hp.pool_aggrs) if hp.pool_aggrs else ["mean"]

        self.input_encoder = InputEncoder(
            id_encoder=id_encoder,
            in_channels=hp.in_channels,
            conv_type=hp.conv_type,
            edge_dim=edge_dim,
            proj_dim=hp.proj_dim,
        )
        self.dropout = hp.dropout
        self.use_checkpointing = hp.gradient_checkpointing
        self.conv_type = hp.conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._proj_dim = hp.proj_dim

        # GATv2Conv runs remove_self_loops + add_self_loops inside its forward,
        # which creates edge-count-shaped intermediates inside the checkpoint
        # region. Pre-add self-loops once here and disable inside each layer
        # so the checkpoint region sees a stable edge_index.
        self._prepend_self_loops = hp.conv_type == "gatv2"
        conv_kwargs: dict = {"add_self_loops": False} if self._prepend_self_loops else {}

        self.convs = nn.ModuleList()
        for i in range(hp.layers):
            in_dim = self.input_encoder.out_dim if i == 0 else hp.hidden * hp.heads
            self.convs.append(
                _make_conv(
                    hp.conv_type,
                    in_dim,
                    hp.hidden,
                    heads=hp.heads,
                    edge_dim=edge_dim if self._uses_edge_attr else None,
                    **conv_kwargs,
                )
            )

        self.jk = JumpingKnowledge(mode="lstm", channels=hp.hidden * hp.heads, num_layers=hp.layers)

        # Pooling — "lstm" JK outputs single-layer dim (not concatenated)
        jk_out_dim = hp.hidden * hp.heads
        if len(pool_aggrs) > 1:
            self.pool = MultiAggregation(pool_aggrs)
            fc_input_dim = jk_out_dim * len(pool_aggrs)
        else:
            self.pool = None
            fc_input_dim = jk_out_dim

        self.fc_layers = nn.ModuleList()
        for _ in range(hp.fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=hp.dropout))
        self.fc_layers.append(nn.Linear(fc_input_dim, hp.num_classes))

        if hp.compile_model:
            from ..base import try_compile

            try_compile(self, conv_type=hp.conv_type, dynamic=True)

    @staticmethod
    def _rebuild_excluded_kwargs(hp: dict) -> dict:
        """Rebuild ``loss_fn`` from saved ``loss_config`` (loss_fn isn't pickleable)."""
        from graphids.core.losses.build import build_loss

        loss_cfg = hp.get("loss_config")
        return {"loss_fn": build_loss("gat", loss_cfg, distillation_config=None)}

    # ------------------------------------------------------------------
    # Architecture primitives
    # ------------------------------------------------------------------

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

        if self._prepend_self_loops:
            num_nodes = x.size(0)
            if edge_attr is not None:
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value="mean", num_nodes=num_nodes
                )
            else:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        attention_weights = [] if return_attention_weights else None

        xs = []
        for conv in self.convs:
            if return_attention_weights and self.conv_type == "gat":
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

    # ------------------------------------------------------------------
    # Trainer-bridge hooks
    # ------------------------------------------------------------------

    def _step(self, batch):
        logits = self(batch)
        loss = self.loss_fn(logits, batch.y, graph=batch)
        acc = (logits.argmax(1) == batch.y).float().mean()
        return loss, acc

    def _training_step_inner(self, batch, _idx):
        loss, acc = self._step(batch)
        bs = batch.num_graphs
        self.log("train_loss", loss, batch_size=bs)
        self.log("train_acc", acc, batch_size=bs)

        log_fn = getattr(self.loss_fn, "log_components", None)
        if log_fn is not None:
            log_fn(self, batch_size=bs, prefix="train_")
        return loss

    def validation_step(self, batch, _idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch.y, graph=batch)
        probs = F.softmax(logits, dim=1)
        acc = (probs.argmax(1) == batch.y).float().mean()
        bs = batch.num_graphs
        self.log("val_loss", loss, batch_size=bs)
        self.log("val_acc", acc, batch_size=bs)
        self._val_probs.append(probs[:, 1].detach().cpu())
        self._val_labels.append(batch.y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        from torchmetrics.functional.classification import binary_auroc

        probs = torch.cat(self._val_probs)
        labels = torch.cat(self._val_labels)
        self.log("val_auroc", binary_auroc(probs, labels))
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch, _idx, dataloader_idx=0):
        logits = self(batch)
        probs = F.softmax(logits, dim=1)
        self._record_test_batch(
            dataloader_idx,
            preds=probs.argmax(1),
            scores=probs,  # (N, K) — consumed by classification_test_metrics
            labels=batch.y,
        )

    def predict_step(self, batch, _idx):
        logits = self(batch)
        scores = F.softmax(logits, dim=1)[:, 1]
        return {"preds": logits.argmax(1), "scores": scores, "labels": batch.y}

    def extract_features(self, batch, device: torch.device) -> dict[str, torch.Tensor]:
        """Per-graph fusion features as named tensors.

        - ``probs``     [N, 2] — prob_0, prob_1
        - ``conf``      [N, 1] — 1 - entropy / log(2)
        - ``emb_stats`` [N, 4] — emb_mean, emb_std, emb_max, emb_min
        """
        logits, emb = self(batch, return_embedding=True)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        conf = (1.0 - entropy / math.log(2)).clamp(0.0, 1.0)
        return {
            "probs": probs,
            "conf": conf.unsqueeze(-1),
            "emb_stats": torch.cat(
                [
                    emb.mean(1, keepdim=True),
                    emb.std(1, keepdim=True),
                    emb.max(1).values.unsqueeze(1),
                    emb.min(1).values.unsqueeze(1),
                ],
                dim=1,
            ),
        }
