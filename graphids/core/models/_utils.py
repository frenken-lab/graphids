"""Shared utilities for model forward passes and Lightning training."""

import contextlib
import functools
from typing import NamedTuple

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv, GPSConv, TransformerConv
from torch_geometric.nn.norm import GraphNorm

_log = structlog.get_logger()


class InputEncoder(nn.Module):
    """Shared input encoding for VGAE and GAT models.

    Encapsulates CAN ID embedding, optional feature projection, and the
    concatenation of embedding + continuous features. Both VGAE and GAT
    use identical input encoding logic.

    Attribute names (id_embedding, feat_proj) are preserved for clarity,
    though composing this as ``self.input_encoder`` changes state_dict
    key prefixes (acceptable on pre-sweep branch).
    """

    def __init__(
        self,
        num_ids: int,
        in_channels: int,
        embedding_dim: int,
        conv_type: str = "gat",
        edge_dim: int | None = None,
        proj_dim: int = 0,
    ):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.num_ids = num_ids
        self.conv_type = conv_type
        self._uses_edge_attr = conv_type in ("transformer", "gatv2", "gps")
        self._edge_dim = edge_dim if self._uses_edge_attr else None
        self._proj_dim = proj_dim

        if proj_dim > 0:
            self.feat_proj = nn.Linear(in_channels, proj_dim)
        else:
            self.feat_proj = None

        cont_dim = proj_dim if proj_dim > 0 else in_channels
        self.out_dim = embedding_dim + cont_dim

    def forward(self, x: Tensor, node_id: Tensor) -> Tensor:
        """Encode node features with CAN ID embedding.

        Args:
            x: Continuous features ``[num_nodes, in_channels]``.
            node_id: Global CAN ID indices ``[num_nodes]`` for embedding lookup.

        Returns:
            Encoded features ``[num_nodes, out_dim]``.
        """
        id_emb = self.id_embedding(node_id)
        if self.feat_proj is not None:
            x = self.feat_proj(x)
        return torch.cat([id_emb, x], dim=1)


class _ProjectedGPS(nn.Module):
    """Linear projection → GPSConv. Bridges dimension mismatch for GPS residuals."""

    def __init__(self, in_dim: int, channels: int, gps: GPSConv):
        super().__init__()
        self.proj = nn.Linear(in_dim, channels)
        self.gps = gps
        # Expose attributes that conv_forward and build_conv_stack inspect
        self.in_channels = in_dim
        self.out_channels = channels
        self.heads = 1  # output is already channels-wide

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None,
                batch: Tensor | None = None) -> Tensor:
        x = self.proj(x)
        return self.gps(x, edge_index, batch=batch, edge_attr=edge_attr)


def _make_conv(
    conv_type: str, in_dim: int, out_dim: int, heads: int, edge_dim: int | None = None, **kwargs
):
    """Factory for graph convolution layers.

    Supports: gat, gatv2, transformer, gps.
    GPS wraps a GATv2Conv with global self-attention (Rampasek et al., 2022).
    """
    if conv_type == "transformer":
        return TransformerConv(
            in_dim, out_dim, heads=heads, edge_dim=edge_dim, concat=True, **kwargs
        )
    elif conv_type == "gps":
        # GPS requires in_dim == out_dim for residual connections.
        # Inner GATv2Conv uses concat=False to preserve dimensionality.
        channels = out_dim * heads
        inner = GATv2Conv(
            channels, channels, heads=heads, concat=False, edge_dim=edge_dim,
        )
        gps = GPSConv(channels, inner, heads=heads, attn_type="multihead", dropout=0.1)
        if in_dim != channels:
            return _ProjectedGPS(in_dim, channels, gps)
        return gps
    elif conv_type == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim, concat=True, **kwargs)
    else:
        return GATConv(in_dim, out_dim, heads=heads, concat=True, **kwargs)


def build_conv_stack(
    conv_type: str,
    in_dim: int,
    target_dims: list[int],
    edge_dim: int | None,
    heads_first: int = 1,
    batch_norm: bool = True,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Build a stack of graph conv layers with optional GraphNorm.

    Resolves multi-head output dimensions: if target_dim is divisible by heads,
    uses target_dim // heads per head; otherwise falls back to heads=1.

    Returns (conv_layers, norm_layers). norm_layers may be shorter than
    conv_layers if batch_norm is False.
    """
    convs = nn.ModuleList()
    norms = nn.ModuleList()
    for i, target_dim in enumerate(target_dims):
        heads = heads_first if i == 0 else 1
        if heads > 1 and target_dim % heads == 0:
            out_per_head = target_dim // heads
        else:
            heads = 1
            out_per_head = target_dim
        convs.append(_make_conv(conv_type, in_dim, out_per_head, heads=heads, edge_dim=edge_dim))
        if batch_norm:
            norms.append(GraphNorm(out_per_head * heads))
        in_dim = out_per_head * heads
    return convs, norms


def _conv_forward_inner(
    conv, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None,
    bn: nn.Module | None, batch: Tensor | None,
    activation, dropout_p: float, training: bool,
) -> Tensor:
    """Full conv block: conv → norm → activation → dropout."""
    x = conv(x, edge_index, edge_attr) if edge_attr is not None else conv(x, edge_index)
    if bn is not None:
        x = bn(x, batch) if isinstance(bn, GraphNorm) else bn(x)
    if activation is not None:
        x = activation(x)
    if dropout_p > 0.0:
        x = torch.nn.functional.dropout(x, p=dropout_p, training=training)
    return x


def conv_forward(
    conv,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None = None,
    bn: nn.Module | None = None,
    batch: Tensor | None = None,
    activation=torch.nn.functional.relu,
    dropout_p: float = 0.0,
    training: bool = True,
    use_checkpointing: bool = False,
) -> Tensor:
    """Apply a graph conv layer with optional norm, activation, and dropout.

    When use_checkpointing is True, the entire block (conv + norm + activation +
    dropout) is wrapped in a single checkpoint segment. This saves ~30-50% of
    activation memory at the cost of recomputing the forward pass during backward.
    Uses use_reentrant=False for torch.compile compatibility.
    """
    _is_gps = isinstance(conv, (GPSConv, _ProjectedGPS))
    if _is_gps:
        # GPS has internal norm, activation, and dropout — run as-is
        return conv(x, edge_index, edge_attr=edge_attr, batch=batch)
    if use_checkpointing and x.requires_grad:
        return checkpoint(
            _conv_forward_inner,
            conv, x, edge_index, edge_attr, bn, batch, activation, dropout_p, training,
            use_reentrant=False,
        )
    return _conv_forward_inner(conv, x, edge_index, edge_attr, bn, batch, activation, dropout_p, training)


# ---------------------------------------------------------------------------
# Lightning training helpers (used by VGAEModule, GATModule, DGIModule)
# ---------------------------------------------------------------------------


class NodeBudgetInfo(NamedTuple):
    """Result of compute_node_budget: budget for DynamicBatchSampler + mean for num_steps."""
    budget: int
    mean_nodes: float


def compute_node_budget(batch_size: int, cfg) -> NodeBudgetInfo:
    """Derive max_num_nodes from batch_size * p95 graph node count."""
    import json
    from graphids.config import cache_dir

    lake_root = cfg.lake_root if hasattr(cfg, "lake_root") else cfg["lake_root"]
    dataset = cfg.dataset if hasattr(cfg, "dataset") else cfg["dataset"]
    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cache_metadata.json not found at {metadata_path}. "
            "Rebuild caches with: python -m graphids stage=preprocess dataset=..."
        )
    meta = json.loads(metadata_path.read_text())
    stats = meta["graph_stats"]["node_count"]
    budget = int(batch_size * stats["p95"])
    _log.info("node_budget_computed", batch_size=batch_size, p95_nodes=stats["p95"],
             mean_nodes=stats["mean"], budget=budget)
    return NodeBudgetInfo(budget=budget, mean_nodes=stats["mean"])


class OOMSkipMixin:
    """Skip batch on CUDA OOM. Lightning natively handles training_step returning None."""

    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning("oom_batch_skipped", batch_idx=batch_idx,
                         num_graphs=batch.num_graphs, num_nodes=batch.num_nodes)
            return None


def soft_label_kd_loss(student_logits, teacher_logits, temperature: float):
    """Hinton soft-label KD loss: KL(student/T || teacher/T) * T^2."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def focal_loss(logits, targets, gamma: float = 2.0):
    """Focal loss (Lin et al. 2017) for class-imbalanced classification."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _get_kd_config(cfg):
    """Get KD auxiliary config, or None if not configured."""
    return next((a for a in cfg.get("auxiliaries", []) if a.type == "kd"), None)


@contextlib.contextmanager
def teacher_on_device(module, device):
    """Move teacher to device for inference, offload back to CPU after."""
    if module.cfg.training.offload_teacher_to_cpu and module._teacher_on_cpu:
        module.teacher.to(device)
        module._teacher_on_cpu = False
    try:
        yield
    finally:
        if module.cfg.training.offload_teacher_to_cpu:
            module.teacher.to("cpu")
            module._teacher_on_cpu = True


def build_optimizer_dict(optimizer, cfg):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    if not cfg.training.use_scheduler or cfg.training.scheduler is None:
        return optimizer
    from hydra.utils import instantiate
    sched = instantiate(cfg.training.scheduler, optimizer=optimizer)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "monitor": cfg.training.monitor_metric}}
