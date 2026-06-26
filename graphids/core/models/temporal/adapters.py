"""Adapters from PyG ``TemporalData`` event batches to static graph tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter


@dataclass(frozen=True)
class TemporalStaticGraph:
    graph: Data
    src: torch.Tensor
    dst: torch.Tensor
    labels: torch.Tensor
    scored_mask: torch.Tensor
    attack_type: torch.Tensor | None
    event_id: torch.Tensor | None


def is_temporal_batch(batch) -> bool:
    return hasattr(batch, "msg") and hasattr(batch, "src") and hasattr(batch, "dst")


def temporal_to_static_graph(batch) -> TemporalStaticGraph:
    """Build an event-induced static graph from one ``TemporalData`` batch.

    Node features are the mean message touching each node in the batch. Edges are
    the temporal events themselves, preserving event order in ``edge_index``.
    """
    src = batch.src.long()
    dst = batch.dst.long()
    msg = batch.msg.float()
    if src.numel() == 0:
        raise ValueError("temporal batch has no events")

    endpoint_ids = torch.cat([src, dst], dim=0)
    node_id, inverse = torch.unique(endpoint_ids, sorted=True, return_inverse=True)
    local_src = inverse[: src.numel()]
    local_dst = inverse[src.numel() :]
    num_nodes = int(node_id.numel())
    endpoint_index = torch.cat([local_src, local_dst], dim=0)
    endpoint_msg = torch.cat([msg, msg], dim=0)
    x = scatter(endpoint_msg, endpoint_index, dim=0, dim_size=num_nodes, reduce="mean")

    graph = Data(
        x=x,
        edge_index=torch.stack([local_src, local_dst], dim=0),
        batch=torch.zeros(num_nodes, dtype=torch.long, device=msg.device),
        node_id=node_id,
        y=batch.y.long(),
    )
    attack_type = getattr(batch, "attack_type", None)
    event_id = getattr(batch, "event_id", None)
    scored_mask = getattr(batch, "is_scored", None)
    if scored_mask is None:
        scored_mask = torch.ones_like(batch.y, dtype=torch.bool)
    return TemporalStaticGraph(
        graph=graph,
        src=local_src,
        dst=local_dst,
        labels=batch.y.long(),
        scored_mask=scored_mask.bool(),
        attack_type=attack_type.long() if attack_type is not None else None,
        event_id=event_id.long() if event_id is not None else None,
    )
