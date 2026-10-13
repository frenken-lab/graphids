"""Tensor contract and explicit masks for live stream generations."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .context import VehicleBehaviorContext
from .preprocessing import SnapshotRecording


@dataclass
class CANSnapshotBatch:
    x: torch.Tensor  # [B,T,N,F]
    live: torch.Tensor
    generation: torch.Tensor
    profiles: torch.Tensor  # [B,N,C]
    in_reference: torch.Tensor
    context_present: torch.Tensor
    neighbors: torch.Tensor  # [B,N,K]
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    labels: torch.Tensor
    scored: torch.Tensor
    normal: torch.Tensor
    counts: torch.Tensor
    payload_change: torch.Tensor
    payload_observed: torch.Tensor
    new_ids: torch.Tensor
    exposure: torch.Tensor
    closes: torch.Tensor
    event_count: torch.Tensor
    recording_keys: tuple[str, ...]
    dataset_keys: tuple[str, ...]
    context_digests: tuple[str, ...]
    node_keys: tuple[tuple[str, ...], ...]
    bin_indices: torch.Tensor

    def to(self, device, **kwargs):
        return type(self)(
            **{
                f.name: (
                    getattr(self, f.name).to(device, **kwargs)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )


def recording_batch(
    snapshot: SnapshotRecording,
    context: VehicleBehaviorContext | None,
    *,
    start: int = 0,
    stop: int | None = None,
    warmup: int = 0,
    neighbor_cap: int = 8,
    anchor_capacity: int = 128,
    transient_capacity: int = 128,
) -> CANSnapshotBatch:
    stop = len(snapshot.closes) if stop is None else stop
    if start < 0 or stop > len(snapshot.closes) or stop <= start or warmup < 0:
        raise ValueError("invalid snapshot window")
    s = slice(start, stop)
    # Pack only nodes that live somewhere in this window, while retaining all anchors.
    anchors = set(context.anchor_ids) if context else set()
    selected = [
        i
        for i, key in enumerate(snapshot.ids)
        if key in anchors or snapshot.live[s, i].any()
    ]
    keys = tuple(snapshot.ids[i] for i in selected)
    n = len(keys)
    if len(anchors) > anchor_capacity:
        raise ValueError(f"anchor overflow: {len(anchors)} > {anchor_capacity}")
    local = {key: i for i, key in enumerate(keys)}
    is_anchor = torch.tensor([key in anchors for key in keys], dtype=torch.bool)
    live = torch.from_numpy(snapshot.live[s][:, selected].copy())
    transient_load = live[:, ~is_anchor].sum(1)
    if transient_load.numel() and int(transient_load.max()) > transient_capacity:
        raise ValueError(
            f"transient overflow: {int(transient_load.max())} > {transient_capacity}; scored events were not truncated"
        )
    profiles = torch.zeros(n, 4)
    neighbors = torch.zeros(n, neighbor_cap, dtype=torch.long)
    edges = torch.zeros(n, neighbor_cap, 2)
    edge_mask = torch.zeros(n, neighbor_cap, dtype=torch.bool)
    x = torch.from_numpy(snapshot.features[s][:, selected].copy())
    if context:
        x = (x - torch.from_numpy(context.center.copy())) / torch.from_numpy(
            context.scale.copy()
        )
        if context.neighbors.shape[1] != neighbor_cap:
            raise ValueError("context and model neighbor capacity differ")
        for ci, key in enumerate(context.anchor_ids):
            i = local[key]
            profiles[i] = torch.from_numpy(context.profiles[ci].copy())
            for slot in range(neighbor_cap):
                if context.edge_mask[ci, slot]:
                    neighbors[i, slot] = local[
                        context.anchor_ids[context.neighbors[ci, slot]]
                    ]
                    edges[i, slot] = torch.from_numpy(context.edges[ci, slot].copy())
                    edge_mask[i, slot] = True
    x[~live] = 0
    scored = torch.arange(stop - start) >= warmup
    # Classification covers terminal observations even when the recording ends
    # exactly at a bin boundary. Forecast likelihoods separately require exposure.

    def node(values):
        return torch.from_numpy(values[s][:, selected].copy()).unsqueeze(0)

    def time(values):
        return torch.from_numpy(values[s].copy()).unsqueeze(0)

    return CANSnapshotBatch(
        x.unsqueeze(0),
        live.unsqueeze(0),
        node(snapshot.generation),
        profiles.unsqueeze(0),
        is_anchor.unsqueeze(0),
        torch.tensor([context is not None]),
        neighbors.unsqueeze(0),
        edges.unsqueeze(0),
        edge_mask.unsqueeze(0),
        time(snapshot.labels),
        scored.unsqueeze(0),
        time(snapshot.normal),
        node(snapshot.counts),
        node(snapshot.change),
        node(snapshot.payload_observed),
        time(snapshot.new_ids),
        time(snapshot.exposure),
        time(snapshot.closes),
        time(snapshot.event_count),
        (snapshot.recording.key,),
        (snapshot.recording.dataset,),
        (context.digest if context else "zero-shot",),
        (keys,),
        torch.arange(start, stop).unsqueeze(0),
    )


def collate_snapshots(items: list[CANSnapshotBatch]) -> CANSnapshotBatch:
    """Pad geometry explicitly; temporal execution omits invalid live segments."""
    if not items:
        raise ValueError("empty snapshot batch")
    time = max(item.x.shape[1] for item in items)
    nodes = max(item.x.shape[2] for item in items)
    result = {}
    node_fields = {
        "x",
        "live",
        "generation",
        "counts",
        "payload_change",
        "payload_observed",
    }
    static_fields = {
        "profiles",
        "in_reference",
        "neighbors",
        "edge_features",
        "edge_mask",
    }
    for f in fields(CANSnapshotBatch):
        values = [getattr(item, f.name) for item in items]
        if not isinstance(values[0], torch.Tensor):
            result[f.name] = tuple(v for group in values for v in group)
        elif f.name == "context_present":
            result[f.name] = torch.cat(values)
        else:
            shape = list(values[0].shape)
            if f.name in node_fields:
                shape[1:3] = [time, nodes]
            elif f.name in static_fields:
                shape[1] = nodes
            else:
                shape[1] = time
            shape[0] = len(items)
            fill = -1 if f.name in ("generation", "labels", "bin_indices") else 0
            out = torch.full(shape, fill, dtype=values[0].dtype)
            for i, value in enumerate(values):
                slices = (
                    slice(i, i + 1),
                    *(slice(0, size) for size in value.shape[1:]),
                )
                out[slices] = value
            result[f.name] = out
    return CANSnapshotBatch(**result)
