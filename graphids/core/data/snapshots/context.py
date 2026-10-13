"""Frozen reference artifacts; weights belong to the shared model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .preprocessing import FEATURES, SnapshotRecording

CONTEXT_WIDTH = 4


@dataclass(frozen=True)
class VehicleBehaviorContext:
    vehicle: str
    anchor_ids: tuple[str, ...]
    profiles: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    neighbors: np.ndarray
    edges: np.ndarray
    edge_mask: np.ndarray
    reference_keys: tuple[str, ...]
    reference_hashes: tuple[str, ...]
    digest: str
    version: int = 1

    def __post_init__(self):
        for name in ("profiles", "center", "scale", "neighbors", "edges", "edge_mask"):
            getattr(self, name).flags.writeable = False


def build_context(
    vehicle: str,
    references: list[SnapshotRecording],
    *,
    neighbor_cap: int = 8,
    max_lag: int = 5,
) -> VehicleBehaviorContext:
    if not references or neighbor_cap < 1 or max_lag < 0:
        raise ValueError(
            "normal references, positive neighbor_cap and nonnegative max_lag required"
        )
    if any(
        not ref.recording.verified_normal
        or ref.recording.vehicle != vehicle
        or not ref.recording.role.endswith("reference")
        for ref in references
    ):
        raise ValueError("context must use verified-normal recordings of one vehicle")
    anchors = tuple(sorted(set(key for ref in references for key in ref.ids)))
    n = len(anchors)
    profiles = np.zeros((n, CONTEXT_WIDTH), dtype=np.float32)
    centers, scales = [], []
    observations: list[np.ndarray] = []
    for ref in references:
        x = np.zeros((len(ref.closes), n, 2), dtype=np.float32)
        lookup = {key: i for i, key in enumerate(ref.ids)}
        for i, key in enumerate(anchors):
            if key in lookup:
                j = lookup[key]
                x[:, i, 0] = ref.counts[:, j] / np.maximum(ref.exposure, 1e-6)
                x[:, i, 1] = ref.change[:, j]
        observations.append(x)
        centers.append(np.median(ref.features, axis=(0, 1)))
        scales.append(np.std(ref.features, axis=(0, 1)))
    all_x = np.concatenate(observations)
    profiles[:, 0] = np.log1p(all_x[:, :, 0].mean(0))
    profiles[:, 1] = np.log1p(all_x[:, :, 0].std(0))
    profiles[:, 2] = all_x[:, :, 1].mean(0)
    profiles[:, 3] = (all_x[:, :, 0] > 0).mean(0)
    center = np.mean(centers, 0).astype(np.float32)
    scale = np.maximum(np.mean(scales, 0), 1.0).astype(np.float32)
    # Leave bytes, validity and boolean features in their canonical units.
    center[4:] = 0
    scale[4:] = 1
    scores = np.zeros((n, n), dtype=np.float64)
    lags = np.zeros((n, n), dtype=np.float32)
    for x in observations:
        for lag in range(max_lag + 1):
            if len(x) - lag < 3:
                continue
            a = x[: len(x) - lag].copy()
            b = x[lag:].copy()
            a -= a.mean(0)
            b -= b.mean(0)
            for channel in range(2):
                numerator = a[:, :, channel].T @ b[:, :, channel]
                denominator = np.sqrt(
                    (a[:, :, channel] ** 2).sum(0)[:, None]
                    * (b[:, :, channel] ** 2).sum(0)[None, :]
                )
                corr = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 1e-12,
                )
                better = np.abs(corr) > np.abs(scores)
                scores[better] = corr[better]
                lags[better] = lag
    neighbors = np.zeros((n, neighbor_cap), dtype=np.int64)
    edges = np.zeros((n, neighbor_cap, 2), dtype=np.float32)
    mask = np.zeros((n, neighbor_cap), dtype=bool)
    for receiver in range(n):
        candidates = [
            j for j in range(n) if j != receiver and abs(scores[j, receiver]) > 1e-6
        ]
        candidates.sort(key=lambda j: (-abs(scores[j, receiver]), anchors[j]))
        for slot, sender in enumerate(candidates[:neighbor_cap]):
            neighbors[receiver, slot] = sender
            edges[receiver, slot] = (scores[sender, receiver], lags[sender, receiver])
            mask[receiver, slot] = True
    provenance = {
        "vehicle": vehicle,
        "keys": [r.recording.key for r in references],
        "hashes": [r.recording.sha256 for r in references],
        "anchors": anchors,
        "k": neighbor_cap,
        "lag": max_lag,
        "features": FEATURES,
        "profiles": profiles.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "neighbors": neighbors.tolist(),
        "edges": edges.tolist(),
        "mask": mask.tolist(),
    }
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return VehicleBehaviorContext(
        vehicle,
        anchors,
        profiles,
        center,
        scale,
        neighbors,
        edges,
        mask,
        tuple(provenance["keys"]),
        tuple(provenance["hashes"]),
        digest,
    )
