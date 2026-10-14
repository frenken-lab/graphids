"""Recording-local causal features, including empty physical-time bins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .manifest import Recording, read_rows

FEATURES = (
    "count",
    "age",
    "gap",
    "change",
    *tuple(f"byte_{i}" for i in range(8)),
    *tuple(f"byte_valid_{i}" for i in range(8)),
    "observed",
    "new_id",
    "exposure",
)


@dataclass(frozen=True)
class CANSnapshotRepresentationCfg:
    kind: Literal["can_snapshot"] = "can_snapshot"
    bin_seconds: float = 0.01
    schema_version: int = 1

    def __post_init__(self):
        if (
            self.kind != "can_snapshot"
            or self.schema_version != 1
            or not np.isfinite(self.bin_seconds)
            or self.bin_seconds <= 0
        ):
            raise ValueError(
                "requires positive bin_seconds and can_snapshot schema_version=1"
            )


@dataclass
class SnapshotRecording:
    recording: Recording
    ids: tuple[str, ...]
    features: np.ndarray  # [T,N,F]
    live: np.ndarray  # [T,N], physical presence of stream state
    generation: np.ndarray  # [T,N], -1 outside a live segment
    labels: np.ndarray  # [T], -1 when unknown
    normal: np.ndarray  # [T], independently verified normal
    counts: np.ndarray
    change: np.ndarray
    payload_observed: np.ndarray
    new_ids: np.ndarray
    closes: np.ndarray
    exposure: np.ndarray
    event_count: np.ndarray


def build_snapshots(
    rec: Recording,
    cfg: CANSnapshotRepresentationCfg,
    *,
    anchor_ids: tuple[str, ...] = (),
    retention_seconds: float = 1.0,
    persistent_anchors: bool = True,
    max_snapshot_bytes: int = 4 * 1024**3,
) -> SnapshotRecording:
    if retention_seconds <= 0:
        raise ValueError("retention_seconds must be positive")
    rows = read_rows(rec)
    origin, end = rows[0]["time"], rows[-1]["time"]
    width = cfg.bin_seconds
    # A message exactly at a boundary belongs to the next half-open bin.
    times = np.array([r["time"] for r in rows])
    offsets = (times - origin) / width
    if np.any(offsets > np.iinfo(np.int64).max - 1):
        raise ValueError(
            f"{rec.key}: clock span cannot be represented as snapshot bin indices"
        )
    near = np.rint(offsets)
    offsets = np.where(np.abs(offsets - near) < 1e-5, near, offsets)
    bins = np.floor(offsets).astype(np.int64)
    total = int(bins.max()) + 1
    closes = origin + (np.arange(total) + 1) * width
    exposure = np.minimum(
        width, np.maximum(0, end - (origin + np.arange(total) * width))
    )
    closes[-1] = end
    ids = tuple(dict.fromkeys((*anchor_ids, *(r["id"] for r in rows))))
    index = {key: i for i, key in enumerate(ids)}
    n = len(ids)
    estimated_bytes = total * n * (len(FEATURES) * 4 + 1 + 8 + 4 + 4 + 1) + total * 33
    if estimated_bytes > max_snapshot_bytes:
        raise ValueError(
            f"recording snapshot cache budget exceeded for {rec.key}: {estimated_bytes} bytes > {max_snapshot_bytes}; increase max_snapshot_bytes after memory preflight or use a source-validated bin width"
        )
    features = np.zeros((total, n, len(FEATURES)), dtype=np.float32)
    counts = np.zeros((total, n), dtype=np.float32)
    changes = np.zeros_like(counts)
    payload_observed = np.zeros((total, n), dtype=bool)
    live = np.zeros((total, n), dtype=bool)
    generation = np.full((total, n), -1, dtype=np.int64)
    labels = np.full(total, -1, dtype=np.int64)
    normal = np.full(total, rec.verified_normal, dtype=bool)
    new_ids = np.zeros(total, dtype=np.float32)
    event_count = np.bincount(bins, minlength=total)
    last_time = np.full(n, -np.inf)
    payload = np.zeros((n, 8), dtype=np.float32)
    validity = np.zeros((n, 8), dtype=np.float32)
    previous: dict[int, bytes] = {}
    born = np.zeros(n, dtype=bool)
    gens = np.zeros(n, dtype=np.int64)
    row_index = 0
    for t in range(total):
        was_live = live[t - 1].copy() if t else np.zeros(n, dtype=bool)
        gaps = np.zeros(n, dtype=np.float32)
        bin_labels = []
        restarted = set()
        while row_index < len(rows) and bins[row_index] == t:
            row = rows[row_index]
            i = index[row["id"]]
            if not born[i]:
                new_ids[t] += i >= len(anchor_ids)
                born[i] = True
            # Expiry is decided at each closed cutoff, matching streaming.
            if (
                (i >= len(anchor_ids) or not persistent_anchors)
                and t
                and not was_live[i]
                and np.isfinite(last_time[i])
                and i not in restarted
            ):
                gens[i] += 1
                new_ids[t] += i >= len(anchor_ids)
                restarted.add(i)
                previous.pop(i, None)
                last_time[i] = -np.inf
                payload[i] = 0
                validity[i] = 0
            if np.isfinite(last_time[i]):
                gaps[i] = min(10.0, max(0.0, row["time"] - last_time[i]))
            data = row["data"]
            changes[t, i] = max(
                changes[t, i],
                (
                    float(i in previous and previous[i] != data)
                    if rec.payload_length_verified
                    else 0.0
                ),
            )
            previous[i] = data
            payload[i] = 0
            payload[i, : len(data)] = np.frombuffer(data, dtype=np.uint8) / 255.0
            validity[i] = 0
            if rec.payload_length_verified:
                validity[i, : len(data)] = 1
                payload_observed[t, i] = bool(data)
            last_time[i] = row["time"]
            counts[t, i] += 1
            bin_labels.append(row["label"])
            row_index += 1
        active = born & ((closes[t] - last_time) <= retention_seconds)
        if persistent_anchors:
            active[: len(anchor_ids)] = True
        # A just-observed ID remains eligible even in a long partial bin.
        active |= counts[t] > 0
        live[t] = active
        generation[t, active] = gens[active]
        age = np.minimum(10.0, np.maximum(0.0, closes[t] - last_time))
        features[t, :, 0] = counts[t]
        features[t, :, 1] = age
        features[t, :, 2] = gaps
        features[t, :, 3] = changes[t]
        features[t, :, 4:12] = payload
        features[t, :, 12:20] = validity
        features[t, :, 20] = counts[t] > 0
        features[t, :, 21] = active & ~was_live
        features[t, :, 22] = exposure[t]
        features[t, ~active] = 0
        if any(x == 1 for x in bin_labels):
            labels[t] = 1
        elif rec.verified_normal or (bin_labels and all(x == 0 for x in bin_labels)):
            labels[t] = 0
        # Intervals independently label silence; unknown empty bins stay unknown.
        start = origin + t * width
        if any(
            start < stop and closes[t] > begin for begin, stop in rec.attack_intervals
        ):
            labels[t] = 1
            normal[t] = False
        if labels[t] == 1:
            normal[t] = False
    return SnapshotRecording(
        rec,
        ids,
        features,
        live,
        generation,
        labels,
        normal,
        counts,
        changes,
        payload_observed,
        new_ids,
        closes,
        exposure,
        event_count,
    )
