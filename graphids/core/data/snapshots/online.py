"""Bounded raw-CAN feature state for continuous online inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .batch import recording_batch
from .context import VehicleBehaviorContext
from .manifest import Recording
from .policy import ReferencePolicy
from .preprocessing import FEATURES, CANSnapshotRepresentationCfg, SnapshotRecording


@dataclass
class _Stream:
    generation: int
    last_time: float = -math.inf
    payload: bytes | None = None
    count: int = 0
    change: bool = False
    gap: float = 0.0


@dataclass
class CANFeatureState:
    streams: dict[str, _Stream] = field(default_factory=dict)
    previous_live: set[str] = field(default_factory=set)
    origin: float | None = None
    last_timestamp: float = -math.inf
    bin_index: int = 0
    next_generation: int = 0
    new_ids: int = 0
    events: int = 0
    positive_label: bool = False
    labels_known: bool = True
    version: int = 1
    recording_key: str = ""
    context_digest: str = ""


class CausalCANBinner:
    """Consume ordered events and yield closed bins without retaining history.

    Callers must consume the iterator returned by push/finish. Empty bins are
    emitted before the next event is applied. State size is bounded by anchors
    plus transient capacity, regardless of recording duration.
    """

    def __init__(
        self,
        recording: Recording,
        context: VehicleBehaviorContext | None = None,
        *,
        representation_cfg: CANSnapshotRepresentationCfg | None = None,
        transient_capacity=128,
        anchor_capacity=128,
        retention_seconds=1.0,
        state: CANFeatureState | None = None,
        reference_policy=None,
    ):
        self.reference_policy = ReferencePolicy.parse(reference_policy)
        self.recording, self.context = recording, context
        self.cfg = representation_cfg or CANSnapshotRepresentationCfg()
        self.transient_capacity, self.anchor_capacity = (
            transient_capacity,
            anchor_capacity,
        )
        self.retention_seconds = retention_seconds
        self.anchors = tuple(context.anchor_ids) if context else ()
        self.state = state or CANFeatureState()
        digest = (
            (context.digest if context else "zero-shot")
            + ":"
            + self.reference_policy.digest
        )
        if self.state.recording_key and (
            self.state.recording_key,
            self.state.context_digest,
        ) != (recording.key, digest):
            raise ValueError(
                "online feature state belongs to another recording/context"
            )
        self.state.recording_key, self.state.context_digest = recording.key, digest
        if (
            len(self.anchors) > anchor_capacity
            or transient_capacity < 1
            or retention_seconds <= 0
        ):
            raise ValueError("invalid online capacity/retention contract")
        for key in self.anchors if self.reference_policy.persistent_anchors else ():
            self.state.streams.setdefault(key, _Stream(0))

    def push(
        self, timestamp: float, arb_id: str, payload: bytes, *, label: int | None = None
    ):
        s = self.state
        if not math.isfinite(timestamp) or timestamp < s.last_timestamp:
            raise ValueError("online clock must be finite and nondecreasing")
        if len(payload) > 8 or not arb_id or label not in (None, 0, 1):
            raise ValueError("invalid classic-CAN observation")
        if self.recording.verified_normal and label == 1:
            raise ValueError("positive label in verified-normal recording")
        if s.origin is None:
            s.origin = timestamp
        offset = (timestamp - s.origin) / self.cfg.bin_seconds
        if abs(offset - round(offset)) < 1e-5:
            offset = float(round(offset))
        event_bin = math.floor(offset)
        while s.bin_index < event_bin:
            yield self._close(
                s.origin + (s.bin_index + 1) * self.cfg.bin_seconds,
                self.cfg.bin_seconds,
            )
        if arb_id not in s.streams:
            if (
                arb_id not in self.anchors
                and sum(key not in self.anchors for key in s.streams)
                >= self.transient_capacity
            ):
                raise ValueError(
                    "online transient overflow; event was not silently discarded"
                )
            s.next_generation += 1
            s.streams[arb_id] = _Stream(s.next_generation)
            s.new_ids += arb_id not in self.anchors
        stream = s.streams[arb_id]
        if math.isfinite(stream.last_time):
            stream.gap = min(10.0, timestamp - stream.last_time)
        if self.recording.payload_length_verified:
            stream.change |= stream.payload is not None and stream.payload != payload
        stream.payload, stream.last_time = payload, timestamp
        stream.count += 1
        s.events += 1
        s.positive_label |= label == 1
        s.labels_known &= self.recording.verified_normal or label is not None
        s.last_timestamp = timestamp

    def finish(self):
        s = self.state
        if s.origin is not None and s.events:
            start = s.origin + s.bin_index * self.cfg.bin_seconds
            yield self._close(s.last_timestamp, max(0.0, s.last_timestamp - start))

    def _close(self, cutoff, exposure):
        s = self.state
        expired = [
            key
            for key, stream in s.streams.items()
            if (key not in self.anchors or not self.reference_policy.persistent_anchors)
            and stream.count == 0
            and cutoff - stream.last_time > self.retention_seconds
        ]
        for key in expired:
            del s.streams[key]
        keys = tuple(s.streams)
        # Retain one masked storage column for an empty zero-shot bin; it has no state.
        storage = keys or ("",)
        x = np.zeros((1, len(storage), len(FEATURES)), dtype=np.float32)
        generation = np.full((1, len(storage)), -1, dtype=np.int64)
        live = np.zeros_like(generation, dtype=bool)
        counts = np.zeros_like(generation, dtype=np.float32)
        change = np.zeros_like(counts)
        payload_observed = np.zeros_like(live)
        for i, key in enumerate(keys):
            stream = s.streams[key]
            live[0, i] = True
            generation[0, i] = stream.generation
            counts[0, i] = stream.count
            change[0, i] = stream.change
            x[0, i, :4] = (
                stream.count,
                min(10.0, max(0.0, cutoff - stream.last_time)),
                stream.gap,
                float(stream.change),
            )
            data = stream.payload or b""
            x[0, i, 4 : 4 + len(data)] = np.frombuffer(data, dtype=np.uint8) / 255.0
            if self.recording.payload_length_verified:
                x[0, i, 12 : 12 + len(data)] = 1
                payload_observed[0, i] = bool(data) and stream.count > 0
            x[0, i, 20:] = (stream.count > 0, key not in s.previous_live, exposure)
        label = (
            1
            if s.positive_label
            else (
                0
                if self.recording.verified_normal or (s.events and s.labels_known)
                else -1
            )
        )
        start = s.origin + s.bin_index * self.cfg.bin_seconds
        if any(
            start < end and cutoff > begin
            for begin, end in self.recording.attack_intervals
        ):
            label = 1
        snapshot = SnapshotRecording(
            self.recording,
            storage,
            x,
            live,
            generation,
            np.array([label]),
            np.array([self.recording.verified_normal and label != 1]),
            counts,
            change,
            payload_observed,
            np.array([s.new_ids], dtype=np.float32),
            np.array([cutoff]),
            np.array([exposure], dtype=np.float32),
            np.array([s.events]),
        )
        batch = recording_batch(
            snapshot,
            self.context,
            anchor_capacity=self.anchor_capacity,
            transient_capacity=self.transient_capacity,
            reference_policy=self.reference_policy,
            neighbor_cap=self.context.neighbors.shape[-1] if self.context else 8,
        )
        batch.bin_indices.fill_(s.bin_index)
        s.previous_live = set(keys)
        s.bin_index += 1
        for stream in s.streams.values():
            stream.count, stream.change, stream.gap = 0, False, 0.0
        s.new_ids, s.events, s.positive_label, s.labels_known = 0, 0, False, True
        return batch
