"""Construct per-family traces from validated CAN captures."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import Capture
from ..decoding.message_types import classify_messages


@dataclass(frozen=True, slots=True)
class MessageTrace:
    arb_id: str
    payload_length: int
    bus: str | None
    samples: tuple[tuple[bytes, ...], ...]
    source_hashes: tuple[str, ...]

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(map(len, self.samples))



def prepare_traces(captures: list[Capture], *, max_ids: int = 64,
                   samples_per_capture: int = 1000,
                   only_ids: tuple[str, ...] | None = None) -> tuple[MessageTrace, ...]:
    if len(captures) < 2 or len({c.sha256 for c in captures}) != len(captures):
        raise ValueError('independent reference captures are required')
    kinds = {m.arb_id: m.kind for m in classify_messages(captures)}
    grouped = []
    for capture in captures:
        mapping = {}
        for frame in capture.frames:
            if frame.payload is None or not 1 <= len(frame.payload) <= 8:
                continue
            key = (frame.arb_id, len(frame.payload), frame.bus or capture.known_bus)
            mapping.setdefault(key, []).append(frame.payload)
        grouped.append(mapping)
    common = set.intersection(*(set(g) for g in grouped))
    if only_ids is not None:
        ids = {x.upper().zfill(3) for x in only_ids}
        common = {key for key in common if key[0].upper().zfill(3) in ids}
    common = {key for key in common if kinds.get(key[0]) not in ('iso_tp', 'j1939_transport')}
    ranked = sorted(common, key=lambda key: (-sum(len(g[key]) for g in grouped), str(key)))[:max_ids]
    out = []
    for arb_id, length, bus in ranked:
        samples = []
        for group in grouped:
            values = group[(arb_id, length, bus)]
            if len(values) > samples_per_capture:
                values = [values[int(i)] for i in np.linspace(0, len(values)-1, samples_per_capture, dtype=int)]
            samples.append(tuple(values))
        out.append(MessageTrace(arb_id, length, bus, tuple(samples), tuple(c.sha256 for c in captures)))
    return tuple(out)
