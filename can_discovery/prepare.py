"""Reusable preparation of unscored CAN field candidates from captures."""
from __future__ import annotations

from dataclasses import dataclass

from .decoding.signals import _verified_checksums, _verified_counters
from .decoding.tokenization import tokenize
from .preprocessing.capture import Capture
from .preprocessing.traces import MessageTrace, prepare_traces


def protocol_bits(captures: list[Capture], trace: MessageTrace) -> frozenset[int]:
    """Find observed counter and checksum bits on full, unsampled captures."""
    arb_id, length, bus = trace.arb_id, trace.payload_length, trace.bus
    counters = _verified_counters(captures, arb_id, length, bus)
    reserved = set().union(*(set(range(start, start + width))
                             for start, width in counters)) if counters else set()
    checksums = _verified_checksums(captures, arb_id, length, reserved, bus)
    return frozenset(reserved | set().union(*(
        set(range(byte * 8, byte * 8 + 8)) for byte, _ in checksums)))


@dataclass(frozen=True, slots=True)
class PreparedFamily:
    trace: MessageTrace
    protocol_bits: frozenset[int]

    def layouts(self, *, max_width: int = 64):
        """Yield unscored field layouts; caller decides how to score them."""
        result = tokenize(self.trace, method='exhaustive', max_width=max_width,
                          excluded_bits=self.protocol_bits)
        return result.candidates()


def prepare_families(captures: list[Capture], *, max_ids: int = 1_000_000,
                     samples_per_capture: int = 512) -> tuple[PreparedFamily, ...]:
    """Build independent message families without roles, DBCs, or scores."""
    traces = prepare_traces(captures, max_ids=max_ids,
                            samples_per_capture=samples_per_capture)
    return tuple(PreparedFamily(trace, protocol_bits(captures, trace))
                 for trace in traces)
