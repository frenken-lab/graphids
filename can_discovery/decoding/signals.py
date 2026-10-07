"""Orchestrate structural tokenization and separate protocol evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..preprocessing.capture import Capture
from ..preprocessing.traces import prepare_traces
from .tokenization import TokenizationResult, tokenize
from .field_values import extract


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    arb_id: str
    start: int
    width: int
    endian: str
    signed: bool | None
    role: str
    limit: float
    confidence: float
    reference_support: tuple[int, ...]
    label_hypotheses: tuple[tuple[str, float], ...]
    evidence: tuple[str, ...]
    payload_length: int = 8
    mux_start: int | None = None
    mux_value: int | None = None
    bus: str | None = None


@dataclass(frozen=True, slots=True)
class UnresolvedSpan:
    arb_id: str
    payload_length: int
    start: int
    width: int
    mux_start: int | None = None
    mux_value: int | None = None


def unresolved_spans(fields: tuple[FieldCandidate, ...],
                     captures: list[Capture]) -> tuple[UnresolvedSpan, ...]:
    """Report physical bits without a trusted signal interpretation."""
    families = set.intersection(*(set((frame.arb_id, len(frame.payload))
                                   for frame in capture.frames if frame.payload is not None)
                                 for capture in captures))
    spans: list[UnresolvedSpan] = []
    for arb_id, length in sorted(families):
        if not 1 <= length <= 8:
            continue
        matching = [field for field in fields if field.arb_id == arb_id
                    and field.payload_length == length]
        modes = {(field.mux_start, field.mux_value) for field in matching}
        if not modes:
            modes = {(None, None)}
        for mux_start, mux_value in sorted(modes, key=lambda item: (-1 if item[0] is None else item[0],
                                                                     -1 if item[1] is None else item[1])):
            owned = set().union(*(_bit_positions(field.start, field.width, field.endian)
                                  for field in matching if (field.mux_start, field.mux_value)
                                  == (mux_start, mux_value)))
            start = None
            for bit in range(length * 8 + 1):
                missing = bit < length * 8 and bit not in owned
                if missing and start is None:
                    start = bit
                elif not missing and start is not None:
                    spans.append(UnresolvedSpan(arb_id, length, start, bit - start,
                                                mux_start, mux_value))
                    start = None
    return tuple(spans)


def _bit_positions(start: int, width: int, endian: str) -> frozenset[int]:
    """Map extractor coordinates to byte-major physical bit positions."""
    if endian == "big":
        return frozenset((pos // 8) * 8 + 7 - pos % 8 for pos in range(start, start + width))
    return frozenset(range(start, start + width))


def _infer_signed(values: list[list[bytes]], start: int, width: int, endian: str) -> bool | None:
    if width < 2:
        return False
    recordings = [np.asarray([extract(payload, start, width, endian, False)
                              for payload in recording], dtype=np.int64)
                  for recording in values if len(recording) > 1]
    if not recordings:
        return False
    series = np.concatenate(recordings)
    sign = 1 << (width - 1)
    if not (np.any(series & sign) and np.any(~series & sign)):
        return None
    raw_jumps = np.concatenate([np.abs(np.diff(recording)) for recording in recordings])
    signed_jumps = np.concatenate([np.abs(np.diff(np.where(recording & sign,
                                  recording - (1 << width), recording)))
                                   for recording in recordings])
    return bool(np.quantile(signed_jumps, .99) < np.quantile(raw_jumps, .99) * .5)


def _detect_mux(values: list[list[bytes]]) -> None:
    """Compatibility hook: tokenization does not infer multiplexing modes."""
    return None


def _verified_counters(captures: list[Capture], arb_id: str, length: int,
                       bus: str | None = None) -> list[tuple[int, int]]:
    """Require a modular increment on consecutive unsampled frames in both trips."""
    found = []
    for start in range(0, length * 8, 4):
        for width in (4, 8):
            if start + width > length * 8 or (width == 8 and start % 8):
                continue
            supports = []
            for capture in captures:
                frames = [frame.payload for frame in capture.frames
                          if frame.arb_id == arb_id and frame.payload is not None
                          and len(frame.payload) == length and
                          (frame.bus or capture.known_bus) == bus]
                if len(frames) < 40:
                    supports = []
                    break
                values = np.asarray([extract(payload, start, width, "little", False)
                                     for payload in frames], dtype=np.int64)
                increments = np.diff(values) % (1 << width)
                # Drops can skip values, but zero and large jumps must be rare.
                supports.append(len(increments) >= 39 and
                                np.mean(increments == 1) >= .9 and
                                np.mean((increments >= 1) & (increments <= 3)) >= .97)
            if len(supports) == len(captures) and all(supports):
                found.append((start, width))
    found.sort(key=lambda item: (-item[1], item[0]))
    selected = []
    occupied: set[int] = set()
    for start, width in found:
        bits = set(range(start, start + width))
        if not bits & occupied:
            selected.append((start, width))
            occupied.update(bits)
    return selected


def _verified_checksums(captures: list[Capture], arb_id: str, length: int,
                        reserved: set[int], bus: str | None = None) -> list[tuple[int, str]]:
    """Validate byte-wide sum or XOR checks on withheld frames of each trip."""
    series = [[frame.payload for frame in capture.frames
               if frame.arb_id == arb_id and frame.payload is not None
               and len(frame.payload) == length and
               (frame.bus or capture.known_bus) == bus] for capture in captures]
    if min(map(len, series)) < 80:
        return []
    checks = []
    for byte in range(length):
        if set(range(byte * 8, byte * 8 + 8)) & reserved:
            continue
        if any(len({payload[byte] for payload in values}) < 32 for values in series):
            continue
        for method in ("sum8", "xor8"):
            offsets = []
            valid = True
            for values in series:
                split = len(values) // 2
                def residual(payload: bytes, byte: int = byte,
                             method: str = method) -> int:
                    others = payload[:byte] + payload[byte + 1:]
                    aggregate = sum(others) if method == "sum8" else 0
                    if method == "xor8":
                        for item in others:
                            aggregate ^= item
                        return payload[byte] ^ aggregate
                    return (payload[byte] - aggregate) & 255
                train = [residual(payload) for payload in values[:split]]
                offset, count = Counter(train).most_common(1)[0]
                held_out = [residual(payload) for payload in values[split:]]
                if count / len(train) < .99 or np.mean(np.asarray(held_out) == offset) < .99:
                    valid = False
                    break
                offsets.append(offset)
            if valid and len(set(offsets)) == 1:
                checks.append((byte, method))
                break
    return checks



@dataclass(frozen=True, slots=True)
class FieldRegistry:
    candidates: tuple[FieldCandidate, ...]
    selected: tuple[FieldCandidate, ...]
    tokenizations: tuple[TokenizationResult, ...] = ()
    conflicts: tuple[dict, ...] = ()

def discover_fields(captures: list[Capture], *, max_ids: int = 64,
                    samples_per_id: int = 512, min_changes: int = 20,
                    max_fields_per_id: int = 4, max_candidates_per_id: int = 16384,
                    only_ids: tuple[str, ...] | None = None,
                    method: str = 'can_d', source_model=None,
                    max_layouts: int = 1) -> FieldRegistry:
    if method == 'exhaustive':
        raise ValueError('exhaustive tokenization yields individual layouts; '
                         'score them with a task-specific scorer')
    traces = prepare_traces(captures, max_ids=max_ids,
                            samples_per_capture=samples_per_id, only_ids=only_ids)
    results = tuple(tokenize(trace, method, source_model=source_model,
                             max_layouts=max_layouts) for trace in traces)
    fields = []
    conflicts = []
    primary_fields = set()
    for result in results:
        trace = result.trace
        arb_id, length = trace.arb_id, trace.payload_length
        candidate_spans = list(result.spans)
        if max_layouts > 1:
            candidate_spans.extend(span for alternative in result.alternatives
                                   for span in alternative.spans if span not in candidate_spans)
        for span in candidate_spans:
            if span.byte_order == 'unresolved':
                continue
            endian = span.byte_order
            bits = span.physical_bits
            start = min(bits) if endian == 'big' else min(bits)
            # Little-endian extractor coordinates count physical LSB first.
            if endian == 'little':
                mapped = {(p//8)*8 + 7-p%8 for p in bits}
                if max(mapped)-min(mapped)+1 != len(mapped):
                    continue
                start = min(mapped)
            try:
                signed = _infer_signed([list(v) for v in trace.samples], start, len(bits), endian)
            except (ValueError, OverflowError):
                signed = None
            fields.append(FieldCandidate(arb_id, start, len(bits), endian, signed,
                'unknown', 0., 1., trace.support, (('unknown',1.),),
                (f'tokenizer:{method}',), length, bus=trace.bus))
            if span in result.spans:
                primary_fields.add((arb_id, length, trace.bus, start, len(bits), endian))
        counters = _verified_counters(captures, arb_id, length, trace.bus)
        reserved = set().union(*(set(range(s,s+w)) for s,w in counters)) if counters else set()
        protocol = [(s,w,'counter','consecutive_unsampled_increment') for s,w in counters]
        protocol += [(byte*8,8,'checksum',f'{kind}_held_out_verified')
                     for byte,kind in _verified_checksums(captures,arb_id,length,reserved,trace.bus)]
        for start,width,role,evidence in protocol:
            positions = _bit_positions(start,width,'little')
            overlapping = [f for f in fields if f.arb_id == arb_id and
                           f.payload_length == length and f.bus == trace.bus and
                           positions & _bit_positions(f.start,f.width,f.endian)]
            if overlapping and not any(positions == _bit_positions(f.start,f.width,f.endian) for f in overlapping):
                conflicts.append({'arb_id':arb_id,'payload_length':length,'bus':trace.bus,
                                  'start':start,'width':width,'role':role,
                                  'reason':'protocol_tokenization_conflict'})
            fields.append(FieldCandidate(arb_id,start,width,'little',False,role,
                1. if role == 'counter' else 0.,1.,trace.support,((role,1.),),
                (evidence,),length,bus=trace.bus))
    candidates = tuple(fields)
    selected = []
    owned = {}
    for field in fields:
        if field.signed is None:
            continue
        if field.role == 'unknown' and (field.arb_id, field.payload_length, field.bus,
                                        field.start, field.width, field.endian) not in primary_fields:
            continue
        key = (field.arb_id, field.payload_length, field.bus)
        bits = _bit_positions(field.start, field.width, field.endian)
        if bits & owned.get(key, set()):
            continue
        selected.append(field)
        owned.setdefault(key, set()).update(bits)
    return FieldRegistry(candidates,tuple(selected),results,tuple(conflicts))

def fit_fields(captures: list[Capture], **kwargs) -> tuple[FieldCandidate, ...]:
    return discover_fields(captures, **kwargs).selected

def propose_fields(captures: list[Capture], *, protocol_fields=None, **kwargs) -> tuple[FieldCandidate, ...]:
    kwargs.pop('max_candidates_per_id',None)
    candidates = discover_fields(captures, **kwargs).candidates
    return candidates if protocol_fields is None else tuple(protocol_fields) + tuple(
        f for f in candidates if f.role == 'unknown')
