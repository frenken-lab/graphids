"""One ordered, capture-local view of raw CAN frames for discovery."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Frame:
    timestamp: float
    ordinal: int
    arb_id: str
    payload: bytes | None
    attack: bool = False
    raw_payload_chars: int = 0
    bus: str | None = None


@dataclass(frozen=True, slots=True)
class Event:
    timestamp: float
    arb_id: str
    kind: str
    payload_length: int
    ordinal: int = 0
    raw_payload_chars: int | None = None

    @property
    def key(self) -> str:
        return f"{self.arb_id}|{self.kind}"


@dataclass(frozen=True, slots=True)
class Capture:
    source: str
    sha256: str
    frames: tuple[Frame, ...]
    start: float
    end: float
    decreasing_timestamps: int
    invalid_payloads: int
    payload_lengths: dict[int, int]
    raw_payload_chars: dict[int, int]
    vehicle: str | None = None
    session: str | None = None
    known_bus: str | None = None
    _events: tuple[Event, ...] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def events(self) -> tuple[Event, ...]:
        if self._events is None:
            chars = {frame.ordinal: frame.raw_payload_chars for frame in self.frames}
            events = tuple(Event(stamp, arb_id, kind, chars[ordinal] // 2,
                                 ordinal, chars[ordinal])
                           for stamp, ordinal, arb_id, kind in derive_events(self.frames))
            object.__setattr__(self, '_events', events)
        return self._events

    @property
    def rows(self) -> int:
        return len(self.frames)


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_capture(path: Path, *, drop_interval: tuple[str, float, float] | None = None,
                 metadata: dict | None = None) -> Capture:
    """Read once, validate, and sort by timestamp then source-row ordinal."""
    frames: list[Frame] = []
    previous = -math.inf
    decreasing = invalid = 0
    lengths: Counter[int] = Counter()
    chars: Counter[int] = Counter()
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'timestamp', 'arbitration_id', 'data_field'} <= set(reader.fieldnames or []):
            raise ValueError(f'missing raw CAN columns: {path}')
        for ordinal, row in enumerate(reader):
            stamp = float(row['timestamp'])
            if not math.isfinite(stamp):
                raise ValueError(f'nonfinite timestamp: {path}:{ordinal + 2}')
            decreasing += stamp < previous
            previous = stamp
            arb_id = row['arbitration_id'].strip()
            if drop_interval and arb_id == drop_interval[0] and drop_interval[1] <= stamp < drop_interval[2]:
                continue
            raw = row['data_field'].strip()
            try:
                payload = bytes.fromhex(raw) if len(raw) % 2 == 0 else None
            except ValueError:
                payload = None
            invalid += payload is None
            lengths[len(raw) // 2] += 1
            chars[len(raw)] += 1
            bus = next((row[key].strip() for key in ('bus', 'channel', 'can_bus')
                        if key in row and row[key] and row[key].strip()), None)
            frames.append(Frame(stamp, ordinal, arb_id, payload,
                                row.get('attack', '0').strip() == '1', len(raw), bus))
    frames.sort(key=lambda frame: (frame.timestamp, frame.ordinal))
    return Capture(str(path), file_hash(path), tuple(frames),
                   frames[0].timestamp if frames else 0.0,
                   frames[-1].timestamp if frames else 0.0,
                   decreasing, invalid, dict(sorted(lengths.items())), dict(sorted(chars.items())),
                   (metadata or {}).get('vehicle'), (metadata or {}).get('session'),
                   (metadata or {}).get('known_bus'))


def audit_capture(capture: Capture) -> dict:
    """Stage 0 audit; every invalid payload is identified by source row."""
    frames = capture.frames
    duplicates = sum(a.arb_id == b.arb_id and a.bus == b.bus and a.payload == b.payload
                     and a.timestamp == b.timestamp for a,b in zip(frames, frames[1:]))
    seen_payloads = set()
    repeated_payloads = 0
    for frame in frames:
        if frame.payload is None:
            continue
        key = (frame.arb_id, frame.bus, frame.payload)
        repeated_payloads += key in seen_payloads
        seen_payloads.add(key)
    return {'source': capture.source, 'sha256': capture.sha256,
            'vehicle': capture.vehicle, 'session': capture.session,
            'known_bus': capture.known_bus,
            'observed_buses': sorted({f.bus for f in frames if f.bus is not None}),
            'rows': len(frames), 'invalid_payloads': capture.invalid_payloads,
            'excluded_rows': [{'row': f.ordinal + 2,
                               'reason': 'malformed_payload' if f.payload is None else 'unsupported_length'}
                              for f in frames if f.payload is None or not 1 <= len(f.payload) <= 8],
            'decreasing_timestamps': capture.decreasing_timestamps,
            'equal_timestamps': sum(a.timestamp == b.timestamp for a,b in zip(frames, frames[1:])),
            'duplicate_frames': duplicates,
            'repeated_payload_content': repeated_payloads,
            'payload_lengths': capture.payload_lengths}


def load_manifest(path: Path) -> tuple[Capture, ...]:
    """Load explicit capture provenance without guessing from file names."""
    data = json.loads(path.read_text())
    entries = data['captures'] if isinstance(data, dict) else data
    captures = []
    for entry in entries:
        source = Path(entry['path'])
        if not source.is_absolute():
            source = path.parent / source
        capture = load_capture(source, metadata=entry)
        if entry.get('sha256') and entry['sha256'] != capture.sha256:
            raise ValueError(f'manifest hash mismatch: {source}')
        captures.append(capture)
    return tuple(captures)


def audit_captures(captures: list[Capture] | tuple[Capture, ...]) -> dict:
    reports = [audit_capture(capture) for capture in captures]
    hashes = Counter(capture.sha256 for capture in captures)
    return {'captures': reports,
            'duplicate_sources': sorted(hash for hash, count in hashes.items() if count > 1),
            'total_excluded_frames': sum(len(item['excluded_rows']) for item in reports),
            'coverage': {'vehicles': sorted({c.vehicle for c in captures if c.vehicle}),
                         'sessions': sorted({c.session for c in captures if c.session}),
                         'observed_buses': sorted({f.bus for c in captures for f in c.frames
                                                   if f.bus is not None})}}


def derive_events(frames: tuple[Frame, ...] | list[Frame]):
    """Yield arrival and changed-payload observations without crossing a capture boundary."""
    previous: dict[str, bytes | None] = {}
    for frame in frames:
        yield (frame.timestamp, frame.ordinal, frame.arb_id, 'arrival')
        if frame.arb_id in previous and frame.payload is not None and previous[frame.arb_id] is not None and frame.payload != previous[frame.arb_id]:
            yield (frame.timestamp, frame.ordinal, frame.arb_id, 'change')
        previous[frame.arb_id] = frame.payload
