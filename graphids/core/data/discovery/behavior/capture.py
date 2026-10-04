"""One ordered, capture-local view of raw CAN frames for discovery."""
from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Frame:
    timestamp: float
    ordinal: int
    arb_id: str
    payload: bytes | None
    attack: bool = False
    raw_payload_chars: int = 0


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


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_capture(path: Path, *, drop_interval: tuple[str, float, float] | None = None) -> Capture:
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
            frames.append(Frame(stamp, ordinal, arb_id, payload,
                                row.get('attack', '0').strip() == '1', len(raw)))
    frames.sort(key=lambda frame: (frame.timestamp, frame.ordinal))
    return Capture(str(path), file_hash(path), tuple(frames),
                   frames[0].timestamp if frames else 0.0,
                   frames[-1].timestamp if frames else 0.0,
                   decreasing, invalid, dict(sorted(lengths.items())), dict(sorted(chars.items())))


def derive_events(frames: tuple[Frame, ...] | list[Frame]):
    """Yield arrival and changed-payload observations without crossing a capture boundary."""
    previous: dict[str, bytes | None] = {}
    for frame in frames:
        yield (frame.timestamp, frame.ordinal, frame.arb_id, 'arrival')
        if frame.arb_id in previous and frame.payload is not None and previous[frame.arb_id] is not None and frame.payload != previous[frame.arb_id]:
            yield (frame.timestamp, frame.ordinal, frame.arb_id, 'change')
        previous[frame.arb_id] = frame.payload
