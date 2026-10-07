"""Conservative CAN message-family classification for field discovery."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..preprocessing.capture import Capture


@dataclass(frozen=True, slots=True)
class MessageType:
    arb_id: str
    kind: str
    pgn: int | None = None


def _j1939_pgn(arb_id: str) -> int | None:
    try:
        value = int(arb_id, 16)
    except ValueError:
        return None
    if not 0x800 <= value <= 0x1FFFFFFF:
        return None
    page = (value >> 24) & 1
    fmt = (value >> 16) & 0xFF
    specific = (value >> 8) & 0xFF if fmt >= 240 else 0
    return (page << 16) | (fmt << 8) | specific


def _is_isotp_sequence(payloads: list[bytes]) -> bool:
    """Require a first frame and ordered consecutive frames in one capture."""
    matches = 0
    for index, payload in enumerate(payloads):
        if len(payload) < 3 or payload[0] >> 4 != 1:
            continue
        size = ((payload[0] & 15) << 8) | payload[1]
        if size <= len(payload) - 2:
            continue
        remaining = size - (len(payload) - 2)
        expected = 1
        for following in payloads[index + 1:index + 65]:
            if not following or following[0] >> 4 != 2:
                if following and following[0] >> 4 == 3:
                    continue  # flow control may share an identifier in a capture
                break
            if following[0] & 15 != expected % 16:
                break
            remaining -= len(following) - 1
            expected += 1
            if remaining <= 0:
                matches += 1
                break
    return matches >= 2


def classify_messages(captures: list[Capture]) -> tuple[MessageType, ...]:
    """Identify transport traffic with evidence in every reference capture."""
    per_capture = []
    for capture in captures:
        grouped: dict[str, list[bytes]] = defaultdict(list)
        for frame in capture.frames:
            if frame.payload is not None:
                grouped[frame.arb_id].append(frame.payload)
        per_capture.append(grouped)
    if not per_capture:
        return ()
    common = set.intersection(*(set(group) for group in per_capture))
    result = []
    for arb_id in sorted(common):
        pgn = _j1939_pgn(arb_id)
        if pgn in (0xEC00, 0xEB00):
            kind = "j1939_transport"
        elif all(_is_isotp_sequence(group[arb_id]) for group in per_capture):
            kind = "iso_tp"
        elif pgn is not None:
            kind = "j1939_pgn"
        else:
            kind = "ordinary"
        result.append(MessageType(arb_id, kind, pgn))
    return tuple(result)
