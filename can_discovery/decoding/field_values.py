"""Shared payload sampling and bit-value extraction for field pilots."""
from __future__ import annotations

from collections import defaultdict


def sample_by_id(rows, cap: int):
    picked = defaultdict(list)
    for _stamp, _ordinal, arb_id, payload, _attack in rows:
        if payload is not None and len(picked[arb_id]) < cap:
            picked[arb_id].append(payload)
    return picked


def extract(payload: bytes, start: int, width: int, endian: str, signed: bool):
    if endian not in ('big', 'little') or signed is None:
        raise ValueError('numeric extraction requires resolved byte order and signedness')
    total = len(payload) * 8
    if start + width > total:
        return None
    value = int.from_bytes(payload, endian)
    shift = total - start - width if endian == 'big' else start
    value = (value >> shift) & ((1 << width) - 1)
    if signed and value & (1 << (width - 1)):
        value -= 1 << width
    return value
