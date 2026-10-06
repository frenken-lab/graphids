"""Shared field identity for graph evidence."""
from __future__ import annotations

from ...decoding.signals import FieldCandidate


def field_key(field: FieldCandidate) -> str:
    return ':'.join(map(str, (field.bus, field.arb_id, field.payload_length,
                              field.mux_start, field.mux_value, field.start,
                              field.width, field.endian, field.signed)))
