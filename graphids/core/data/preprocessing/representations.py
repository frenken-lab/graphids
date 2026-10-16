"""Versioned causal CAN event representation identity."""

import hashlib
import json
from dataclasses import asdict

from pydantic import TypeAdapter

from graphids.core.data.events.representation import CANEventRepresentationCfg

RepresentationCfg = CANEventRepresentationCfg


def parse_representation(value):
    if isinstance(value, dict) and value.get("kind") == "can_snapshot":
        raise ValueError("can_snapshot retired; restore archives/snapshot-2026-10-16.tar.gz; use can_event")
    return TypeAdapter(RepresentationCfg).validate_python(value)


def representation_kind(cfg: RepresentationCfg) -> str:
    return cfg.kind


def representation_payload(cfg: RepresentationCfg) -> dict[str, object]:
    return asdict(cfg)


def representation_digest(cfg: RepresentationCfg) -> str:
    payload = json.dumps(
        representation_payload(cfg), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
