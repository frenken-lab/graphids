"""Versioned causal CAN snapshot representation identity."""

import hashlib
import json
from dataclasses import asdict

from graphids.core.data.snapshots.preprocessing import CANSnapshotRepresentationCfg

RepresentationCfg = CANSnapshotRepresentationCfg


def representation_kind(cfg: RepresentationCfg) -> str:
    return cfg.kind


def representation_payload(cfg: RepresentationCfg) -> dict[str, object]:
    return asdict(cfg)


def representation_digest(cfg: RepresentationCfg) -> str:
    payload = json.dumps(
        representation_payload(cfg), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
