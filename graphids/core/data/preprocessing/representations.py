"""Data representation configs used by preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class TemporalRepresentationCfg:
    kind: Literal["temporal"] = "temporal"
    features: Literal["raw", "behavioral", "behavioral_timing", "behavioral_payload"] = "raw"


RepresentationCfg = TemporalRepresentationCfg


def representation_kind(cfg: RepresentationCfg) -> str:
    return cfg.kind


def representation_payload(cfg: RepresentationCfg) -> dict[str, object]:
    payload = asdict(cfg)
    # Preserve the existing raw cache identity; behavioral caches get a new digest.
    if payload.get("features") == "raw":
        payload.pop("features")
    return payload


def representation_digest(cfg: RepresentationCfg) -> str:
    payload = json.dumps(representation_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
