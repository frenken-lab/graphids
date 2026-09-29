"""Compatibility exports for the temporal hybrid model.

Implementation lives under :mod:`graphids.core.models.temporal._hybrid`; this
module remains the public import and checkpoint class path.
"""

from __future__ import annotations

from ._hybrid import (
    TemporalHybridModel,
    TemporalIdMemory,
    TemporalInputEncoder,
    TemporalMotifContext,
    TemporalRhythmContext,
    TemporalStreamBackbone,
    TemporalTimeEncoder,
)

for _exported in (
    TemporalHybridModel,
    TemporalIdMemory,
    TemporalInputEncoder,
    TemporalMotifContext,
    TemporalRhythmContext,
    TemporalStreamBackbone,
    TemporalTimeEncoder,
):
    _exported.__module__ = __name__

__all__ = [
    "TemporalHybridModel",
    "TemporalInputEncoder",
    "TemporalTimeEncoder",
    "TemporalIdMemory",
    "TemporalRhythmContext",
    "TemporalMotifContext",
    "TemporalStreamBackbone",
]
