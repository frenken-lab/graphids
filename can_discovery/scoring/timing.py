"""Family-level arrival timing evidence; currently diagnostic only."""
from __future__ import annotations

import numpy as np

from ..preprocessing.capture import Capture
from .evidence import ScoreEvidence


def arrival_timing_evidence(captures: list[Capture], family: tuple,
                            *, min_gaps: int = 30) -> ScoreEvidence:
    """Measure period stability without claiming that cadence identifies a role."""
    bus, arb_id, length = family
    medians = []
    variation = []
    for capture in captures:
        times = np.asarray([frame.timestamp for frame in capture.frames
            if frame.payload is not None and len(frame.payload) == length
            and frame.arb_id.upper() == arb_id.upper()
            and (frame.bus or capture.known_bus) == bus])
        gaps = np.diff(times)
        gaps = gaps[gaps > 0]
        if len(gaps) < min_gaps:
            return ScoreEvidence('arrival_timing', 'family', 'unavailable',
                reason='insufficient_arrivals',
                capture_hashes=tuple(c.sha256 for c in captures))
        median = float(np.median(gaps))
        medians.append(median)
        variation.append(float(np.median(np.abs(gaps - median)) / median))
    return ScoreEvidence('arrival_timing', 'family', 'measured',
        raw_value=max(variation), capture_hashes=tuple(c.sha256 for c in captures),
        details={'period_seconds': medians, 'relative_median_deviation': variation,
                 'ranking_weight': 0.})
