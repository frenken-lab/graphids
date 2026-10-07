"""Named scoring contributions with explicit availability and scope."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ScoreEvidence:
    name: str
    scope: str
    status: str
    contribution: float = 0.
    raw_value: float | None = None
    reason: str = ''
    capture_hashes: tuple[str, ...] = ()
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in ('family', 'layout', 'pair', 'group'):
            raise ValueError(f'unknown score scope: {self.scope}')
        if self.status not in ('measured', 'unavailable'):
            raise ValueError(f'unknown score status: {self.status}')
        if not math.isfinite(self.contribution):
            raise ValueError('score contribution must be finite')
        if self.status == 'unavailable' and self.contribution != 0:
            raise ValueError('unavailable evidence cannot change a score')
        if self.raw_value is not None and not math.isfinite(self.raw_value):
            raise ValueError('raw evidence value must be finite')


def combine_score(evidence: tuple[ScoreEvidence, ...]) -> float:
    """Combine already calibrated terms once, with no implicit missing-data penalty."""
    names = [item.name for item in evidence]
    if len(names) != len(set(names)):
        raise ValueError('duplicate scoring evidence')
    return sum(item.contribution for item in evidence)
