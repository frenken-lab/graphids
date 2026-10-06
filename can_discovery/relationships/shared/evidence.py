"""Evidence records shared by physical factor and model evaluators."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FactorEvidence:
    name: str
    roles: tuple[str, ...]
    status: str
    effect: float = 0.0
    control_effect: float = 0.0
    lag_bins: int = 0
    direction: int = 0
    valid_samples: int = 0
    reason: str = ''
    residual_quality: float = 0.0
    members: tuple[int, ...] = ()
