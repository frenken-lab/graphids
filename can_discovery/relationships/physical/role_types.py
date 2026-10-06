"""Physical role assignment data types."""
from __future__ import annotations

from dataclasses import dataclass

from ...decoding.signals import FieldCandidate
from ..shared.evidence import FactorEvidence

ROLES = ("speed", "rpm", "gear", "acceleration", "steering", "brake", "throttle",
         "yaw", "lateral_acceleration")
RUNTIME_ROLES = frozenset(("speed", "rpm", "gear", "acceleration", "steering",
                           "brake", "throttle"))


@dataclass(frozen=True, slots=True)
class RoleHypothesis:
    candidate: FieldCandidate
    role: str
    unary_score: float
    support: tuple[int, ...]
    evidence: tuple[str, ...]
    equivalence_group: int = -1
    update_limit: float = 0.0
    allowed_values: tuple[int, ...] = ()
    calibration: str = "physics_only"


@dataclass(frozen=True, slots=True)
class CandidateCombination:
    """A mutually compatible role assignment; a missing role is the null choice."""
    assignments: tuple[int | None, ...]
    weight: float
    score: float
    relation_evidence: tuple[tuple[str, float], ...]
    factors: tuple[FactorEvidence, ...] = ()
    runtime_weight: float | None = None


@dataclass(frozen=True, slots=True)
class PhysicalDiscovery:
    hypotheses: tuple[RoleHypothesis, ...]
    combinations: tuple[CandidateCombination, ...]
    protocol_fields: tuple[FieldCandidate, ...]
    rejected: tuple[dict, ...]
    factor_evidence: tuple[FactorEvidence, ...] = ()
