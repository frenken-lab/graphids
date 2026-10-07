"""Adapters for existing joint physical tests; no implicit promotion to a role."""
from __future__ import annotations

from dataclasses import asdict

from ..decoding.signals import FieldCandidate
from ..preprocessing.capture import Capture
from ..relationships.physical.physics_factors import FACTOR_ROLES, evaluate_factor
from .evidence import ScoreEvidence


def physical_factor_evidence(captures: list[Capture], name: str,
                             fields: tuple[FieldCandidate, ...] | None) -> ScoreEvidence:
    """Evaluate a complete candidate group; return zero ranking weight for now."""
    if name not in FACTOR_ROLES:
        raise ValueError(f'unknown physical factor: {name}')
    hashes = tuple(c.sha256 for c in captures)
    scope = 'pair' if len(FACTOR_ROLES[name]) == 2 else 'group'
    if fields is None:
        return ScoreEvidence(name, scope, 'unavailable', reason='missing_role_candidate',
                             capture_hashes=hashes)
    result = evaluate_factor(captures, fields, name)
    return ScoreEvidence(name, scope,
        'unavailable' if result.status == 'unavailable' else 'measured',
        raw_value=result.effect if result.status != 'unavailable' else None,
        reason=result.reason, capture_hashes=hashes,
        details={'factor': asdict(result), 'ranking_weight': 0.})
