"""Preserve the existing three-term label-completion ranking as named evidence."""
from __future__ import annotations

from dataclasses import asdict

from .evidence import ScoreEvidence, combine_score


def unary_evidence(score: float, tags: tuple[str, ...] = ()) -> ScoreEvidence:
    return ScoreEvidence('unary_payload', 'layout', 'measured', score, score,
                         details={'role_properties': list(tags)})


def rank_evidence(unary_score: float, contrast_vote: float,
                  relation_vote: float, relation_net: float | None,
                  *, assisted: bool, contrast_available: bool,
                  contrast_valid: bool = True) -> tuple[ScoreEvidence, ...]:
    return (
        unary_evidence(unary_score),
        ScoreEvidence('source_contrast', 'layout',
                      'measured' if assisted and contrast_available and contrast_valid
                      else 'unavailable',
                      contrast_vote),
        ScoreEvidence('speed_rpm_relation', 'pair',
                      'measured' if assisted and relation_net is not None else 'unavailable',
                      relation_vote, relation_net),
    )


def score_record(evidence: tuple[ScoreEvidence, ...]) -> dict:
    return {'score': combine_score(evidence),
            'evidence': [asdict(item) for item in evidence]}
