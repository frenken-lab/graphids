"""Compatibility import for normal-profile tracking."""

from .behavior.tracker import (
    BehaviorFeatures,
    BehaviorScores,
    BehaviorState,
    BehaviorTracker,
    score_capture,
)

__all__ = [
    'BehaviorFeatures', 'BehaviorScores', 'BehaviorState', 'BehaviorTracker',
    'score_capture',
]
