"""Explicit, immutable controls for frozen-weight reference interventions."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReferencePolicy:
    profiles: bool = True
    normalization: bool = True
    graph_edges: bool = True
    persistent_anchors: bool = True
    reference_flags: bool = True
    recent_classifier_pooling: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("reference policy controls must be booleans")

    @classmethod
    def parse(cls, value=None):
        return value if isinstance(value, cls) else cls(**(value or {}))

    @property
    def digest(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


INTERVENTIONS = {
    "zero-shot": ReferencePolicy(),
    "full-reference": ReferencePolicy(),
    "no-profiles": ReferencePolicy(profiles=False),
    "no-graph-edges": ReferencePolicy(graph_edges=False),
    "no-normalization": ReferencePolicy(normalization=False),
    "no-reference-flags": ReferencePolicy(reference_flags=False),
    "recent-classifier-pooling": ReferencePolicy(recent_classifier_pooling=True),
    "no-persistent-anchors": ReferencePolicy(persistent_anchors=False),
}
