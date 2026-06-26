"""Representation config contract."""

from __future__ import annotations

import pytest

from graphids.core.data.preprocessing.representations import (
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
    representation_window_defaults,
)


def test_representation_identity_defaults_and_validation_are_stable():
    snapshot = SnapshotRepresentationCfg(window_size=11, stride=7)
    sequence = SnapshotSequenceRepresentationCfg(
        window_size=12,
        stride=6,
        sequence_length=3,
        sequence_stride=2,
    )

    assert representation_kind(snapshot) == "snapshot"
    assert representation_kind(sequence) == "snapshot_sequence"
    assert representation_payload(sequence) == {
        "kind": "snapshot_sequence",
        "window_size": 12,
        "stride": 6,
        "sequence_length": 3,
        "sequence_stride": 2,
    }
    assert representation_window_defaults(snapshot) == (11, 7)
    assert representation_window_defaults(sequence) == (12, 6)
    assert representation_digest(sequence) != representation_digest(
        SnapshotSequenceRepresentationCfg(sequence_length=4)
    )

    invalid_configs = (
        lambda: SnapshotRepresentationCfg(window_size=0),
        lambda: SnapshotRepresentationCfg(stride=0),
        lambda: SnapshotSequenceRepresentationCfg(sequence_length=0),
        lambda: SnapshotSequenceRepresentationCfg(sequence_stride=0),
    )
    for make_invalid in invalid_configs:
        with pytest.raises(ValueError, match="must be positive"):
            make_invalid()
