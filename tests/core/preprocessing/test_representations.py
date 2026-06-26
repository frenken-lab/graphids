"""Representation config contract tests."""

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


def test_representation_kinds_are_explicit():
    assert representation_kind(SnapshotRepresentationCfg()) == "snapshot"
    assert representation_kind(SnapshotSequenceRepresentationCfg()) == "snapshot_sequence"


def test_representation_payload_is_stable_mapping():
    cfg = SnapshotSequenceRepresentationCfg(
        window_size=12,
        stride=6,
        sequence_length=3,
        sequence_stride=2,
    )

    assert representation_payload(cfg) == {
        "kind": "snapshot_sequence",
        "window_size": 12,
        "stride": 6,
        "sequence_length": 3,
        "sequence_stride": 2,
    }


def test_representation_digest_changes_with_full_config():
    short = SnapshotSequenceRepresentationCfg(sequence_length=2)
    long = SnapshotSequenceRepresentationCfg(sequence_length=4)

    assert representation_digest(short) != representation_digest(long)


def test_representation_window_defaults_are_derived():
    assert representation_window_defaults(SnapshotRepresentationCfg(window_size=11, stride=7)) == (
        11,
        7,
    )
    assert representation_window_defaults(
        SnapshotSequenceRepresentationCfg(window_size=12, stride=6, sequence_length=3)
    ) == (12, 6)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: SnapshotRepresentationCfg(window_size=0), "window_size must be positive"),
        (lambda: SnapshotRepresentationCfg(stride=0), "stride must be positive"),
        (
            lambda: SnapshotSequenceRepresentationCfg(sequence_length=0),
            "sequence_length must be positive",
        ),
        (
            lambda: SnapshotSequenceRepresentationCfg(sequence_stride=0),
            "sequence_stride must be positive",
        ),
    ],
)
def test_representation_configs_validate_impossible_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
