"""Representation contract tests for view/segment routing."""

from __future__ import annotations

from graphids.core.data.preprocessing.representations import (
    EntityRepresentationCfg,
    MultiScaleRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
    TemporalRepresentationCfg,
    representation_kind,
    representation_plan,
    representation_segment,
    representation_temporal_spec,
    representation_window_defaults,
    representation_view,
)


def test_representation_kinds_are_explicit():
    assert representation_kind(SnapshotRepresentationCfg()) == "snapshot"
    assert representation_kind(SnapshotSequenceRepresentationCfg()) == "snapshot_sequence"
    assert representation_kind(MultiScaleRepresentationCfg()) == "multi_scale"
    assert representation_kind(TemporalRepresentationCfg()) == "temporal"
    assert representation_kind(EntityRepresentationCfg()) == "entity"


def test_representation_view_bridge():
    assert representation_view(SnapshotRepresentationCfg()).__class__.__name__ == "SnapshotViewCfg"
    assert (
        representation_view(SnapshotSequenceRepresentationCfg()).__class__.__name__
        == "SnapshotSequenceViewCfg"
    )
    assert (
        representation_view(MultiScaleRepresentationCfg()).__class__.__name__
        == "MultiScaleViewCfg"
    )
    assert representation_view(TemporalRepresentationCfg()).__class__.__name__ == "RollingStreamViewCfg"
    assert representation_view(EntityRepresentationCfg()).__class__.__name__ == "EntityViewCfg"


def test_representation_segment_bridge():
    assert representation_segment(SnapshotRepresentationCfg()).__class__.__name__ == "WindowSegmentCfg"
    assert (
        representation_segment(SnapshotSequenceRepresentationCfg()).__class__.__name__
        == "SequenceSegmentCfg"
    )
    assert (
        representation_segment(MultiScaleRepresentationCfg()).__class__.__name__
        == "MultiScaleSegmentCfg"
    )
    assert representation_segment(EntityRepresentationCfg()).__class__.__name__ == "EntitySegmentCfg"


def test_representation_temporal_bridge():
    spec = representation_temporal_spec(TemporalRepresentationCfg())
    assert spec.__class__.__name__ == "TemporalGraphSpec"


def test_representation_plan_wraps_cfg():
    plan = representation_plan(EntityRepresentationCfg())
    assert plan.kind == "entity"
    assert isinstance(plan.cfg, EntityRepresentationCfg)


def test_representation_window_defaults_are_derived():
    assert representation_window_defaults(SnapshotRepresentationCfg(window_size=11, stride=7)) == (
        11,
        7,
    )
    assert representation_window_defaults(
        SnapshotSequenceRepresentationCfg(window_size=12, stride=6, sequence_length=3)
    ) == (12, 6)
    assert representation_window_defaults(
        MultiScaleRepresentationCfg(window_sizes=(8, 16, 32), stride=4)
    ) == (8, 4)
    assert representation_window_defaults(
        EntityRepresentationCfg(history_window_size=9, future_window_size=2)
    ) == (12, 2)
