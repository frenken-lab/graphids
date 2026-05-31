"""Representation contract tests for view/segment routing."""

from __future__ import annotations

import pytest

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
    representation_view,
    representation_window_defaults,
)
from graphids.core.data.preprocessing.segments import segment_kind
from graphids.core.data.preprocessing.views import (
    EventChunkViewCfg,
    RollingStreamViewCfg,
    SnapshotSequenceViewCfg,
    view_kind,
)


def test_representation_kinds_are_explicit():
    assert representation_kind(SnapshotRepresentationCfg()) == "snapshot"
    assert representation_kind(SnapshotSequenceRepresentationCfg()) == "snapshot_sequence"
    assert representation_kind(MultiScaleRepresentationCfg()) == "multi_scale"
    assert representation_kind(TemporalRepresentationCfg()) == "temporal"
    assert representation_kind(EntityRepresentationCfg()) == "entity"


def test_representation_view_bridge():
    assert view_kind(representation_view(SnapshotRepresentationCfg())) == "snapshot"
    assert (
        view_kind(representation_view(SnapshotSequenceRepresentationCfg()))
        == "snapshot_sequence"
    )
    assert view_kind(representation_view(MultiScaleRepresentationCfg())) == "multi_scale"
    assert view_kind(representation_view(TemporalRepresentationCfg())) == "rolling_stream"
    assert view_kind(representation_view(EntityRepresentationCfg())) == "entity"
    assert view_kind(EventChunkViewCfg()) == "event_chunk"
    assert view_kind(RollingStreamViewCfg()) == "rolling_stream"


def test_representation_segment_bridge():
    assert segment_kind(representation_segment(SnapshotRepresentationCfg())) == "window"
    assert (
        segment_kind(representation_segment(SnapshotSequenceRepresentationCfg())) == "sequence"
    )
    assert segment_kind(representation_segment(MultiScaleRepresentationCfg())) == "multi_scale"
    assert segment_kind(representation_segment(EntityRepresentationCfg())) == "entity"


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


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: SnapshotRepresentationCfg(window_size=0), "window_size must be positive"),
        (
            lambda: SnapshotSequenceRepresentationCfg(sequence_length=0),
            "sequence_length must be positive",
        ),
        (lambda: MultiScaleRepresentationCfg(window_sizes=()), "window_sizes must not be empty"),
        (lambda: TemporalRepresentationCfg(time_col=""), "time_col must not be empty"),
        (
            lambda: EntityRepresentationCfg(history_window_size=0, future_window_size=0),
            "cannot both be zero",
        ),
        (lambda: SnapshotSequenceViewCfg(sequence_stride=0), "sequence_stride must be positive"),
        (
            lambda: EventChunkViewCfg(message_count=None, duration_ms=None),
            "message_count or duration_ms is required",
        ),
        (lambda: RollingStreamViewCfg(prediction_horizon=0), "prediction_horizon must be positive"),
    ],
)
def test_representation_and_view_configs_validate_impossible_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
