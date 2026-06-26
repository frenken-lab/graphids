"""Representation identity reaches CAN public helpers."""

from __future__ import annotations

from graphids.core.data.preprocessing.representations import (
    SnapshotSequenceRepresentationCfg,
)
from graphids.primitives_data import CANBusCfg, can_bus


def _catalog():
    return {"dummy": {}}


def test_representation_defaults_and_cache_key_reach_can_helpers(monkeypatch):
    monkeypatch.setattr("graphids.primitives_data.load_catalog", _catalog)
    representation = SnapshotSequenceRepresentationCfg(
        window_size=12,
        stride=4,
        sequence_length=3,
    )
    helper_cfg = can_bus(
        dataset="dummy",
        seed=7,
        representation_cfg=representation,
    )
    cfg = CANBusCfg(
        name="dummy",
        seed=7,
        representation_cfg=representation,
    )
    from graphids.core.data.datasets.can_bus import CANBusSource

    a = CANBusSource(
        name="dummy",
        lake_root="/tmp/graphids-test",
        representation_cfg=SnapshotSequenceRepresentationCfg(
            window_size=12,
            stride=4,
            sequence_length=2,
        ),
    )
    b = CANBusSource(
        name="dummy",
        lake_root="/tmp/graphids-test",
        representation_cfg=SnapshotSequenceRepresentationCfg(
            window_size=12,
            stride=4,
            sequence_length=4,
        ),
    )
    assert helper_cfg.window_size == 12
    assert helper_cfg.stride == 4
    assert cfg.resolved_window_size_stride() == (12, 4)
    assert a.cache_key != b.cache_key
