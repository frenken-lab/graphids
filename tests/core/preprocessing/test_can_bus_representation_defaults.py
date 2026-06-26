"""Representation identity reaches CAN public helpers."""

from __future__ import annotations

from graphids.core.data.preprocessing.representations import (
    SnapshotSequenceRepresentationCfg,
    representation_kind,
)
from graphids.primitives_data import CANBusCfg, can_bus, temporal_dm


def _catalog():
    return {"dummy": {}}


def test_temporal_is_default_for_can_helpers(monkeypatch):
    monkeypatch.setattr("graphids.primitives_data.load_catalog", _catalog)
    helper_cfg = can_bus(dataset="dummy", seed=7)
    cfg = CANBusCfg(name="dummy", seed=7)

    assert representation_kind(helper_cfg.representation_cfg) == "temporal"
    assert representation_kind(cfg.representation_cfg) == "temporal"
    dm_cfg = temporal_dm(
        source=helper_cfg,
        batch_size=64,
        val_warmup_events=2,
        test_warmup_events=3,
    )
    assert dm_cfg.type == "temporal_dm"
    assert dm_cfg.batch_size == 64
    assert dm_cfg.val_warmup_events == 2
    assert dm_cfg.test_warmup_events == 3


def test_legacy_snapshot_representation_and_cache_key_reach_graph_helpers(monkeypatch):
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
