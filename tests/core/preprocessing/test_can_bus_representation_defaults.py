"""CAN helper defaults derive from explicit representation configs."""

from __future__ import annotations

from graphids.core.data.preprocessing.representations import (
    EntityRepresentationCfg,
    MultiScaleRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
)
from graphids.primitives_data import CANBusCfg, can_bus


def _catalog():
    return {"dummy": {}}


def test_can_bus_defaults_follow_snapshot_representation(monkeypatch):
    monkeypatch.setattr("graphids.primitives_data.load_catalog", _catalog)
    cfg = can_bus(dataset="dummy", seed=7, representation_cfg=SnapshotRepresentationCfg(window_size=11, stride=3))
    assert cfg.window_size == 11
    assert cfg.stride == 3


def test_can_bus_defaults_follow_sequence_representation(monkeypatch):
    monkeypatch.setattr("graphids.primitives_data.load_catalog", _catalog)
    cfg = can_bus(
        dataset="dummy",
        seed=7,
        representation_cfg=SnapshotSequenceRepresentationCfg(window_size=12, stride=4, sequence_length=3),
    )
    assert cfg.window_size == 12
    assert cfg.stride == 4


def test_can_bus_defaults_follow_multiscale_representation(monkeypatch):
    monkeypatch.setattr("graphids.primitives_data.load_catalog", _catalog)
    cfg = can_bus(
        dataset="dummy",
        seed=7,
        representation_cfg=MultiScaleRepresentationCfg(window_sizes=(16, 32, 64), stride=8),
    )
    assert cfg.window_size == 16
    assert cfg.stride == 8


def test_can_bus_defaults_follow_entity_representation(monkeypatch):
    monkeypatch.setattr("graphids.primitives_data.load_catalog", _catalog)
    cfg = can_bus(
        dataset="dummy",
        seed=7,
        representation_cfg=EntityRepresentationCfg(history_window_size=9, future_window_size=2),
    )
    assert cfg.window_size == 12
    assert cfg.stride == 2


def test_can_bus_cfg_resolves_representation_defaults():
    cfg = CANBusCfg(
        name="dummy",
        seed=7,
        representation_cfg=SnapshotSequenceRepresentationCfg(
            window_size=12,
            stride=4,
            sequence_length=3,
        ),
    )
    assert cfg.resolved_window_size_stride() == (12, 4)
