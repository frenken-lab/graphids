"""Canonical entity registry and feature-frame tests."""

from __future__ import annotations

import polars as pl


def test_registry_resolves_vehicle_specific_aliases():
    from graphids.core.data.discovery.canonical import (
        CanonicalEntitySpec,
        CanonicalRegistry,
    )

    registry = CanonicalRegistry(
        entities=(
            CanonicalEntitySpec(
                canonical_id="engine_speed",
                name="engine_speed",
                aliases=("rpm",),
                vehicle_aliases={"truck_a": ("eng_rpm",)},
            ),
            CanonicalEntitySpec(
                canonical_id="vehicle_speed",
                name="vehicle_speed",
                aliases=("speed",),
            ),
        )
    )
    assert registry.resolve("rpm").canonical_id == "engine_speed"
    assert registry.resolve("eng_rpm", vehicle="truck_a").canonical_id == "engine_speed"
    assert registry.resolve("speed").canonical_id == "vehicle_speed"


def test_build_canonical_feature_frame_from_wide_rows():
    from graphids.core.data.discovery.canonical import (
        CanonicalEntitySpec,
        CanonicalFeatureFrameSpec,
        CanonicalRegistry,
        build_canonical_feature_frame,
    )

    registry = CanonicalRegistry(
        entities=(
            CanonicalEntitySpec(
                canonical_id="engine_speed",
                name="engine_speed",
                aliases=("rpm",),
                vehicle_aliases={"truck_a": ("eng_rpm",)},
            ),
            CanonicalEntitySpec(
                canonical_id="vehicle_speed",
                name="vehicle_speed",
                aliases=("speed",),
            ),
        )
    )
    df = pl.DataFrame(
        {
            "timestamp": [1.0, 2.0],
            "vehicle_id": ["truck_a", "truck_b"],
            "rpm": [1000.0, 1100.0],
            "speed": [30.0, 35.0],
        }
    )
    out = build_canonical_feature_frame(
        df,
        registry,
        spec=CanonicalFeatureFrameSpec(
            feature_cols=("rpm", "speed"),
            alias_col="signal",
            value_col="value",
            keep_cols=("timestamp",),
            vehicle_col="vehicle_id",
        ),
    )
    assert set(out.columns) >= {
        "timestamp",
        "vehicle_id",
        "signal",
        "canonical_id",
        "canonical_name",
        "kind",
        "value",
    }
    assert out.filter(pl.col("signal") == "rpm")["canonical_id"].to_list() == [
        "engine_speed",
        "engine_speed",
    ]
    assert out.filter(pl.col("signal") == "speed")["canonical_id"].to_list() == [
        "vehicle_speed",
        "vehicle_speed",
    ]


def test_build_canonical_feature_frame_drop_unmapped():
    from graphids.core.data.discovery.canonical import (
        CanonicalEntitySpec,
        CanonicalFeatureFrameSpec,
        CanonicalRegistry,
        build_canonical_feature_frame,
    )

    registry = CanonicalRegistry(
        entities=(CanonicalEntitySpec(canonical_id="engine_speed", name="engine_speed", aliases=("rpm",)),)
    )
    df = pl.DataFrame(
        {
            "timestamp": [1.0],
            "vehicle_id": ["truck_a"],
            "rpm": [1000.0],
            "temp": [12.0],
        }
    )
    out = build_canonical_feature_frame(
        df,
        registry,
        spec=CanonicalFeatureFrameSpec(
            feature_cols=("rpm", "temp"),
            alias_col="signal",
            value_col="value",
            keep_cols=("timestamp",),
            vehicle_col="vehicle_id",
            unmapped="drop",
        ),
    )
    assert out.shape[0] == 1
    assert out["canonical_id"].to_list() == ["engine_speed"]
