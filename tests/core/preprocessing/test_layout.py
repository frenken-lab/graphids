"""Storage-layout contract tests."""

from __future__ import annotations


def test_data_layer_layout_paths(tmp_path):
    from graphids.core.data.discovery.layout import (
        DataLayerLayout,
        MaterializedViewSpec,
    )

    layout = DataLayerLayout(tmp_path)
    assert layout.raw.path() == tmp_path / "raw_can_events"
    assert layout.views.path() == tmp_path / "materialized_views" / "snapshot"
    assert MaterializedViewSpec(tmp_path, view_kind="rolling_stream").path() == (
        tmp_path / "materialized_views" / "rolling_stream"
    )
    assert layout.hypotheses_path.name == "canonical_hypotheses.parquet"
    assert layout.profiles_path.name == "signal_profiles.parquet"
