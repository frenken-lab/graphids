"""Signal profile and hypothesis-store tests."""

from __future__ import annotations

import polars as pl


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "vehicle_id": ["veh_a", "veh_a", "veh_b", "veh_b"],
            "arb_id": ["0x100", "0x100", "0x200", "0x200"],
            "timestamp": [1.0, 2.0, 1.0, 3.0],
            "byte_0": [1.0, 2.0, 10.0, 11.0],
            "byte_1": [0.0, 0.0, 1.0, 1.0],
            "entropy": [0.1, 0.2, 0.3, 0.4],
            "attack": [0, 0, 0, 1],
        }
    )


def test_build_signal_profiles_groups_by_vehicle_and_signal():
    from graphids.core.data.discovery.hypotheses import build_signal_profiles

    out = build_signal_profiles(_frame())
    assert set(out.columns) >= {
        "vehicle_id",
        "arb_id",
        "signal_key",
        "msg_count",
        "timestamp_min",
        "timestamp_max",
        "duration",
        "entropy_mean",
        "byte_0_mean",
        "byte_1_mean",
        "attack_max",
    }
    assert out.shape[0] == 2
    assert set(out["signal_key"].to_list()) == {"veh_a::0x100", "veh_b::0x200"}


def test_initialize_hypotheses_produces_empty_provisional_table(tmp_path):
    from graphids.core.data.discovery.hypotheses import (
        DiscoveryStore,
        build_signal_profiles,
        initialize_hypotheses,
    )

    profiles = build_signal_profiles(_frame())
    hyps = initialize_hypotheses(profiles)
    assert set(hyps.columns) >= {
        "vehicle_id",
        "arb_id",
        "signal_key",
        "candidate_canonical_id",
        "confidence",
        "status",
        "evidence",
    }
    assert hyps["status"].to_list() == ["unreviewed", "unreviewed"]
    store = DiscoveryStore(tmp_path)
    assert store.write_profiles(profiles).exists()
    assert store.write_hypotheses(hyps).exists()
    assert store.load_profiles().shape[0] == 2
    assert store.load_hypotheses().shape[0] == 2
