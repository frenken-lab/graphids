"""Merge semantics for cache_metadata.json v2.

Regression guards for §1.1 failure modes: metadata overwritten per split,
no per-split accounting, no invariant mismatch detection.
"""

from __future__ import annotations

import json

import pytest

from graphids.core.data.preprocessing.metadata import (
    METADATA_SCHEMA_VERSION,
    load_metadata,
    merge_split_into_metadata,
    validate_metadata,
)

INVARIANTS = {
    "preprocessing_version": "8.0.0",
    "window_size": 100,
    "stride": 100,
    "val_fraction": 0.2,
    "seed": 42,
    "vocab_digest": "a" * 64,
    # Source of truth for required keys: graphids.core.data.preprocessing.metadata.INVARIANT_KEYS.
    # Co-update this fixture whenever that tuple grows.
    "scaler_strategy": "standard",
}


def _entry(num_graphs: int, min_nodes: int = 5) -> dict:
    return {
        "num_graphs": num_graphs,
        "num_raw_samples": num_graphs * 100,
        "bytes_on_disk": num_graphs * 1024,
        "source_dirs": ["train_01_attack_free"],
        "attack_balance": {"benign": num_graphs},
        "graph_stats": {
            "node_count": {
                "min": min_nodes,
                "max": min_nodes + 20,
                "mean": 12.0,
                "p95": min_nodes + 15,
                "p99": min_nodes + 18,
            },
            "edge_count": {
                "min": min_nodes * 2,
                "max": min_nodes * 4,
                "mean": 24.0,
                "p95": min_nodes * 3,
                "p99": min_nodes * 4,
            },
        },
    }


def test_first_split_seeds_top_level_fields(tmp_path):
    # INVARIANT: the first writer populates schema version, dataset,
    # invariants, num_arb_ids, and splits[<name>].
    merge_split_into_metadata(
        tmp_path,
        "train",
        _entry(100),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    meta = load_metadata(tmp_path)
    assert meta["metadata_schema_version"] == METADATA_SCHEMA_VERSION
    assert meta["dataset"] == "hcrl_sa"
    assert meta["num_arb_ids"] == 128
    for k, v in INVARIANTS.items():
        assert meta[k] == v
    assert set(meta["splits"]) == {"train"}


def test_second_split_merges_without_overwriting_first(tmp_path):
    # REGRESSION (§1.1): second process() used to overwrite cache_metadata.json
    # wholesale. With the merge writer both entries must survive.
    merge_split_into_metadata(
        tmp_path,
        "train",
        _entry(100),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    merge_split_into_metadata(
        tmp_path,
        "test_01",
        _entry(30, min_nodes=3),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    meta = load_metadata(tmp_path)
    assert set(meta["splits"]) == {"train", "test_01"}
    assert meta["splits"]["train"]["num_graphs"] == 100
    assert meta["splits"]["test_01"]["num_graphs"] == 30


def test_aggregate_recomputed_after_each_merge(tmp_path):
    # CONTRACT: aggregate.num_* is a running sum derived from splits,
    # so callers (resource profile Stage 2) can read it without
    # recomputing.
    merge_split_into_metadata(
        tmp_path,
        "train",
        _entry(100),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    merge_split_into_metadata(
        tmp_path,
        "val",
        {"num_graphs": 25, "derived_from": "train", "val_fraction_seed": [0.2, 42]},
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    merge_split_into_metadata(
        tmp_path,
        "test_01",
        _entry(30),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    merge_split_into_metadata(
        tmp_path,
        "test_02",
        _entry(40),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    meta = load_metadata(tmp_path)
    agg = meta["aggregate"]
    assert agg["num_train_graphs"] == 100
    assert agg["num_val_graphs"] == 25
    assert agg["num_test_graphs"] == 70
    assert agg["num_graphs"] == 195


def test_invariant_mismatch_raises(tmp_path):
    # INVARIANT: mismatched window_size/stride/etc between writers means
    # the cache is inconsistent. Fail loud, don't silently overwrite.
    merge_split_into_metadata(
        tmp_path,
        "train",
        _entry(100),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    with pytest.raises(ValueError, match="invariant mismatch"):
        merge_split_into_metadata(
            tmp_path,
            "test_01",
            _entry(30),
            invariants={**INVARIANTS, "window_size": 50},
            dataset_name="hcrl_sa",
            num_arb_ids=128,
        )


def test_load_metadata_rejects_v1(tmp_path):
    # CONTRACT: v1 metadata must be rebuilt, not migrated in place.
    (tmp_path / "cache_metadata.json").write_text(
        json.dumps({"window_size": 100, "graph_stats": {"node_count": {"mean": 12}}})
    )
    with pytest.raises(ValueError, match="schema version"):
        load_metadata(tmp_path)


def test_validate_metadata_clean(tmp_path):
    merge_split_into_metadata(
        tmp_path,
        "train",
        _entry(100),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    meta = load_metadata(tmp_path)
    assert validate_metadata(meta, dataset="hcrl_sa") == []


def test_validate_metadata_catches_missing_test_subdirs(tmp_path):
    merge_split_into_metadata(
        tmp_path,
        "train",
        _entry(100),
        invariants=INVARIANTS,
        dataset_name="hcrl_sa",
        num_arb_ids=128,
    )
    meta = load_metadata(tmp_path)
    errs = validate_metadata(
        meta,
        dataset="hcrl_sa",
        test_subdirs=["test_01_foo", "test_02_bar"],
    )
    assert any("test splits missing" in e for e in errs)
