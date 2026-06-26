"""Temporal event materialization contract."""

from __future__ import annotations

import csv

import polars as pl
import pytest

from graphids.core.data.preprocessing.temporal import (
    SPLIT_NAME_TO_ID,
    TEMPORAL_MSG_COL_ORDER,
    assert_temporal_splits_disjoint,
    build_temporal_event_table,
    prepare_temporal_eval_table,
    split_temporal_train_val_tables,
    temporal_to_pyg,
)


def _rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.4, 0.0, 0.3],
            "arb_id": ["0x001", "0x002", "0x999", "0x003", "0xABC"],
            "node_id": [1, 2, 0, 3, 0],
            "byte_0": [1.0, 5.0, 9.0, 2.0, 4.0],
            "byte_1": [0.0, 1.0, 1.0, 0.0, 8.0],
            "entropy": [0.0, 0.2, 0.5, 0.1, 0.3],
            "attack": [0, 1, 0, 0, 1],
            "attack_type": [0, 2, 0, 0, 5],
            "vehicle_id": ["veh_a", "veh_a", "veh_a", "veh_a", "veh_a"],
            "source_dir": ["train", "train", "train", "test", "test"],
            "source_file": ["a.csv", "a.csv", "a.csv", "b.csv", "b.csv"],
        }
    )


def test_temporal_event_table_preserves_transitions_unknowns_and_resets():
    table = build_temporal_event_table(_rows())

    assert table["event_id"].to_list() == [0, 1, 2, 3, 4]
    assert table["stream_id"].to_list() == [0, 0, 0, 1, 1]
    assert table["src_id"].to_list() == [1, 1, 2, 3, 3]
    assert table["dst_id"].to_list() == [1, 2, 0, 3, 0]
    assert table["src_raw"].to_list() == ["0x001", "0x001", "0x002", "0x003", "0x003"]
    assert table["dst_raw"].to_list() == ["0x001", "0x002", "0x999", "0x003", "0xABC"]
    assert table["dst_is_unknown"].to_list() == [False, False, True, False, True]
    assert table["src_is_unknown"].to_list() == [False, False, False, False, False]
    assert table["dst_unknown_bucket"].to_list()[2] > 0
    assert table["dst_unknown_bucket"].to_list()[4] > 0
    assert table["reset_after"].to_list() == [False, False, True, False, True]
    assert table["iat"].to_list() == pytest.approx([0.0, 0.1, 0.3, 0.0, 0.3])
    assert table["byte_0_delta"].to_list() == [0.0, 4.0, 4.0, 0.0, 2.0]
    assert table.select("vehicle_id", "source_dir", "source_file").rows()[0] == (
        "veh_a",
        "train",
        "a.csv",
    )


def test_temporal_to_pyg_emits_event_tensor_contract():
    table = build_temporal_event_table(_rows())
    data = temporal_to_pyg(table)

    assert data.src.tolist() == table["src_id"].to_list()
    assert data.dst.tolist() == table["dst_id"].to_list()
    assert data.y.tolist() == [0, 1, 0, 0, 1]
    assert data.attack_type.tolist() == [0, 2, 0, 0, 5]
    assert data.stream_id.tolist() == [0, 0, 0, 1, 1]
    assert data.reset_after.tolist() == [False, False, True, False, True]
    assert data.event_id.tolist() == [0, 1, 2, 3, 4]
    assert tuple(data.msg.shape) == (5, len(TEMPORAL_MSG_COL_ORDER))


def test_temporal_train_val_split_is_stream_local_and_masks_warmup():
    table = build_temporal_event_table(_rows().filter(pl.col("source_dir") == "train"))
    train, val = split_temporal_train_val_tables(
        table,
        val_fraction=0.4,
        val_warmup_events=1,
    )

    assert train["event_id"].to_list() == [0, 1]
    assert val["event_id"].to_list() == [2]
    assert_temporal_splits_disjoint(train, val)
    assert train["reset_after"].to_list() == [False, True]
    assert val["reset_after"].to_list() == [True]
    assert val["src_id"].to_list() == val["dst_id"].to_list()
    assert val["iat"].to_list() == [0.0]
    assert val["byte_0_delta"].to_list() == [0.0]
    assert val["split_name"].to_list() == ["val"]
    assert val["split_id"].to_list() == [SPLIT_NAME_TO_ID["val"]]
    assert val["is_warmup"].to_list() == [True]
    assert val["is_scored"].to_list() == [False]

    data = temporal_to_pyg(val)
    assert data.split_name == "val"
    assert data.split_id.tolist() == [SPLIT_NAME_TO_ID["val"]]
    assert data.is_warmup.tolist() == [True]
    assert data.is_scored.tolist() == [False]


def test_temporal_eval_table_marks_warmup_per_stream():
    table = prepare_temporal_eval_table(
        build_temporal_event_table(_rows()),
        split_name="test",
        warmup_events=1,
    )

    assert table["split_name"].to_list() == ["test"] * 5
    assert table["split_id"].to_list() == [SPLIT_NAME_TO_ID["test"]] * 5
    assert table["is_warmup"].to_list() == [True, False, False, True, False]
    assert table["is_scored"].to_list() == [False, True, True, False, True]


def _write_can_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "arb_id", "data_field", "attack"])
        writer.writerows(rows)


def test_can_temporal_source_uses_train_only_vocab_for_unseen_test_ids(tmp_path, monkeypatch):
    from graphids.core.data.datasets.can_bus import CANBusTemporalSource

    raw = tmp_path / "raw"
    _write_can_csv(
        raw / "train" / "normal.csv",
        [
            [0.0, "0x001", "0102030405060708", 0],
            [0.1, "0x002", "0203040506070809", 0],
        ],
    )
    _write_can_csv(
        raw / "test" / "attack.csv",
        [
            [0.0, "0x001", "0102030405060708", 0],
            [0.1, "0x999", "0908070605040302", 1],
        ],
    )

    monkeypatch.setattr(
        "graphids.paths.load_catalog",
        lambda: {"dummy": {"train_subdir": "train", "test_subdirs": ["test"]}},
    )
    monkeypatch.setattr("graphids.paths.data_dir", lambda lake, name: raw)
    monkeypatch.setattr("graphids.paths.cache_dir", lambda lake, name: tmp_path / "cache" / name)

    state = CANBusTemporalSource(
        name="dummy",
        lake_root="lake",
        test_warmup_events=1,
    ).build()

    assert state.train.dst.tolist() == [1]
    assert state.val.dst.tolist() == [2]
    assert state.test["test"].dst.tolist() == [1, 0]
    assert state.train.split_name == "train"
    assert state.val.split_name == "val"
    assert state.test["test"].split_name == "test"
    assert state.train.is_scored.tolist() == [True]
    assert state.val.is_scored.tolist() == [True]
    assert state.test["test"].is_warmup.tolist() == [True, False]
    assert state.test["test"].is_scored.tolist() == [False, True]
    assert state.test["test"].msg.shape[1] == len(TEMPORAL_MSG_COL_ORDER)
