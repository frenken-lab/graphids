"""Artifact contracts tested without constructing a model or Trainer."""

import json

import numpy as np
import pytest

from graphids.core.evaluation import EventScores, iter_score_chunks
from graphids.core.metrics import raw_average_precision


@pytest.fixture
def scores():
    # Tied clocks/scores, a 29-bit ID, unknown labels and unsupported payloads.
    return np.array(
        [
            [100, 2**29 - 1, 1, 0.8, 2.0, 1],
            [100, 0xC9, 0, 0.8, 1.0, 1],
            [101, 0xC9, -1, 0.99, 50.0, 1],
            [102, 0x1F5, 1, 0.2, 100.0, 0],
            [103, 0xC9, 0, 0.1, 3.0, 1],
            [104, 0x1F5, 1, 0.9, 4.0, 1],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("lengths", [(6,), (1, 2, 3), (1,) * 6])
def test_packed_scores_preserve_rows_and_metrics_ignore_chunk_boundaries(
    tmp_path, scores, lengths
):
    """Execution chunks must not change scores, AP eligibility or physical rows."""
    writer = EventScores(tmp_path)
    start = 0
    for length in lengths:
        writer.write(
            scores[start : start + length],
            recording="capture",
            source_digest="source-hash",
            row_start=start,
        )
        start += length
    metrics = writer.finish({"capture": len(scores)})
    assert {
        key: metrics[key] for key in ("classification_frame_ap", "payload_frame_ap")
    } == {
        "classification_frame_ap": raw_average_precision(scores[:, 3], scores[:, 2]),
        "payload_frame_ap": raw_average_precision(
            scores[scores[:, 5] == 1, 4], scores[scores[:, 5] == 1, 2]
        ),
    }
    restored = []
    for chunk, values in iter_score_chunks(tmp_path):
        assert chunk["source_digest"] == "source-hash"
        assert chunk["recording"] == "capture"
        assert chunk["row_start"] == sum(len(block) for block in restored)
        restored.append(values)
    np.testing.assert_array_equal(np.concatenate(restored), scores)
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["metrics"] == metrics
    assert index["coverage"] == {"capture": len(scores)}
    assert index["columns"] == [
        "timestamp",
        "can_id",
        "label",
        "classification",
        "payload_nll",
        "payload_supported",
    ]
    assert index["schema_version"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json", "scores.bin"]
    assert index["prediction_unit"] == "observed_frame"
    assert index["by_recording"]["capture"] == metrics
    assert metrics["classification_frame_auroc"] == pytest.approx(0.75)
    assert [metrics[f"classification_{k}"] for k in ("tp", "fp", "fn", "tn")] == [
        2,
        1,
        1,
        1,
    ]
    assert metrics["classification_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("next_row", [0, 3])
def test_duplicate_or_missing_rows_cannot_publish_results(tmp_path, scores, next_row):
    """A plausible total count cannot hide overlapping or omitted physical rows."""
    writer = EventScores(tmp_path)
    writer.write(scores[:2], recording="a", source_digest="hash", row_start=0)
    with pytest.raises(ValueError, match="duplicate/missing"):
        writer.write(
            scores[2:4], recording="a", source_digest="hash", row_start=next_row
        )
    assert not (tmp_path / "index.json").exists()


@pytest.mark.parametrize("expected", [{"a": 3}, {"a": 2, "b": 2}])
def test_incomplete_recording_or_cohort_cannot_publish_results(
    tmp_path, scores, expected
):
    """An index is evidence of complete cohort coverage, not a partial epoch."""
    writer = EventScores(tmp_path)
    writer.write(scores[:2], recording="a", source_digest="hash", row_start=0)
    with pytest.raises(ValueError, match="incomplete"):
        writer.finish(expected)
    assert not (tmp_path / "index.json").exists()


def test_recordings_keep_independent_row_positions_and_single_class_metrics_are_omitted(
    tmp_path, scores
):
    """Switching recordings must neither reset nor mix their coverage counters."""
    writer = EventScores(tmp_path)
    for recording, row_start in [("a", 0), ("b", 0), ("a", 1)]:
        writer.write(
            scores[:1],
            recording=recording,
            source_digest=recording,
            row_start=row_start,
        )
    metrics = writer.finish({"a": 2, "b": 1})
    assert "classification_frame_ap" not in metrics
    assert "classification_frame_auroc" not in metrics
    assert metrics["classification_tp"] == 3
    assert writer.by_recording["a"]["classification_frames"] == 2
    assert writer.by_recording["b"]["classification_frames"] == 1
    restored = list(iter_score_chunks(tmp_path))
    assert [chunk["recording"] for chunk, _ in restored] == ["a", "b", "a"]
    np.testing.assert_array_equal(
        np.concatenate([v for _, v in restored]), np.tile(scores[:1], (3, 1))
    )


def test_existing_epoch_and_changed_source_are_rejected(tmp_path, scores):
    writer = EventScores(tmp_path)
    writer.write(scores[:1], recording="a", source_digest="one", row_start=0)
    with pytest.raises(ValueError, match="source changed"):
        writer.write(scores[:1], recording="a", source_digest="two", row_start=1)
    with pytest.raises(FileExistsError):
        EventScores(tmp_path).write(
            scores, recording="a", source_digest="one", row_start=0
        )


def test_truncated_packed_scores_cannot_publish_or_be_read(tmp_path, scores):
    writer = EventScores(tmp_path)
    writer.write(scores, recording="a", source_digest="one", row_start=0)
    writer.finish({"a": len(scores)})
    with (tmp_path / "scores.bin").open("r+b") as f:
        f.truncate(8)
    with pytest.raises(ValueError, match="incomplete packed"):
        list(iter_score_chunks(tmp_path))


def test_empty_epoch_and_finished_writer(tmp_path, scores):
    writer = EventScores(tmp_path)
    assert writer.finish({}) == {}
    assert list(iter_score_chunks(tmp_path)) == []
    with pytest.raises(ValueError, match="already finished"):
        writer.write(scores, recording="a", source_digest="one", row_start=0)


def legacy_epoch(directory, scores):
    """A real legacy layout, independent of the new writer implementation."""
    chunks = []
    for number, start in enumerate(range(0, len(scores), 2)):
        block = scores[start : start + 2]
        np.savez(
            directory / f"{number:08d}.npz",
            values=block,
            recording="capture",
            source_digest="source",
            row_start=start,
        )
        chunks.append(dict(recording="capture", row_start=start, count=len(block)))
    metrics = EventScores._summarize(scores[:, 2:])
    index = dict(
        schema_version=1,
        prediction_unit="observed_frame",
        columns=[
            "timestamp",
            "can_id",
            "label",
            "classification",
            "payload_nll",
            "payload_supported",
        ],
        coverage=dict(capture=len(scores)),
        chunks=chunks,
        metrics=metrics,
        by_recording=dict(capture=metrics),
    )
    (directory / "index.json").write_text(json.dumps(index))
    return index


def test_legacy_migration_keeps_every_value_and_existing_metrics(tmp_path, scores):
    from scripts.pack_event_scores import pack_epoch

    original = legacy_epoch(tmp_path, scores)
    legacy = list(iter_score_chunks(tmp_path))
    result = pack_epoch(tmp_path, remove_shards=True)
    assert result["removed"] == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json", "scores.bin"]
    packed = list(iter_score_chunks(tmp_path))
    for (old_meta, old), (new_meta, new) in zip(legacy, packed, strict=True):
        np.testing.assert_array_equal(old, new)
        assert old_meta.items() <= new_meta.items()
    np.testing.assert_array_equal(np.concatenate([v for _, v in packed]), scores)
    current = json.loads((tmp_path / "index.json").read_text())
    assert current["metrics"] == original["metrics"]
    assert current["by_recording"] == original["by_recording"]
    assert pack_epoch(tmp_path, remove_shards=True)["status"] == "already packed"


def test_bad_legacy_shard_cannot_replace_index_or_delete_evidence(tmp_path, scores):
    from scripts.pack_event_scores import pack_epoch

    original = legacy_epoch(tmp_path, scores)
    np.savez(
        tmp_path / "00000001.npz",
        values=scores[2:4],
        recording="capture",
        source_digest="source",
        row_start=99,
    )
    with pytest.raises(ValueError, match="metadata/coverage"):
        pack_epoch(tmp_path, remove_shards=True)
    assert json.loads((tmp_path / "index.json").read_text()) == original
    assert len(list(tmp_path.glob("*.npz"))) == 3
    assert not (tmp_path / "scores.bin").exists()


def test_unknown_labels_and_no_predictions_do_not_invent_precision():
    from graphids.core.metrics import frame_metrics

    metrics = frame_metrics([0.1, 0.2, 1.0], [0, 1, -1], threshold=0.5)
    assert metrics["frame_auroc"] == 1
    assert metrics["recall"] == metrics["f1"] == 0
    assert metrics["precision"] is None
    assert metrics["frames"] == 2


def test_auc_ties_receive_half_credit():
    from graphids.core.metrics import frame_metrics

    assert frame_metrics([0.5, 0.5], [0, 1])["frame_auroc"] == 0.5
