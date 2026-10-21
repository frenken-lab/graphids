"""Artifact contracts tested without constructing a model or Trainer."""

import json

import numpy as np
import pytest

from graphids.core.evaluation import EventScores
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


@pytest.mark.parametrize("lengths", [(6,), (1, 2, 3)])
def test_shards_preserve_rows_and_metrics_ignore_chunk_boundaries(
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
    assert metrics == {
        "classification_frame_ap": raw_average_precision(scores[:, 3], scores[:, 2]),
        "payload_frame_ap": raw_average_precision(
            scores[scores[:, 5] == 1, 4], scores[scores[:, 5] == 1, 2]
        ),
    }
    restored = []
    for path in sorted(tmp_path.glob("*.npz")):
        with np.load(path) as shard:
            assert str(shard["source_digest"]) == "source-hash"
            assert str(shard["recording"]) == "capture"
            assert int(shard["row_start"]) == sum(len(block) for block in restored)
            restored.append(shard["values"])
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
    assert index["schema_version"] == 1
    assert index["prediction_unit"] == "observed_frame"


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
    assert writer.finish({"a": 2, "b": 1}) == {}
