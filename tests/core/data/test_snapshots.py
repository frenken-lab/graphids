"""Causality, provenance, node lifecycle, and objective alignment contracts."""

import csv
from dataclasses import replace

import numpy as np
import pytest

from graphids.core.data.snapshots.context import build_context
from graphids.core.data.snapshots.manifest import (
    CANRecordingManifest,
    Recording,
    file_hash,
    read_rows,
)
from graphids.core.data.snapshots.preprocessing import (
    CANSnapshotRepresentationCfg,
    build_snapshots,
)


def write_record(
    tmp_path,
    *,
    key="capture",
    vehicle="v",
    role="train_query",
    normal=False,
    rows=None,
    session=None,
):
    path = tmp_path / f"{key}.csv"
    rows = rows or [
        (0.0, "A", "00", 0),
        (0.01, "A", "01", 0),
        (0.03, "B", "02", 1),
        (0.04, "A", "03", 0),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "arbitration_id", "data_field", "attack"])
        writer.writerows(rows)
    return Recording(
        key,
        "set_01",
        vehicle,
        session or key,
        str(path),
        file_hash(path),
        role,
        verified_normal=normal,
        payload_length_verified=True,
        labels_verified=True,
    )


def test_empty_bins_boundaries_partial_and_unknown_labels(tmp_path):
    rec = write_record(tmp_path)
    s = build_snapshots(rec, CANSnapshotRepresentationCfg())
    assert s.counts[:, 0].tolist() == [1, 1, 0, 0, 1]
    assert s.labels.tolist() == [0, 0, -1, 1, 0]
    assert s.exposure[-1] == 0
    assert s.features[2, 0, 1] > s.features[1, 0, 1]
    assert s.live[0, 1] == 0  # B must not exist before its first observation.
    assert s.new_ids.tolist() == [1, 0, 0, 1, 0]


def test_snapshot_budget_rejected_before_allocating_dense_history(tmp_path):
    rec = write_record(tmp_path)
    with pytest.raises(ValueError, match="snapshot cache budget exceeded"):
        build_snapshots(rec, CANSnapshotRepresentationCfg(), max_snapshot_bytes=1)


def test_future_observations_and_labels_do_not_change_past(tmp_path):
    rec = write_record(tmp_path, key="one")
    altered = write_record(
        tmp_path,
        key="two",
        rows=[
            (0.0, "A", "00", 0),
            (0.01, "A", "01", 0),
            (0.03, "B", "FF", 0),
            (0.04, "A", "FF", 1),
        ],
    )
    a, b = (build_snapshots(r, CANSnapshotRepresentationCfg()) for r in (rec, altered))
    np.testing.assert_array_equal(a.features[:3], b.features[:3])


def test_no_fabricated_payload_validity_or_normality(tmp_path):
    rec = replace(
        write_record(tmp_path), payload_length_verified=False, labels_verified=False
    )
    s = build_snapshots(rec, CANSnapshotRepresentationCfg())
    assert not s.payload_observed.any()
    assert not s.features[:, :, 12:20].any()
    assert not s.normal.any()
    assert (s.labels == -1).all()


def test_missing_stream_intervals_and_anchor_silence(tmp_path):
    rec = replace(write_record(tmp_path), attack_intervals=((0.02, 0.03),))
    s = build_snapshots(rec, CANSnapshotRepresentationCfg(), anchor_ids=("missing",))
    assert s.live[:, 0].all()
    assert not s.counts[:, 0].any()
    assert s.labels[2] == 1


def test_slot_generations_reset_once_per_bin(tmp_path):
    rec = write_record(
        tmp_path,
        rows=[
            (0.0, "A", "00", 0),
            (0.01, "A", "01", 0),
            (0.05, "A", "02", 0),
            (0.051, "A", "03", 0),
            (0.06, "A", "04", 0),
        ],
    )
    s = build_snapshots(rec, CANSnapshotRepresentationCfg(), retention_seconds=0.015)
    assert s.generation[0, 0] == 0
    assert s.generation[3, 0] == -1
    assert s.generation[5, 0] == 1


def test_manifest_rejects_duplicates_sessions_and_unreviewed(tmp_path):
    rec = write_record(tmp_path)
    with pytest.raises(ValueError, match="reviewed"):
        CANRecordingManifest((rec,)).validate()
    with pytest.raises(ValueError, match="duplicate recording content"):
        CANRecordingManifest(
            (rec, replace(rec, key="copy")), provenance_reviewed=True
        ).validate()
    other = write_record(
        tmp_path,
        key="other",
        rows=[(0, "C", "00", 0), (0.03, "C", "01", 0)],
        role="val_query",
        session=rec.session,
    )
    with pytest.raises(ValueError, match="session crosses"):
        CANRecordingManifest((rec, other), provenance_reviewed=True).validate()


def test_clock_and_payload_rejection(tmp_path):
    rec = write_record(tmp_path, rows=[(0.02, "A", "00", 0), (0.01, "A", "00", 0)])
    with pytest.raises(ValueError, match="decreasing clock"):
        read_rows(rec)
    rec = write_record(
        tmp_path, key="odd", rows=[(0, "A", "0", 0), (0.01, "A", "00", 0)]
    )
    with pytest.raises(ValueError, match="payload length"):
        read_rows(rec)


def test_reference_context_is_frozen_and_provenance_checked(tmp_path):
    rec = write_record(
        tmp_path,
        role="train_reference",
        normal=True,
        rows=[
            (0, "A", "00", 0),
            (0.01, "B", "00", 0),
            (0.02, "A", "01", 0),
            (0.03, "B", "01", 0),
        ],
    )
    s = build_snapshots(rec, CANSnapshotRepresentationCfg())
    c = build_context("v", [s])
    assert c.reference_keys == (rec.key,)
    assert len(c.digest) == 64
    with pytest.raises(ValueError):
        c.profiles[0, 0] = 100
    with pytest.raises(ValueError, match="verified-normal"):
        build_context("different", [s])
