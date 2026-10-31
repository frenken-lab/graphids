import csv

from graphids.data.recordings import Recording, file_hash


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


def event_record(
    tmp_path, *, key="frames", normal=False, role="train_query", local_context=False
):
    """A tiny recording with tied clocks, repeated IDs, and varying payloads."""
    from graphids.data.events import EventRecording, materialize_events

    rec = write_record(
        tmp_path,
        key=key,
        normal=normal,
        role=role,
        rows=[
            (100.0, "0C9", "00010203040506", 0),
            (100.001, "0C9", "FFFFFFFFFFFFFF", 0 if normal else 1),
            (100.001, "0C9", "00010203040506", 0),
            (100.009, "1F5", "00", 0),
            (100.01, "0C9", "00010203040506", 0),
            (100.02, "1F5", "", 0),
        ],
    )
    return rec, EventRecording(
        materialize_events(rec, tmp_path / "cache"), local_context=local_context
    )


def model(backend="reference", **kwargs):
    """A reproducible small model for CPU and official-backend comparisons."""
    import torch

    from graphids.models.event_graph_mamba import EventGraphMamba

    torch.manual_seed(42)
    kwargs = dict(schema_version=1, use_id_gap=False) | kwargs
    return EventGraphMamba(
        width=8,
        layers=2,
        d_state=3,
        d_conv=3,
        expand=1,
        dropout=0,
        backend=backend,
        **kwargs,
    )


def source_dm(tmp_path, *, representation_cfg=None):
    """Reviewed source fixtures with an inaccessible reserved target recording."""
    import json
    from dataclasses import asdict, replace

    from graphids.data.events import CANEventDataModule
    from graphids.data.recordings import CANRecordingManifest

    train, _ = event_record(tmp_path, key="train", normal=True)
    val, _ = event_record(tmp_path, key="validation", role="val_query")
    # Distinct content/provenance, independent source sessions.
    path = tmp_path / "manifest.json"
    # A reserved target path must never be opened by the event path.
    target = replace(
        val,
        key="reserved",
        vehicle="target",
        session="reserved",
        role="target_attack_test",
        path="/nonexistent/reserved.csv",
        sha256="f" * 64,
    )
    manifest = CANRecordingManifest((train, val, target), provenance_reviewed=True)
    path.write_text(json.dumps(asdict(manifest)))
    dm = CANEventDataModule(
        manifest_path=str(path),
        datasets=["set_01"],
        cache_root=str(tmp_path / "cache"),
        chunk_events=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        representation_cfg=representation_cfg,
    )
    dm.setup()
    assert len(dm.event_recordings) == 2
    assert all(not r.role.startswith("target") for r in dm.manifest.records)
    return dm
