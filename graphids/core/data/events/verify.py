"""Lossless source-only event cache verification."""

import hashlib

import numpy as np

from graphids.core.data.recordings import iter_rows


def verify_recording(events):
    arrays = events.arrays
    digest = hashlib.sha256()
    positives, unknown, tied = 0, 0, 0
    previous_time = None
    count = 0
    for i, row in enumerate(iter_rows(events.recording)):
        label = -1 if row["label"] is None else row["label"]
        length = len(row["data"])
        assert float(arrays["times"][i]) == row["time"], (
            events.recording.key,
            i,
            "time",
        )
        assert int(arrays["ids"][i]) == int(row["id"], 16), (
            events.recording.key,
            i,
            "id",
        )
        assert int(arrays["lengths"][i]) == length, (events.recording.key, i, "length")
        assert int(arrays["labels"][i]) == label, (events.recording.key, i, "label")
        assert arrays["payload"][i, :length].tobytes() == row["data"], (
            events.recording.key,
            i,
            "payload",
        )
        assert not arrays["payload"][i, length:].any(), (
            events.recording.key,
            i,
            "padding",
        )
        positives += label == 1
        unknown += label < 0
        tied += previous_time == row["time"]
        previous_time = row["time"]
        count += 1
    assert count == events.count
    for name, values in arrays.items():
        digest.update(name.encode())
        for lo in range(0, count, 65536):
            digest.update(values[lo : lo + 65536].tobytes())
    return dict(
        recording=events.recording.key,
        role=events.recording.role,
        vehicle=events.recording.vehicle,
        frames=count,
        positive_frames=positives,
        unknown_frames=unknown,
        equal_timestamp_successors=tied,
        input_sha256=events.recording.sha256,
        cache_fields_sha256=digest.hexdigest(),
        distinct_ids=len(np.unique(arrays["ids"])),
        status="complete",
    )


def verify_events(config, output):
    import json
    from pathlib import Path

    from graphids.exp.config import ExperimentConfig
    from graphids.exp.runtime import build_component

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="running",
        representation="can_event",
        source_only=True,
        records=[],
        trained=False,
        target_evaluated=False,
    )

    def persist():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)

    persist()
    try:
        cfg = ExperimentConfig.from_yaml(config)
        dm = build_component(cfg.config["data"])
        dm.setup()
        result["source_manifest_digest"] = dm.manifest.digest
        for recording in dm.event_recordings.values():
            record = verify_recording(recording)
            result["records"].append(record)
            persist()
            print(
                f"Verified {record['recording']}: {record['frames']} frames", flush=True
            )
        result.update(
            status="complete", frames=sum(r["frames"] for r in result["records"])
        )
    except Exception as exc:
        result.update(status="incomplete", error=repr(exc))
        raise
    finally:
        persist()
    return result
