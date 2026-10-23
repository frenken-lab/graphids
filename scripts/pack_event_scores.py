"""Losslessly pack complete schema-1 event epochs; optionally remove verified shards.

python scripts/pack_event_scores.py EPOCH_DIRECTORY --remove-shards
Metrics and physical rows are preserved. An index is replaced only after the
new packed file's SHA256 agrees with the canonical source rows. Incomplete
epochs without an index are deliberately excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from graphids.core.evaluation import SCORE_COLUMNS, SCORE_DTYPE, SCORE_FILE


def pack_epoch(directory: Path, *, remove_shards=False):
    directory = Path(directory)
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    if index["schema_version"] == 2:
        return dict(directory=str(directory), status="already packed", removed=0)
    if index["schema_version"] != 1 or index["prediction_unit"] != "observed_frame":
        raise ValueError("requires a complete schema-1 observed-frame epoch")
    if tuple(index["columns"]) != SCORE_COLUMNS:
        raise ValueError("unexpected event score columns")
    target = directory / SCORE_FILE
    chunks, coverage, sources, offset = [], {}, {}, 0
    digest = hashlib.sha256()
    shard_paths = []
    try:
        with target.open("xb") as output:
            for number, chunk in enumerate(index["chunks"]):
                path = directory / f"{number:08d}.npz"
                with np.load(path, allow_pickle=False) as shard:
                    recording = str(shard["recording"])
                    source = str(shard["source_digest"])
                    start = int(shard["row_start"])
                    values = np.asarray(shard["values"], dtype=SCORE_DTYPE, order="C")
                    if (
                        values.shape != (chunk["count"], len(SCORE_COLUMNS))
                        or recording != chunk["recording"]
                        or start != chunk["row_start"]
                        or start != coverage.get(recording, 0)
                        or (recording in sources and sources[recording] != source)
                    ):
                        raise ValueError(
                            f"legacy shard metadata/coverage differs: {path}"
                        )
                    data = values.tobytes(order="C")
                    output.write(data)
                    digest.update(data)
                    chunks.append(dict(chunk, source_digest=source, offset=offset))
                    offset += len(values)
                    coverage[recording] = start + len(values)
                    sources[recording] = source
                shard_paths.append(path)
            output.flush()
            os.fsync(output.fileno())
        if coverage != index["coverage"]:
            raise ValueError("legacy cohort coverage differs from index")
        with target.open("rb") as packed:
            actual = hashlib.file_digest(packed, "sha256").hexdigest()
        if actual != digest.hexdigest():
            raise ValueError("packed event score hash mismatch")
        updated = dict(
            index,
            schema_version=2,
            chunks=chunks,
            storage=dict(
                file=SCORE_FILE,
                dtype=SCORE_DTYPE.str,
                shape=[offset, len(SCORE_COLUMNS)],
                order="C",
            ),
            migration=dict(from_schema=1, packed_sha256=actual),
        )
        temporary = directory / "index.json.tmp"
        temporary.write_text(json.dumps(updated, indent=2) + "\n")
        temporary.replace(index_path)
    except FileExistsError:
        raise  # Never remove an existing packed file belonging to another attempt.
    except BaseException:
        # The legacy index/shards remain authoritative until index replacement.
        target.unlink(missing_ok=True)
        raise
    if remove_shards:
        for path in shard_paths:
            path.unlink()
    return dict(
        directory=str(directory),
        status="packed",
        rows=offset,
        sha256=actual,
        removed=len(shard_paths) if remove_shards else 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--remove-shards", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        print(
            json.dumps(pack_epoch(directory, remove_shards=args.remove_shards)),
            flush=True,
        )


if __name__ == "__main__":
    main()
