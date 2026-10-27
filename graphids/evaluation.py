"""Frame-score artifacts and exact metrics, independent of torch and Lightning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .metrics import frame_metrics

SCORE_COLUMNS = (
    "timestamp",
    "can_id",
    "label",
    "classification",
    "payload_nll",
    "payload_supported",
)
SCORE_DTYPE = np.dtype("<f8")
SCORE_FILE = "scores.bin"


class EventScores:
    """Append bounded chunks to one file; publish only after exact coverage checks.

    Values use ``SCORE_COLUMNS`` order. Only chunk metadata stays in memory;
    epoch metrics read labels and scalar scores through a local temporary mapping.
    Chunk boundaries are metadata, not separate filesystem objects.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.chunks: list[dict] = []
        self.coverage: dict[str, int] = {}
        self.by_recording: dict[str, dict] = {}
        self._rows = 0
        self._sources: dict[str, str] = {}
        self._finished = False

    def write(self, values, *, recording: str, source_digest: str, row_start: int):
        if self._finished:
            raise ValueError("event scores already finished")
        if values.ndim != 2 or values.shape[1] != len(SCORE_COLUMNS) or not len(values):
            raise ValueError(
                "event scores require nonempty rows in SCORE_COLUMNS order"
            )
        if row_start != self.coverage.get(recording, 0):
            raise ValueError("duplicate/missing event evaluation rows")
        if recording in self._sources and self._sources[recording] != source_digest:
            raise ValueError("event recording source changed")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents mixing an old/partial epoch into a new run.
        mode = "ab" if self.chunks else "xb"
        with (self.directory / SCORE_FILE).open(mode) as stream:
            np.asarray(values, dtype=SCORE_DTYPE, order="C").tofile(stream)
        self.chunks.append(
            dict(
                recording=recording,
                source_digest=source_digest,
                row_start=row_start,
                count=len(values),
                offset=self._rows,
            )
        )
        self._rows += len(values)
        self._sources[recording] = source_digest
        self.coverage[recording] = row_start + len(values)

    def finish(self, expected: dict[str, int]) -> dict[str, float | None]:
        if self._finished:
            raise ValueError("event scores already finished")
        if self.coverage != expected:
            raise ValueError(
                "incomplete event evaluation; every validation frame is required"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.chunks:
            (self.directory / SCORE_FILE).touch(exist_ok=False)
        storage = dict(
            file=SCORE_FILE,
            dtype=SCORE_DTYPE.str,
            shape=[self._rows, len(SCORE_COLUMNS)],
            order="C",
        )
        if (self.directory / SCORE_FILE).stat().st_size != self._rows * len(
            SCORE_COLUMNS
        ) * SCORE_DTYPE.itemsize:
            raise ValueError("incomplete packed event scores")
        metrics = self._metrics() if self.chunks else {}
        (self.directory / "index.json").write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    prediction_unit="observed_frame",
                    metrics=metrics,
                    by_recording=self.by_recording,
                    coverage=self.coverage,
                    columns=SCORE_COLUMNS,
                    chunks=self.chunks,
                    storage=storage,
                    calibration="unfitted; classification threshold 0.5 is diagnostic only",
                ),
                indent=2,
            )
            + "\n"
        )
        self._finished = True
        return metrics

    def _metrics(self):
        total = sum(chunk["count"] for chunk in self.chunks)
        packed = np.memmap(
            self.directory / SCORE_FILE,
            mode="r",
            dtype=SCORE_DTYPE,
            shape=(total, len(SCORE_COLUMNS)),
        )
        # The scratch file belongs on local TMPDIR, not the persistent lake.
        with tempfile.TemporaryFile() as backing:
            # Keep only label, classification, payload NLL and support columns.
            values = np.memmap(backing, mode="w+", dtype=np.float64, shape=(total, 4))
            # Group each recording contiguously, even if shards were interleaved.
            starts, offset = {}, 0
            for recording, count in self.coverage.items():
                starts[recording] = offset
                offset += count
            for chunk in self.chunks:
                start = chunk["offset"]
                block = packed[start : start + chunk["count"], 2:]
                offset = starts[chunk["recording"]] + chunk["row_start"]
                values[offset : offset + len(block)] = block
            self.by_recording = {
                recording: self._summarize(
                    values[start : start + self.coverage[recording]]
                )
                for recording, start in starts.items()
            }
            return self._summarize(values)

    @staticmethod
    def _summarize(values):
        metrics = {}
        for head, column in (("classification", 1), ("payload", 2)):
            eligible = values[:, 0] >= 0
            if head == "payload":
                eligible &= values[:, 3].astype(bool)
            results = frame_metrics(
                values[eligible, column],
                values[eligible, 0],
                threshold=0.5 if head == "classification" else None,
            )
            metrics.update((f"{head}_{key}", value) for key, value in results.items())
        return metrics


def iter_score_chunks(directory: Path):
    """Read complete legacy shards or packed scores with the same chunk contract."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if index["schema_version"] == 1:
        for number, chunk in enumerate(index["chunks"]):
            with np.load(directory / f"{number:08d}.npz", allow_pickle=False) as shard:
                yield (
                    dict(chunk, source_digest=str(shard["source_digest"])),
                    shard["values"],
                )
        return
    if index["schema_version"] != 2:
        raise ValueError("unsupported event score schema")
    storage = index["storage"]
    shape = tuple(storage["shape"])
    if (
        storage["file"] != SCORE_FILE
        or storage["dtype"] != SCORE_DTYPE.str
        or storage["order"] != "C"
        or len(shape) != 2
        or shape[1] != len(SCORE_COLUMNS)
        or shape[0] < 0
    ):
        raise ValueError("invalid packed event score storage")
    path = directory / SCORE_FILE
    if path.stat().st_size != int(np.prod(shape)) * SCORE_DTYPE.itemsize:
        raise ValueError("incomplete packed event scores")
    if not shape[0]:
        return
    values = np.memmap(path, mode="r", dtype=SCORE_DTYPE, shape=shape)
    offset = 0
    for chunk in index["chunks"]:
        if (
            chunk["offset"] != offset
            or chunk["count"] <= 0
            or offset + chunk["count"] > len(values)
        ):
            raise ValueError("invalid packed event chunk offsets")
        yield chunk, values[offset : offset + chunk["count"]]
        offset += chunk["count"]
    if offset != len(values):
        raise ValueError("incomplete packed event chunk coverage")
