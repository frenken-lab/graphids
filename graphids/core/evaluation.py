"""Frame-score artifacts and exact metrics, independent of torch and Lightning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .metrics import raw_average_precision

SCORE_COLUMNS = (
    "timestamp",
    "can_id",
    "label",
    "classification",
    "payload_nll",
    "payload_supported",
)


class EventScores:
    """Write bounded chunks; publish an index only after exact coverage checks.

    Values use ``SCORE_COLUMNS`` order. Only chunk metadata stays in memory;
    epoch metrics read labels and scalar scores through a temporary disk mapping.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.chunks: list[dict] = []
        self.coverage: dict[str, int] = {}

    def write(self, values, *, recording: str, source_digest: str, row_start: int):
        if values.ndim != 2 or values.shape[1] != len(SCORE_COLUMNS) or not len(values):
            raise ValueError(
                "event scores require nonempty rows in SCORE_COLUMNS order"
            )
        if row_start != self.coverage.get(recording, 0):
            raise ValueError("duplicate/missing event evaluation rows")
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.directory / f"{len(self.chunks):08d}.npz",
            values=values,
            row_start=row_start,
            recording=recording,
            source_digest=source_digest,
        )
        self.chunks.append(
            dict(recording=recording, row_start=row_start, count=len(values))
        )
        self.coverage[recording] = row_start + len(values)

    def finish(self, expected: dict[str, int]) -> dict[str, float]:
        if self.coverage != expected:
            raise ValueError(
                "incomplete event evaluation; every validation frame is required"
            )
        metrics = self._metrics() if self.chunks else {}
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "index.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    prediction_unit="observed_frame",
                    metrics=metrics,
                    coverage=self.coverage,
                    columns=SCORE_COLUMNS,
                    chunks=self.chunks,
                    calibration="unfitted; classification threshold 0.5 is diagnostic only",
                ),
                indent=2,
            )
            + "\n"
        )
        return metrics

    def _metrics(self):
        total = sum(chunk["count"] for chunk in self.chunks)
        with tempfile.TemporaryFile(dir=self.directory) as backing:
            # Keep only label, classification, payload NLL and support columns.
            values = np.memmap(backing, mode="w+", dtype=np.float64, shape=(total, 4))
            offset = 0
            for index in range(len(self.chunks)):
                with np.load(self.directory / f"{index:08d}.npz") as shard:
                    block = shard["values"][:, 2:]
                    values[offset : offset + len(block)] = block
                    offset += len(block)
            labels = values[:, 0]
            known = labels >= 0
            metrics = {}
            for head, column in (("classification", 1), ("payload", 2)):
                eligible = (
                    known
                    if head == "classification"
                    else known & values[:, 3].astype(bool)
                )
                ap = raw_average_precision(values[eligible, column], labels[eligible])
                if ap is not None:
                    metrics[f"{head}_frame_ap"] = ap
            return metrics
