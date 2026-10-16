"""Lossless frame caches and chronological source-only event chunks.

Caches preserve byte values, lengths, numeric CAN IDs, FP64 recording clocks,
row order (including ties), and unknown labels. Chunks never merge or drop rows.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset

from graphids.core.data.recordings import (
    CANRecordingManifest,
    Recording,
    file_hash,
    iter_rows,
)

from .representation import CANEventRepresentationCfg

COLUMNS = {
    "times": (np.float64, ()),
    "ids": (np.int64, ()),
    "payload": (np.uint8, (8,)),
    "lengths": (np.int64, ()),
    "labels": (np.int64, ()),
}


def recording_digest(rec):
    return hashlib.sha256(json.dumps(asdict(rec), sort_keys=True).encode()).hexdigest()


def materialize_events(rec: Recording, root: Path):
    """Two bounded-memory CSV passes; publish only fully written caches."""
    digest = recording_digest(rec)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"event-v1-{digest}"
    with FileLock(str(destination) + ".lock"):
        if (destination / "metadata.json").exists():
            return destination
        if file_hash(Path(rec.path)) != rec.sha256:
            raise ValueError(f"recording hash changed: {rec.key}")
        count = sum(1 for _ in iter_rows(rec))
        staging = Path(tempfile.mkdtemp(prefix=".events-", dir=root))
        try:
            arrays = {
                name: np.lib.format.open_memmap(
                    staging / f"{name}.npy",
                    mode="w+",
                    dtype=dtype,
                    shape=(count, *tail),
                )
                for name, (dtype, tail) in COLUMNS.items()
            }
            for i, row in enumerate(iter_rows(rec)):
                try:
                    numeric_id = int(row["id"], 16)
                except ValueError as exc:
                    raise ValueError(f"{rec.key}: invalid hexadecimal CAN ID") from exc
                if not 0 <= numeric_id < 2**29:
                    raise ValueError(f"{rec.key}: CAN ID exceeds 29 bits")
                arrays["times"][i] = row["time"]
                arrays["ids"][i] = numeric_id
                arrays["payload"][i] = 0
                arrays["payload"][i, : len(row["data"])] = np.frombuffer(
                    row["data"], np.uint8
                )
                arrays["lengths"][i] = len(row["data"])
                arrays["labels"][i] = -1 if row["label"] is None else row["label"]
            # Detect concurrent source changes across either parsing pass.
            if file_hash(Path(rec.path)) != rec.sha256:
                raise ValueError(
                    f"recording hash changed during cache build: {rec.key}"
                )
            for array in arrays.values():
                array.flush()
            metadata = dict(
                schema_version=1, count=count, recording=asdict(rec), digest=digest
            )
            (staging / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            if destination.exists():
                shutil.rmtree(destination)  # only an unpublished, incomplete cache
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return destination


@dataclass
class CANEventBatch:
    recording: str
    source_digest: str
    row_start: int
    timestamps: torch.Tensor  # FP64 physical seconds, never input labels
    payload: torch.Tensor  # exact uint8 values, [events, 8]
    lengths: torch.Tensor
    ids: torch.Tensor  # exact 29-bit IDs, used as message fields and state keys
    times: torch.Tensor  # [events, 2]: elapsed physical time and preceding bus gap
    labels: torch.Tensor
    scored: torch.Tensor
    verified_normal: bool
    payload_length_verified: bool
    routing_ids: tuple[
        int, ...
    ]  # same IDs on CPU, avoiding per-event GPU synchronizations

    recording_end: bool = False

    def to(self, device, **kwargs):
        return replace(
            self,
            **{
                name: value.to(device, **kwargs)
                for name, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
        )

    def pin_memory(self):
        return replace(
            self,
            **{
                name: value.pin_memory()
                for name, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
        )

    def slice(self, start, stop):
        if not 0 <= start < stop <= len(self.ids):
            raise ValueError("invalid event slice")
        return replace(
            self,
            row_start=self.row_start + start,
            recording_end=self.recording_end and stop == len(self.ids),
            routing_ids=self.routing_ids[start:stop],
            **{
                name: value[start:stop]
                for name, value in vars(self).items()
                if isinstance(value, torch.Tensor)
            },
        )


class EventRecording:
    def __init__(self, path):
        self.path = Path(path)
        metadata = json.loads((self.path / "metadata.json").read_text())
        self.recording = Recording(**metadata["recording"])
        self.digest, self.count = metadata["digest"], metadata["count"]
        self.arrays = {
            name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in COLUMNS
        }
        if any(len(a) != self.count for a in self.arrays.values()):
            raise ValueError("incomplete event cache")

    def batch(self, start, stop, *, score_start=None):
        if not 0 <= start < stop <= self.count:
            raise ValueError("invalid event window")
        rec, arrays = self.recording, self.arrays
        tensors = {
            name: torch.from_numpy(np.array(values[start:stop]))
            for name, values in arrays.items()
            if name != "times"
        }
        physical = np.array(arrays["times"][start:stop])
        previous = np.array(arrays["times"][max(0, start - 1) : stop - 1])
        gaps = physical - previous if start else np.r_[0.0, np.diff(physical)]
        times = np.stack((physical - arrays["times"][0], gaps), -1)
        return CANEventBatch(
            rec.key,
            self.digest,
            start,
            torch.from_numpy(physical),
            tensors["payload"],
            tensors["lengths"],
            tensors["ids"],
            torch.from_numpy(times),
            tensors["labels"],
            torch.arange(start, stop)
            >= (start if score_start is None else score_start),
            rec.verified_normal,
            rec.payload_length_verified,
            tuple(arrays["ids"][start:stop].tolist()),
            stop == self.count,
        )


class EventChunks(Dataset):
    def __init__(self, recordings, chunk_events, warmup_events=0):
        self.recordings, self.windows = recordings, []
        for index, rec in enumerate(recordings):
            self.windows.extend(
                (index, start, min(rec.count, start + chunk_events))
                for start in range(0, rec.count, chunk_events)
            )
        self.warmup_events = warmup_events

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        rec, start, stop = self.windows[index]
        return self.recordings[rec].batch(start, stop)


@dataclass
class EventTrainingWindow:
    """One optimizer step, with CPU target totals for exact accumulation."""

    groups: list[list[CANEventBatch]]
    classification_targets: int
    payload_targets: int

    def to(self, device, **kwargs):
        return replace(
            self, groups=[[b.to(device, **kwargs) for b in g] for g in self.groups]
        )

    def pin_memory(self):
        return replace(self, groups=[[b.pin_memory() for b in g] for g in self.groups])


class EventStreamWindows(Dataset):
    """Schedule independent live recordings; group tails by exact length.

    Replacement starts only after a slot's recording ends. No fake frames,
    warmup replay or padding. Metadata only is retained by the scheduler.
    """

    def __init__(self, recordings, chunk_events, streams=1, effective_events=8192):
        self.recordings, self.windows = recordings, []
        pending = iter(range(len(recordings)))
        active = [
            (i, 0)
            for i in [next(pending, None) for _ in range(streams)]
            if i is not None
        ]
        groups, events = [], 0
        while active:
            by_length, following = {}, []
            for i, start in active:
                stop = min(start + chunk_events, recordings[i].count)
                if stop <= start:
                    raise ValueError("empty recording cannot form an event stream")
                by_length.setdefault(stop - start, []).append((i, start, stop))
                if stop < recordings[i].count:
                    following.append((i, stop))
                else:
                    replacement = next(pending, None)
                    if replacement is not None:
                        following.append((replacement, 0))
            for group in by_length.values():
                count = sum(stop - start for _, start, stop in group)
                if groups and events + count > effective_events:
                    self.windows.append(groups)
                    groups, events = [], 0
                groups.append(group)
                events += count
                if events == effective_events:
                    self.windows.append(groups)
                    groups, events = [], 0
            active = following
        if groups:
            self.windows.append(groups)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        groups = [
            [self.recordings[i].batch(start, stop) for i, start, stop in group]
            for group in self.windows[index]
        ]
        batches = [b for g in groups for b in g]
        return EventTrainingWindow(
            groups,
            sum(int((b.scored & (b.labels >= 0)).sum()) for b in batches),
            sum(
                int((b.lengths * b.scored).sum())
                for b in batches
                if b.verified_normal and b.payload_length_verified
            ),
        )


class CANEventDataModule(pl.LightningDataModule):
    """Explicit source-only learning path; no handcrafted reference features.

    Continuous chronological training and evaluation on one device.
    Each optimizer window declares independent objective target totals.
    """

    def __init__(
        self,
        *,
        manifest_path,
        datasets,
        cache_root,
        representation_cfg=None,
        chunk_events=512,
        warmup_events=0,
        stream_batch_size=1,
        effective_batch_events=8192,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.manifest_path, self.datasets = manifest_path, datasets
        self.cache_root = Path(cache_root)
        self.representation_cfg = representation_cfg or CANEventRepresentationCfg()
        if warmup_events:
            raise ValueError(
                "warmup_events retired: continuous history scores every row once; set warmup_events=0"
            )
        if (
            stream_batch_size not in (1, 2, 4, 8)
            or chunk_events * stream_batch_size > 8192
        ):
            raise ValueError(
                "use 1, 2, 4 or 8 streams and at most 8192 events per microbatch"
            )
        if effective_batch_events < chunk_events * stream_batch_size:
            raise ValueError("effective batch must cover one microbatch")
        self.chunk_events, self.warmup_events = chunk_events, warmup_events
        self.stream_batch_size, self.effective_batch_events = (
            stream_batch_size,
            effective_batch_events,
        )
        self.num_workers, self.pin_memory = num_workers, pin_memory
        self.train_sampler = None  # disable Lightning's automatic padded DDP sampler
        self.event_recordings = {}

    def setup(self, stage=None):
        if self.event_recordings:
            return
        raw = CANRecordingManifest.load(self.manifest_path, verify_files=False)
        # Validate reviewed metadata globally; never open reserved target files.
        self.manifest = CANRecordingManifest(
            tuple(
                r
                for r in raw.records
                if r.role.startswith(("train_", "val_")) and r.dataset in self.datasets
            ),
            provenance_reviewed=True,
        )
        self.manifest.validate(verify_files=False)
        recordings = {}
        for rec in self.manifest.records:
            if rec.role not in ("train_query", "val_query"):
                continue
            if file_hash(Path(rec.path)) != rec.sha256:
                raise ValueError(f"recording hash changed: {rec.key}")
            recordings[rec.key] = EventRecording(
                materialize_events(rec, self.cache_root)
            )
        if not recordings:
            raise ValueError("no reviewed source query recordings selected")
        self.event_recordings = recordings
        self.records = {r.key: r for r in self.manifest.records}
        self.test_data = {"source": None}

    def _loader(self, role, *, training=False):
        if self.trainer is not None and self.trainer.world_size != 1:
            raise ValueError(
                "can_event_dm currently requires one device; no stream may be padded or split across ranks"
            )
        records = [
            r for r in self.event_recordings.values() if r.recording.role == role
        ]
        return DataLoader(
            EventStreamWindows(
                records,
                self.chunk_events,
                self.stream_batch_size,
                self.effective_batch_events,
            )
            if training
            else EventChunks(records, self.chunk_events),
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Dataloader base seeds must not consume the model/dropout RNG;
            # iterator construction differs on epoch-boundary resume.
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return self._loader("train_query", training=True)

    def val_dataloader(self):
        return self._loader("val_query")

    def test_dataloader(self):
        return self._loader("val_query")
