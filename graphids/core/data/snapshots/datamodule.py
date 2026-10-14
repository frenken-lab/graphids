"""Recording episodes, stream batching, and nonduplicating distributed schedules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset, Sampler

from .batch import collate_snapshots, recording_batch
from .context import build_context
from .manifest import CANRecordingManifest
from .policy import ReferencePolicy
from .preprocessing import (
    CANSnapshotRepresentationCfg,
    SnapshotRecording,
    build_snapshots,
)


def power_of_two(value: int) -> int:
    return 1 << max(0, (max(1, value) - 1).bit_length())


class EpisodeDataset(Dataset):
    def __init__(self, dm, entries, *, training: bool):
        self.dm, self.entries, self.training = dm, entries, training

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        dummy = index == -1
        index = 0 if dummy else index
        key, start, stop, warmup, mode = self.entries[index]
        context = self.dm.context_for(key) if mode == "conditioned" else None
        snapshot = self.dm.snapshot_for(key, context)
        batch = recording_batch(
            snapshot,
            context,
            start=start,
            stop=stop,
            warmup=warmup,
            neighbor_cap=self.dm.neighbor_cap,
            anchor_capacity=self.dm.anchor_capacity,
            transient_capacity=self.dm.transient_capacity,
            reference_policy=self.dm.reference_policy,
        )
        work = batch.live.numel()
        messages = batch.edge_mask.sum().item() * batch.x.shape[1]
        if work > self.dm.max_node_bin_tokens or messages > self.dm.max_graph_messages:
            raise ValueError(
                f"snapshot workload overflow for {key}: {work} node-bins, {messages} graph messages"
            )
        if dummy:
            batch.scored.zero_()
            batch.normal.zero_()
            batch.labels.fill_(-1)
        return batch


class EpisodeSampler(Sampler):
    """A shared schedule, sharded by rank. Dummy tails keep DDP steps equal."""

    def __init__(self, dataset: EpisodeDataset, *, seed: int, epoch: int = 0):
        self.dataset, self.seed, self.epoch = dataset, seed, epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def schedule(self):
        groups = {}
        for i, entry in enumerate(self.dataset.entries):
            rec = self.dataset.dm.records[entry[0]]
            groups.setdefault(rec.dataset, {}).setdefault(rec.vehicle, []).append(i)
        rng = np.random.default_rng(self.seed + self.epoch)
        # Equal unique windows per vehicle and dataset; rotate omitted windows next epoch.
        per_vehicle = min(
            len(v) for vehicles in groups.values() for v in vehicles.values()
        )
        per_dataset = min(len(vehicles) * per_vehicle for vehicles in groups.values())
        selected = []
        for dataset in sorted(groups):
            windows = []
            for _vehicle, indices in sorted(groups[dataset].items()):
                rng.shuffle(indices)
                windows.extend(indices[:per_vehicle])
            rng.shuffle(windows)
            selected.extend(windows[:per_dataset])
        rng.shuffle(selected)
        return selected

    def __iter__(self):
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if distributed else 0
        world = torch.distributed.get_world_size() if distributed else 1
        entries = self.schedule()
        entries += [-1] * ((-len(entries)) % world)
        return iter(entries[rank::world])

    def __len__(self):
        world = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        return math.ceil(len(self.schedule()) / world)


class RecordingEvalSampler(Sampler):
    """Assign complete recording/mode sequences to ranks, never individual bins."""

    def __init__(self, dataset):
        self.dataset = dataset

    def _schedules(self):
        world = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        groups = {}
        for i, (key, _start, _stop, _warmup, mode) in enumerate(self.dataset.entries):
            groups.setdefault((key, mode), []).append(i)
        schedules = [[] for _ in range(world)]
        for indices in sorted(groups.values(), key=lambda group: -len(group)):
            rank = min(range(world), key=lambda r: (len(schedules[r]), r))
            schedules[rank].extend(indices)
        steps = max(map(len, schedules), default=0)
        return [schedule + [-1] * (steps - len(schedule)) for schedule in schedules]

    def __iter__(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return iter(self._schedules()[rank])

    def __len__(self):
        return len(self._schedules()[0])


class CANSnapshotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        manifest_path: str,
        datasets: list[str],
        representation_cfg: CANSnapshotRepresentationCfg | None = None,
        sequence_bins: int = 128,
        warmup_bins: int = 128,
        stream_batch_size: int = 1,
        eval_chunk_bins: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        context_withholding: float = 0.25,
        neighbor_cap: int = 8,
        retention_seconds: float = 1.0,
        anchor_capacity: int | None = None,
        transient_capacity: int | None = None,
        max_node_bin_tokens: int = 131072,
        max_graph_messages: int = 1048576,
        max_snapshot_bytes: int = 4 * 1024**3,
        cache_root: str | None = None,
        reference_policy=None,
    ):
        super().__init__()
        self.manifest_path, self.datasets = manifest_path, tuple(datasets)
        self.representation_cfg = representation_cfg or CANSnapshotRepresentationCfg()
        self.sequence_bins, self.warmup_bins = sequence_bins, warmup_bins
        self.stream_batch_size, self.eval_chunk_bins, self.num_workers = (
            stream_batch_size,
            eval_chunk_bins,
            num_workers,
        )
        self.pin_memory = pin_memory
        self.seed, self.context_withholding, self.neighbor_cap = (
            seed,
            context_withholding,
            neighbor_cap,
        )
        self.retention_seconds = retention_seconds
        self.anchor_capacity, self.transient_capacity = (
            anchor_capacity,
            transient_capacity,
        )
        self.max_node_bin_tokens, self.max_graph_messages = (
            max_node_bin_tokens,
            max_graph_messages,
        )
        self.max_snapshot_bytes = max_snapshot_bytes
        self.cache_root = Path(cache_root) if cache_root else None
        self._setup = False
        self.reference_policy = ReferencePolicy.parse(reference_policy)
        self._snapshots = {}
        self._contexts = {}
        if not datasets or len(set(datasets)) != len(datasets):
            raise ValueError("datasets must be a nonempty unique list")
        if (
            min(
                sequence_bins,
                stream_batch_size,
                eval_chunk_bins,
                neighbor_cap,
                max_node_bin_tokens,
                max_graph_messages,
            )
            < 1
            or warmup_bins < 0
            or num_workers < 0
        ):
            raise ValueError(
                "snapshot dimensions/budgets must be positive and warmup nonnegative"
            )
        if not 0 <= context_withholding <= 1:
            raise ValueError("context_withholding must be in [0,1]")

    def setup(self, stage=None):
        if self._setup:
            return
        manifest = CANRecordingManifest.load(self.manifest_path)
        available = {r.dataset for r in manifest.records}
        missing = set(self.datasets) - available
        if missing:
            raise ValueError(
                f"requested datasets missing from reviewed manifest: {sorted(missing)}"
            )
        self.manifest = manifest
        # datasets selects source training families; target families are declared
        # independently by the reviewed manifest for genuine dataset holdout.
        self.records = {
            r.key: r
            for r in manifest.records
            if r.dataset in self.datasets or r.role.startswith("target_")
        }
        if self.cache_root is None:
            from graphids.paths import cache_dir, lake_root

            self.cache_root = cache_dir(lake_root(), "can_snapshot")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        for vehicle in sorted({r.vehicle for r in self.records.values()}):
            for phase in ("train", "val", "target"):
                refs = [
                    r
                    for r in self.records.values()
                    if r.vehicle == vehicle and r.role == f"{phase}_reference"
                ]
                if not refs and phase == "val":
                    refs = [
                        r
                        for r in self.records.values()
                        if r.vehicle == vehicle and r.role == "train_reference"
                    ]
                if refs:
                    snapshots = [self.snapshot_for(r.key, None) for r in refs]
                    self._contexts[vehicle, phase] = build_context(
                        vehicle, snapshots, neighbor_cap=self.neighbor_cap
                    )
                    context = self._contexts[vehicle, phase]
                    context_path = self.cache_root / f"context-{context.digest}.json"
                    with FileLock(str(context_path) + ".lock"):
                        if not context_path.exists():
                            payload = asdict(context)
                            temp = context_path.with_suffix(".tmp")
                            temp.write_text(
                                json.dumps(
                                    payload,
                                    default=lambda value: value.tolist(),
                                    sort_keys=True,
                                )
                            )
                            temp.replace(context_path)
        anchor_load = max(
            (
                len(c.anchor_ids)
                for (v, p), c in self._contexts.items()
                if p != "target"
            ),
            default=1,
        )
        transient_load = 1
        for rec in self.records.values():
            if not rec.role.startswith(("train_", "val_")):
                continue
            snapshot = self.snapshot_for(rec.key, None)
            transient_load = max(transient_load, int(snapshot.live.sum(1).max()))
        self.anchor_capacity = self.anchor_capacity or power_of_two(anchor_load)
        self.transient_capacity = self.transient_capacity or power_of_two(
            transient_load
        )
        if min(self.anchor_capacity, self.transient_capacity) < 1:
            raise ValueError("capacities must be positive")
        train = self._entries("train_query", training=True)
        self.val_entries = self._entries("val_query", training=False)
        if not train or not self.val_entries:
            raise ValueError(
                "manifest requires source training and validation query recordings"
            )
        source_datasets = {self.records[e[0]].dataset for e in train}
        if source_datasets != set(self.datasets):
            raise ValueError(
                "each requested pooled dataset requires source training queries"
            )
        self.train_data = EpisodeDataset(self, train, training=True)
        self.train_sampler = EpisodeSampler(self.train_data, seed=self.seed)
        self.val_data = EpisodeDataset(self, self.val_entries, training=False)
        tests = {}
        for role in ("target_calibration", "target_normal_test", "target_attack_test"):
            for mode in ("zero-shot", "conditioned"):
                entries = self._entries(role, training=False, forced_mode=mode)
                if entries:
                    tests[f"{role}/{mode}"] = EpisodeDataset(
                        self, entries, training=False
                    )
        self.test_data = tests
        self.in_channels = 23
        self._setup = True

    def context_for(self, key):
        rec = self.records[key]
        phase = rec.role.split("_", 1)[0]
        return self._contexts.get((rec.vehicle, phase))

    def snapshot_for(self, key, context):
        rec = self.records[key]
        digest = hashlib.sha256(
            json.dumps(
                {
                    "hash": rec.sha256,
                    "record": asdict(rec),
                    "cfg": asdict(self.representation_cfg),
                    "retention": self.retention_seconds,
                    "anchors": context.anchor_ids if context else (),
                    # Only lifecycle controls affect canonical snapshots. Other
                    # controls apply to copied batches or model inputs.
                    "reference_policy": ReferencePolicy(
                        persistent_anchors=self.reference_policy.persistent_anchors
                        if context
                        else True
                    ).digest,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if digest in self._snapshots:
            return self._snapshots[digest]
        root = self.cache_root / digest if self.cache_root else None
        if root:
            root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(root / "build.lock")):
                if not (root / "metadata.json").exists():
                    snapshot = build_snapshots(
                        rec,
                        self.representation_cfg,
                        anchor_ids=context.anchor_ids if context else (),
                        retention_seconds=self.retention_seconds,
                        persistent_anchors=self.reference_policy.persistent_anchors,
                        max_snapshot_bytes=self.max_snapshot_bytes,
                    )
                    for name, value in vars(snapshot).items():
                        if isinstance(value, np.ndarray):
                            temp = root / f"{name}.tmp"
                            with temp.open("wb") as handle:
                                np.save(handle, value)
                            temp.replace(root / f"{name}.npy")
                    temp = root / "metadata.tmp"
                    temp.write_text(
                        json.dumps(
                            {
                                "ids": snapshot.ids,
                                "arrays": [
                                    name
                                    for name, value in vars(snapshot).items()
                                    if isinstance(value, np.ndarray)
                                ],
                            }
                        )
                    )
                    temp.replace(root / "metadata.json")
                meta = json.loads((root / "metadata.json").read_text())
                snapshot = SnapshotRecording(
                    rec,
                    tuple(meta["ids"]),
                    **{
                        name: np.load(root / f"{name}.npy", mmap_mode="r")
                        for name in meta["arrays"]
                    },
                )
        else:
            snapshot = build_snapshots(
                rec,
                self.representation_cfg,
                anchor_ids=context.anchor_ids if context else (),
                persistent_anchors=self.reference_policy.persistent_anchors,
                retention_seconds=self.retention_seconds,
                max_snapshot_bytes=self.max_snapshot_bytes,
            )
        self._snapshots[digest] = snapshot
        return snapshot

    def _entries(self, role, *, training, forced_mode=None):
        entries = []
        for rec in sorted(self.records.values(), key=lambda r: r.key):
            if rec.role != role:
                continue
            context = self.context_for(rec.key)
            modes = (
                (forced_mode,)
                if forced_mode
                else (
                    ("conditioned", "zero-shot")
                    if not training and context
                    else ("zero-shot",)
                )
            )
            size = len(self.snapshot_for(rec.key, None).closes)
            if training:
                modes = ("training",)
            for mode in modes:
                if mode == "conditioned" and context is None:
                    raise ValueError(
                        f"conditioned evaluation requires a disjoint reference for {rec.key}"
                    )
                chunk = self.sequence_bins if training else self.eval_chunk_bins
                for cutoff in range(0, size, chunk):
                    start = max(0, cutoff - self.warmup_bins) if training else cutoff
                    selected_mode = mode
                    if training:
                        value = (
                            int(
                                hashlib.sha256(
                                    f"{self.seed}:{rec.key}:{cutoff}".encode()
                                ).hexdigest()[:8],
                                16,
                            )
                            / 2**32
                        )
                        selected_mode = (
                            "conditioned"
                            if context and value >= self.context_withholding
                            else "zero-shot"
                        )
                    entries.append(
                        (
                            rec.key,
                            start,
                            min(size, cutoff + chunk),
                            cutoff - start,
                            selected_mode,
                        )
                    )
        return entries

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_snapshots"] = {}  # workers reopen read-only memmaps lazily
        state["trainer"] = None  # Lightning owns iterators/CUDA state: never pickle it
        return state

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device, non_blocking=self.pin_memory)

    def _loader_options(self):
        options = dict(num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.num_workers:
            options.update(
                persistent_workers=True,
                prefetch_factor=2,
                multiprocessing_context="spawn",
            )
        return options

    def collate(self, items):
        batch = collate_snapshots(items)
        work = batch.live.numel()
        messages = batch.edge_mask.sum().item() * batch.x.shape[1]
        if work > self.max_node_bin_tokens or messages > self.max_graph_messages:
            raise ValueError(
                f"combined stream batch exceeds work budgets: {work} node-bins, {messages} messages"
            )
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.stream_batch_size,
            sampler=self.train_sampler,
            **self._loader_options(),
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=1,
            **self._loader_options(),
            sampler=RecordingEvalSampler(self.val_data),
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        if not self.test_data:
            raise ValueError("test requires declared target query recordings")
        return [
            DataLoader(
                data,
                batch_size=1,
                **self._loader_options(),
                sampler=RecordingEvalSampler(data),
                collate_fn=self.collate,
            )
            for data in self.test_data.values()
        ]
