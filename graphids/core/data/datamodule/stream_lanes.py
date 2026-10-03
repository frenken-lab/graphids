"""Ordered lane batches with recording-owned labels and optional section replay."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class TemporalStreamLaneBatch:
    """Padding never advances state. Sections reset independently of recordings."""

    src: torch.Tensor
    dst: torch.Tensor
    t: torch.Tensor
    msg: torch.Tensor
    y: torch.Tensor
    attack_type: torch.Tensor
    event_id: torch.Tensor
    is_scored: torch.Tensor
    stream_id: torch.Tensor
    valid_mask: torch.Tensor
    lane_reset: torch.Tensor
    lane_stream_end: torch.Tensor
    reset_after: torch.Tensor
    source_index: torch.Tensor | None = None
    successor_index: torch.Tensor | None = None
    successor_dst: torch.Tensor | None = None
    successor_msg: torch.Tensor | None = None
    lane_section_end: torch.Tensor | None = None
    warmup_mask: torch.Tensor | None = None

    def _map(self, fn):
        return type(self)(**{
            field: None if getattr(self, field) is None else fn(getattr(self, field))
            for field in self.__dataclass_fields__
        })

    def to(self, device: torch.device | str, *args: Any, **kwargs: Any) -> TemporalStreamLaneBatch:
        return self._map(lambda value: value.to(device, *args, **kwargs))

    def pin_memory(self) -> TemporalStreamLaneBatch:
        try:
            return self._map(lambda value: value.pin_memory())
        except RuntimeError:
            return self

    @property
    def num_valid_events(self) -> int:
        return int(self.valid_mask.sum().item())


@dataclass(frozen=True)
class StreamDescriptor:
    """A slice for cached sorted data, or stable tensor indices for interleaving."""

    stream_id: int
    start: int
    length: int
    indices: torch.Tensor | None = None

    def take(self, start: int, end: int):
        if self.indices is None:
            return slice(self.start + start, self.start + end)
        return self.indices[start:end]

    def source_indices(self, start: int, end: int):
        if self.indices is None:
            return torch.arange(self.start + start, self.start + end)
        return self.indices[start:end]


@dataclass(frozen=True)
class SectionDescriptor:
    stream_idx: int
    warmup_start: int
    core_start: int
    core_end: int


@dataclass(frozen=True)
class _LaneBatchItem:
    lane_idx: int
    section_idx: int
    offset: int
    end: int
    lane_reset: bool
    section_end: bool


@dataclass(frozen=True)
class _LaneBatchPlan:
    items: tuple[_LaneBatchItem, ...]


class TemporalStreamLaneLoader(Dataset[TemporalStreamLaneBatch]):
    """Map-style ordered plans: workers materialize batches without owning state.

    Continuous recordings retain lane state between chunks (gradients detach).
    Experimental sections own disjoint cores, replay preceding unscored context,
    and may occupy multiple lanes for the same parent recording.
    """

    def __init__(self, data, *, stream_lanes: int, chunk_size: int,
                 drop_partial_batches: bool = False, stream_execution: str = "continuous",
                 segment_events: int = 2048, segment_warmup_events: int = 256,
                 seed: int = 42):
        self.data = data
        self.stream_lanes = int(stream_lanes)
        self.chunk_size = int(chunk_size)
        self.drop_partial_batches = bool(drop_partial_batches)
        if self.stream_lanes < 2 or self.chunk_size < 1:
            raise ValueError("stream_lanes must be >= 2 and chunk_size must be positive")
        if stream_execution not in {"continuous", "segmented", "exact_replay"}:
            raise ValueError("stream_execution must be continuous, segmented, or exact_replay")
        if segment_events < 1 or segment_warmup_events < 0:
            raise ValueError("segment_events must be positive; warmup must be nonnegative")
        self._streams = self._partition_streams(data)
        sections = []
        for idx, stream in enumerate(self._streams):
            width = stream.length if stream_execution == "continuous" else segment_events
            for start in range(0, stream.length, width):
                sections.append(SectionDescriptor(
                    idx, 0 if stream_execution == "exact_replay" else max(0, start - segment_warmup_events), start,
                    min(start + width, stream.length),
                ))
        if stream_execution == "segmented":
            order = torch.randperm(len(sections), generator=torch.Generator().manual_seed(seed))
            sections = [sections[i] for i in order.tolist()]
        self.sections = tuple(sections)
        self._tensors = self._event_tensors()
        self.diagnostic_gaps = 0
        self._plans = tuple(self._filter_plans(self._iter_plans()))

    @staticmethod
    def _partition_streams(data) -> list[StreamDescriptor]:
        n = int(data.dst.numel())
        if not n:
            return []
        ids = getattr(data, "stream_id", None)
        if ids is None:
            return [StreamDescriptor(0, 0, n)]
        ids = ids.detach().cpu()
        runs, counts = torch.unique_consecutive(ids, return_counts=True)
        if runs.unique().numel() == runs.numel():
            result, offset = [], 0
            for sid, count in zip(runs.tolist(), counts.tolist(), strict=True):
                result.append(StreamDescriptor(sid, offset, count))
                offset += count
            return result
        # Stable grouping retains chronological input order without Python event lists.
        order = torch.argsort(ids, stable=True)
        runs, counts = torch.unique_consecutive(ids[order], return_counts=True)
        result, offset = [], 0
        for sid, count in zip(runs.tolist(), counts.tolist(), strict=True):
            result.append(StreamDescriptor(sid, 0, count, order[offset:offset + count]))
            offset += count
        return result

    def __len__(self) -> int:
        return len(self._plans)

    def _event_tensors(self):
        # Defaults are scalar/batch-sized, never dataset-sized temporary tensors.
        return {name: getattr(self.data, name, None) for name in
                ("src", "dst", "t", "msg", "y", "attack_type", "event_id", "is_scored", "stream_id", "reset_after")}

    def _empty_batch(self):
        shape = (self.stream_lanes, self.chunk_size)
        device = self.data.dst.device
        def zeros(dtype=torch.long, tail=()):
            return torch.zeros((*shape, *tail), dtype=dtype, device=device)
        return TemporalStreamLaneBatch(
            src=zeros(self.data.src.dtype), dst=zeros(self.data.dst.dtype),
            t=zeros(self.data.t.dtype if self._tensors["t"] is not None else self.data.msg.dtype),
            msg=zeros(self.data.msg.dtype, self.data.msg.shape[1:]), y=zeros(self.data.y.dtype),
            attack_type=zeros(), event_id=zeros().fill_(-1), is_scored=zeros(torch.bool),
            stream_id=zeros().fill_(-1), valid_mask=zeros(torch.bool),
            lane_reset=torch.zeros(self.stream_lanes, dtype=torch.bool, device=device),
            lane_stream_end=torch.zeros(self.stream_lanes, dtype=torch.bool, device=device),
            reset_after=zeros(torch.bool), source_index=zeros().fill_(-1),
            successor_index=zeros().fill_(-1), successor_dst=zeros(self.data.dst.dtype),
            successor_msg=zeros(self.data.msg.dtype, self.data.msg.shape[1:]),
            lane_section_end=torch.zeros(self.stream_lanes, dtype=torch.bool, device=device),
            warmup_mask=zeros(torch.bool),
        )

    def _iter_plans(self) -> Iterator[_LaneBatchPlan]:
        active = [None] * self.stream_lanes
        next_section = 0
        while next_section < len(self.sections) or any(slot is not None for slot in active):
            resets = [False] * self.stream_lanes
            for lane, slot in enumerate(active):
                if slot is None and next_section < len(self.sections):
                    section = self.sections[next_section]
                    active[lane] = (next_section, section.warmup_start)
                    resets[lane] = True
                    next_section += 1
            items = []
            for lane, slot in enumerate(active):
                if slot is None:
                    continue
                idx, offset = slot
                section = self.sections[idx]
                # Replay ends at a batch boundary even for non-aligned settings,
                # so detached state prevents gradients crossing into replay.
                boundary = section.core_start if offset < section.core_start else section.core_end
                end = min(offset + self.chunk_size, boundary)
                done = end == section.core_end
                items.append(_LaneBatchItem(lane, idx, offset, end, resets[lane], done))
                active[lane] = None if done else (idx, end)
            yield _LaneBatchPlan(tuple(items))

    def _filter_plans(self, plans):
        skipped = set()
        for plan in plans:
            if self.drop_partial_batches and sum(i.end - i.offset for i in plan.items) != self.stream_lanes * self.chunk_size:
                skipped.update(i.lane_idx for i in plan.items)
                self.diagnostic_gaps += sum(i.end - i.offset for i in plan.items)
                continue
            yield _LaneBatchPlan(tuple(replace(i, lane_reset=True) if i.lane_idx in skipped else i for i in plan.items))
            skipped.difference_update(i.lane_idx for i in plan.items)

    def __getitem__(self, index):
        batch = self._empty_batch()
        for item in self._plans[index].items:
            section = self.sections[item.section_idx]
            stream = self._streams[section.stream_idx]
            take = stream.take(item.offset, item.end)
            if isinstance(take, torch.Tensor):
                take = take.to(self.data.dst.device)
            width = item.end - item.offset
            lane, sl = item.lane_idx, slice(0, width)
            source = stream.source_indices(item.offset, item.end).to(self.data.dst.device)
            for name, tensor in self._tensors.items():
                if tensor is not None:
                    getattr(batch, name)[lane, sl] = tensor[take]
            batch.source_index[lane, sl] = source
            if self._tensors["event_id"] is None:
                batch.event_id[lane, sl] = source
            batch.stream_id[lane, sl] = stream.stream_id
            batch.valid_mask[lane, sl] = True
            warmup = torch.arange(item.offset, item.end, device=source.device) < section.core_start
            batch.warmup_mask[lane, sl] = warmup
            batch.is_scored[lane, sl] = ~warmup
            batch.lane_reset[lane] = item.lane_reset
            successor_end = min(item.end, stream.length - 1)
            successor_width = max(0, successor_end - item.offset)
            if successor_width:
                nxt = stream.take(item.offset + 1, successor_end + 1)
                if isinstance(nxt, torch.Tensor):
                    nxt = nxt.to(source.device)
                target_sl = slice(0, successor_width)
                batch.successor_index[lane, target_sl] = stream.source_indices(item.offset + 1, successor_end + 1).to(source.device)
                batch.successor_dst[lane, target_sl] = self.data.dst[nxt]
                batch.successor_msg[lane, target_sl] = self.data.msg[nxt]
            # Evaluation's scoring warmup is already marked in source data.
            if self._tensors["is_scored"] is not None:
                batch.is_scored[lane, sl] &= self._tensors["is_scored"][take].bool()
            if item.section_end:
                batch.lane_section_end[lane] = True
                batch.lane_stream_end[lane] = item.end == stream.length
                batch.reset_after[lane, width - 1] = True
        return batch

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
