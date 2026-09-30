"""Lane-batched temporal event iteration."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class TemporalStreamLaneBatch:
    """A fixed-lane batch of independent temporal streams.

    Event tensors are shaped ``[lanes, chunk_size, ...]``. ``valid_mask`` marks
    real events; padded slots carry neutral values and must not affect loss or
    metrics.
    """

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

    def to(self, device: torch.device | str, *args: Any, **kwargs: Any) -> TemporalStreamLaneBatch:
        values = {
            field: getattr(self, field).to(device, *args, **kwargs)
            for field in self.__dataclass_fields__
        }
        return type(self)(**values)

    def pin_memory(self) -> TemporalStreamLaneBatch:
        try:
            values = {field: getattr(self, field).pin_memory() for field in self.__dataclass_fields__}
        except RuntimeError:
            return self
        return type(self)(**values)

    @property
    def num_valid_events(self) -> int:
        return int(self.valid_mask.sum().item())


@dataclass(frozen=True)
class _LaneBatchItem:
    lane_idx: int
    stream_idx: int
    offset: int
    end: int
    lane_reset: bool
    stream_end: bool


@dataclass(frozen=True)
class _LaneBatchPlan:
    items: tuple[_LaneBatchItem, ...]


class TemporalStreamLaneLoader(Dataset[TemporalStreamLaneBatch]):
    """Map-style dataset that packs independent streams into fixed temporal lanes."""

    def __init__(self, data, *, stream_lanes: int, chunk_size: int):
        self.data = data
        self.stream_lanes = int(stream_lanes)
        self.chunk_size = int(chunk_size)
        if self.stream_lanes < 2:
            raise ValueError("stream_lanes must be >= 2 for batch_mode='stream_lanes'")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive for batch_mode='stream_lanes'")
        self._streams = self._partition_streams(data)
        self._tensors = self._event_tensors()
        self._plans = tuple(self._iter_plans())

    @staticmethod
    def _partition_streams(data) -> list[torch.Tensor]:
        n = int(data.dst.numel())
        stream_id = getattr(data, "stream_id", None)
        if stream_id is None:
            stream_id = torch.zeros(n, dtype=torch.long)
        streams: OrderedDict[int, list[int]] = OrderedDict()
        for event_idx, sid in enumerate(stream_id.detach().cpu().tolist()):
            streams.setdefault(int(sid), []).append(event_idx)
        return [torch.tensor(indices, dtype=torch.long) for indices in streams.values()]

    def __len__(self) -> int:
        return len(self._plans)

    @staticmethod
    def _event_tensor(data, name: str, default: torch.Tensor) -> torch.Tensor:
        value = getattr(data, name, None)
        return default if value is None else value

    def _empty_batch(self) -> TemporalStreamLaneBatch:
        lanes = self.stream_lanes
        chunk = self.chunk_size
        n = int(self.data.dst.numel())
        device = self.data.dst.device
        msg_shape = tuple(self.data.msg.shape[1:])
        t_default = torch.zeros(n, dtype=self.data.msg.dtype, device=device)
        attack_default = torch.zeros(n, dtype=torch.long, device=device)
        event_default = torch.arange(n, dtype=torch.long, device=device)

        return TemporalStreamLaneBatch(
            src=torch.zeros((lanes, chunk), dtype=self.data.src.dtype, device=device),
            dst=torch.zeros((lanes, chunk), dtype=self.data.dst.dtype, device=device),
            t=torch.zeros((lanes, chunk), dtype=t_default.dtype, device=device),
            msg=torch.zeros((lanes, chunk, *msg_shape), dtype=self.data.msg.dtype, device=device),
            y=torch.zeros((lanes, chunk), dtype=self.data.y.dtype, device=device),
            attack_type=torch.zeros((lanes, chunk), dtype=attack_default.dtype, device=device),
            event_id=torch.full((lanes, chunk), -1, dtype=event_default.dtype, device=device),
            is_scored=torch.zeros((lanes, chunk), dtype=torch.bool, device=device),
            stream_id=torch.full((lanes, chunk), -1, dtype=torch.long, device=device),
            valid_mask=torch.zeros((lanes, chunk), dtype=torch.bool, device=device),
            lane_reset=torch.zeros(lanes, dtype=torch.bool, device=device),
            lane_stream_end=torch.zeros(lanes, dtype=torch.bool, device=device),
            reset_after=torch.zeros((lanes, chunk), dtype=torch.bool, device=device),
        )

    def _event_tensors(self) -> dict[str, torch.Tensor]:
        n = int(self.data.dst.numel())
        return {
            "t": self._event_tensor(
                self.data,
                "t",
                torch.zeros(n, dtype=self.data.msg.dtype, device=self.data.dst.device),
            ),
            "attack_type": self._event_tensor(
                self.data,
                "attack_type",
                torch.zeros(n, dtype=torch.long, device=self.data.dst.device),
            ),
            "event_id": self._event_tensor(
                self.data,
                "event_id",
                torch.arange(n, dtype=torch.long, device=self.data.dst.device),
            ),
            "is_scored": self._event_tensor(
                self.data,
                "is_scored",
                torch.ones(n, dtype=torch.bool, device=self.data.dst.device),
            ),
            "stream_id": self._event_tensor(
                self.data,
                "stream_id",
                torch.zeros(n, dtype=torch.long, device=self.data.dst.device),
            ),
            "reset_after": self._event_tensor(
                self.data,
                "reset_after",
                torch.zeros(n, dtype=torch.bool, device=self.data.dst.device),
            ),
        }

    def _iter_plans(self) -> Iterator[_LaneBatchPlan]:
        active: list[tuple[int, int] | None] = [None] * self.stream_lanes
        next_stream = 0

        while next_stream < len(self._streams) or any(slot is not None for slot in active):
            lane_resets = [False] * self.stream_lanes
            for lane_idx, slot in enumerate(active):
                if slot is None and next_stream < len(self._streams):
                    active[lane_idx] = (next_stream, 0)
                    lane_resets[lane_idx] = True
                    next_stream += 1

            items: list[_LaneBatchItem] = []
            for lane_idx, slot in enumerate(active):
                if slot is None:
                    continue
                stream_idx, offset = slot
                stream = self._streams[stream_idx]
                end = min(offset + self.chunk_size, int(stream.numel()))
                if end <= offset:
                    active[lane_idx] = None
                    continue
                stream_end = end >= int(stream.numel())
                items.append(
                    _LaneBatchItem(
                        lane_idx=lane_idx,
                        stream_idx=stream_idx,
                        offset=offset,
                        end=end,
                        lane_reset=lane_resets[lane_idx],
                        stream_end=stream_end,
                    )
                )
                active[lane_idx] = None if stream_end else (stream_idx, end)

            yield _LaneBatchPlan(tuple(items))

    def _batch_from_plan(
        self,
        plan: _LaneBatchPlan,
        tensors: dict[str, torch.Tensor],
    ) -> TemporalStreamLaneBatch:
        batch = self._empty_batch()
        for item in plan.items:
            stream = self._streams[item.stream_idx]
            take = stream[item.offset:item.end].to(device=self.data.dst.device)
            width = int(take.numel())
            if width == 0:
                continue

            lane_idx = item.lane_idx
            sl = slice(0, width)
            batch.lane_reset[lane_idx] = item.lane_reset
            batch.src[lane_idx, sl] = self.data.src[take]
            batch.dst[lane_idx, sl] = self.data.dst[take]
            batch.t[lane_idx, sl] = tensors["t"][take]
            batch.msg[lane_idx, sl] = self.data.msg[take]
            batch.y[lane_idx, sl] = self.data.y[take]
            batch.attack_type[lane_idx, sl] = tensors["attack_type"][take]
            batch.event_id[lane_idx, sl] = tensors["event_id"][take]
            batch.is_scored[lane_idx, sl] = tensors["is_scored"][take].bool()
            batch.stream_id[lane_idx, sl] = tensors["stream_id"][take].long()
            batch.reset_after[lane_idx, sl] = tensors["reset_after"][take].bool()
            batch.valid_mask[lane_idx, sl] = True

            if item.stream_end:
                batch.lane_stream_end[lane_idx] = True
                batch.reset_after[lane_idx, width - 1] = True
        return batch

    def __getitem__(self, index: int) -> TemporalStreamLaneBatch:
        return self._batch_from_plan(self._plans[index], self._tensors)

    def __iter__(self) -> Iterator[TemporalStreamLaneBatch]:
        for idx in range(len(self)):
            yield self[idx]
