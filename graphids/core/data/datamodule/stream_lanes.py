"""Lane-batched temporal event iteration."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch


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

    @property
    def num_valid_events(self) -> int:
        return int(self.valid_mask.sum().item())


class TemporalStreamLaneLoader:
    """Deterministically pack independent streams into fixed temporal lanes."""

    def __init__(self, data, *, stream_lanes: int, chunk_size: int):
        self.data = data
        self.stream_lanes = int(stream_lanes)
        self.chunk_size = int(chunk_size)
        if self.stream_lanes < 2:
            raise ValueError("stream_lanes must be >= 2 for batch_mode='stream_lanes'")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive for batch_mode='stream_lanes'")
        self._streams = self._partition_streams(data)
        self._length = self._compute_length()

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

    def _compute_length(self) -> int:
        active: list[tuple[int, int] | None] = [None] * self.stream_lanes
        next_stream = 0
        emitted = 0
        while next_stream < len(self._streams) or any(slot is not None for slot in active):
            for lane_idx, slot in enumerate(active):
                if slot is None and next_stream < len(self._streams):
                    active[lane_idx] = (next_stream, 0)
                    next_stream += 1
            for lane_idx, slot in enumerate(active):
                if slot is None:
                    continue
                stream_idx, offset = slot
                next_offset = offset + self.chunk_size
                active[lane_idx] = None if next_offset >= self._streams[stream_idx].numel() else (stream_idx, next_offset)
            emitted += 1
        return emitted

    def __len__(self) -> int:
        return self._length

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

    def __iter__(self) -> Iterator[TemporalStreamLaneBatch]:
        active: list[tuple[int, int] | None] = [None] * self.stream_lanes
        next_stream = 0

        t = self._event_tensor(
            self.data,
            "t",
            torch.zeros(int(self.data.dst.numel()), dtype=self.data.msg.dtype, device=self.data.dst.device),
        )
        attack_type = self._event_tensor(
            self.data,
            "attack_type",
            torch.zeros(int(self.data.dst.numel()), dtype=torch.long, device=self.data.dst.device),
        )
        event_id = self._event_tensor(
            self.data,
            "event_id",
            torch.arange(int(self.data.dst.numel()), dtype=torch.long, device=self.data.dst.device),
        )
        is_scored = self._event_tensor(
            self.data,
            "is_scored",
            torch.ones(int(self.data.dst.numel()), dtype=torch.bool, device=self.data.dst.device),
        )
        stream_ids = self._event_tensor(
            self.data,
            "stream_id",
            torch.zeros(int(self.data.dst.numel()), dtype=torch.long, device=self.data.dst.device),
        )
        reset_after = self._event_tensor(
            self.data,
            "reset_after",
            torch.zeros(int(self.data.dst.numel()), dtype=torch.bool, device=self.data.dst.device),
        )

        while next_stream < len(self._streams) or any(slot is not None for slot in active):
            batch = self._empty_batch()
            for lane_idx, slot in enumerate(active):
                if slot is None and next_stream < len(self._streams):
                    active[lane_idx] = (next_stream, 0)
                    batch.lane_reset[lane_idx] = True
                    next_stream += 1

            for lane_idx, slot in enumerate(active):
                if slot is None:
                    continue
                stream_idx, offset = slot
                stream = self._streams[stream_idx]
                end = min(offset + self.chunk_size, int(stream.numel()))
                take = stream[offset:end].to(device=self.data.dst.device)
                width = int(take.numel())
                if width == 0:
                    active[lane_idx] = None
                    batch.lane_stream_end[lane_idx] = True
                    continue

                sl = slice(0, width)
                batch.src[lane_idx, sl] = self.data.src[take]
                batch.dst[lane_idx, sl] = self.data.dst[take]
                batch.t[lane_idx, sl] = t[take]
                batch.msg[lane_idx, sl] = self.data.msg[take]
                batch.y[lane_idx, sl] = self.data.y[take]
                batch.attack_type[lane_idx, sl] = attack_type[take]
                batch.event_id[lane_idx, sl] = event_id[take]
                batch.is_scored[lane_idx, sl] = is_scored[take].bool()
                batch.stream_id[lane_idx, sl] = stream_ids[take].long()
                batch.reset_after[lane_idx, sl] = reset_after[take].bool()
                batch.valid_mask[lane_idx, sl] = True

                if end >= int(stream.numel()):
                    active[lane_idx] = None
                    batch.lane_stream_end[lane_idx] = True
                    batch.reset_after[lane_idx, width - 1] = True
                else:
                    active[lane_idx] = (stream_idx, end)

            yield batch
