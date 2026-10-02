"""Lightning data module for temporal PyG event streams."""

from __future__ import annotations

import lightning.pytorch as pl

from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader
from graphids.core.data.state import get_or_build


def _identity_collate(batch):
    return batch


class TemporalDataModule(pl.LightningDataModule):
    """Serve temporal event streams with PyG's TemporalDataLoader."""

    def __init__(
        self,
        dataset,
        batch_size: int = 256,
        *,
        batch_mode: str = "events",
        stream_lanes: int | None = None,
        chunk_size: int | None = None,
        num_workers: int = 0,
        prebuild_batches: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_partial_lane_batches: bool = False,
        coverage_policy: str = "full",
        stream_execution: str = "continuous",
        segment_events: int = 2048,
        segment_warmup_events: int = 256,
        eval_batch_mode: str = "events",
        val_prefix_events: int | None = None,
    ):
        super().__init__()
        self.source = dataset
        self.batch_size = batch_size
        self.batch_mode = str(batch_mode)
        self.stream_lanes = None if stream_lanes is None else int(stream_lanes)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.num_workers = int(num_workers)
        self.prebuild_batches = int(prebuild_batches)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.drop_partial_lane_batches = bool(drop_partial_lane_batches)
        if self.batch_mode not in {"events", "stream_lanes"}:
            raise ValueError("batch_mode must be one of: events, stream_lanes")
        if self.batch_mode == "stream_lanes":
            if self.stream_lanes is None or self.stream_lanes < 2:
                raise ValueError("stream_lanes must be >= 2 for batch_mode='stream_lanes'")
            if self.chunk_size is None or self.chunk_size < 1:
                raise ValueError("chunk_size must be positive for batch_mode='stream_lanes'")
        self.coverage_policy = coverage_policy
        self.stream_execution = stream_execution
        self.segment_events = segment_events
        self.segment_warmup_events = segment_warmup_events
        self.eval_batch_mode = eval_batch_mode
        self.val_prefix_events = val_prefix_events
        if getattr(dataset, "val_split", None) == "recordings" and val_prefix_events is not None:
            raise ValueError("recording validation requires complete streams; remove val_prefix_events")
        if coverage_policy not in {"full", "diagnostic"}:
            raise ValueError("coverage_policy must be full or diagnostic")
        if stream_execution not in {"continuous", "segmented"}:
            raise ValueError("stream_execution must be continuous or segmented")
        if eval_batch_mode not in {"events", "stream_lanes"}:
            raise ValueError("eval_batch_mode must be events or stream_lanes")
        if stream_execution == "segmented" and batch_mode != "stream_lanes":
            raise ValueError("segmented execution requires stream_lanes")
        if eval_batch_mode == "stream_lanes" and (stream_lanes is None or chunk_size is None):
            raise ValueError("lane evaluation requires stream_lanes and chunk_size")
        if coverage_policy == "full" and batch_mode == "stream_lanes" and drop_partial_lane_batches:
            raise ValueError("full coverage requires drop_partial_lane_batches: false")
        self._train_lane_dataset = None
        self._train = None
        self._val = None
        self._tests: dict[str, object] = {}

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        st = get_or_build(self.source)
        self._train, self._val, self._tests = st.train, st.val, st.test
        if self.val_prefix_events is not None:
            import torch
            streams = TemporalStreamLaneLoader._partition_streams(self._val)
            indices = torch.cat([s.source_indices(0, min(s.length, self.val_prefix_events)) for s in streams])
            self._val = self._val[indices.to(self._val.dst.device)]
            # Prefix truncation creates new true split terminals.
            for stream in TemporalStreamLaneLoader._partition_streams(self._val):
                self._val.reset_after[stream.source_indices(stream.length - 1, stream.length)] = True

    def _loader(self, data):
        from torch_geometric.loader import TemporalDataLoader

        kwargs = self._torch_loader_kwargs()
        return TemporalDataLoader(data, batch_size=self.batch_size, **kwargs)

    def _torch_loader_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prebuild_batches > 0:
                kwargs["prefetch_factor"] = self.prebuild_batches
        return kwargs

    def _all_data(self) -> list[object]:
        data = [d for d in (self._train, self._val) if d is not None]
        data.extend(self._tests.values())
        return data

    @staticmethod
    def _max_tensor_value(data, key: str) -> int:
        tensor = getattr(data, key, None)
        if tensor is None or tensor.numel() == 0:
            return 0
        return int(tensor.max().item())

    @property
    def train_data(self):
        assert self._train is not None
        return self._train

    @property
    def val_data(self):
        assert self._val is not None
        return self._val

    @property
    def test_data(self) -> dict[str, object]:
        return self._tests

    @property
    def test_datasets(self) -> dict[str, object]:
        """Compatibility name used by model test-set bookkeeping."""
        return self._tests

    @property
    def attack_type_names(self) -> dict[int, str]:
        return dict(getattr(self.source, "attack_type_names", {0: "benign"}))

    @property
    def num_ids(self) -> int:
        """Embedding table size for temporal source/destination ids."""
        max_id = 0
        for data in self._all_data():
            max_id = max(max_id, self._max_tensor_value(data, "src"), self._max_tensor_value(data, "dst"))
        return max_id + 1

    @property
    def in_channels(self) -> int:
        if self._train is None:
            raise RuntimeError("TemporalDataModule.setup() must run before reading in_channels")
        return int(self._train.msg.shape[1])

    @property
    def num_classes(self) -> int:
        max_label = 0
        for data in self._all_data():
            max_label = max(max_label, self._max_tensor_value(data, "y"))
        return max(2, max_label + 1)

    def _lane_loader(self, data, *, training=False):
        from torch.utils.data import DataLoader

        if training and self._train_lane_dataset is not None:
            dataset = self._train_lane_dataset
        else:
            dataset = TemporalStreamLaneLoader(
                data, stream_lanes=self.stream_lanes, chunk_size=self.chunk_size,
                drop_partial_batches=self.drop_partial_lane_batches if training else False,
                stream_execution=self.stream_execution if training else "continuous",
                segment_events=self.segment_events, segment_warmup_events=self.segment_warmup_events,
            )
            if training:
                self._train_lane_dataset = dataset
        # Sequential sampler + ordered DataLoader results keep state independent of prefetch.
        return DataLoader(dataset, batch_size=None, collate_fn=_identity_collate,
                          shuffle=False, **self._torch_loader_kwargs())

    def train_dataloader(self):
        trainer = getattr(self, "trainer", None)
        if self.batch_mode == "stream_lanes" and self.coverage_policy == "full" and trainer is not None:
            from graphids.core.data.datamodule.coverage import TemporalCoverageAudit
            if not any(isinstance(cb, TemporalCoverageAudit) and cb.require_complete for cb in trainer.callbacks):
                trainer.callbacks.append(TemporalCoverageAudit())
        if self.batch_mode == "stream_lanes":
            return self._lane_loader(self._train, training=True)
        return self._loader(self._train)

    def val_dataloader(self):
        if self.eval_batch_mode == "stream_lanes":
            return self._lane_loader(self._val)
        return self._loader(self._val)

    def test_dataloader(self):
        if self._tests:
            loader = self._lane_loader if self.eval_batch_mode == "stream_lanes" else self._loader
            return [loader(td) for td in self._tests.values()]
        return self.val_dataloader()
