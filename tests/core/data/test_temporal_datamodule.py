from __future__ import annotations

import torch
from torch_geometric.data import TemporalData

from graphids.core.data.datamodule.temporal import TemporalDataModule
from graphids.core.data.state import clear_cache


def _temporal(labels: list[int]) -> TemporalData:
    n = len(labels)
    ids = torch.arange(n, dtype=torch.long)
    return TemporalData(
        src=ids,
        dst=ids + 1,
        t=torch.arange(n, dtype=torch.float32),
        msg=torch.randn(n, 4),
        y=torch.tensor(labels, dtype=torch.long),
        attack_type=torch.tensor(labels, dtype=torch.long),
        stream_id=torch.zeros(n, dtype=torch.long),
        reset_after=torch.zeros(n, dtype=torch.bool),
        event_id=ids,
        is_scored=torch.ones(n, dtype=torch.bool),
    )


def _multi_stream_temporal() -> TemporalData:
    return TemporalData(
        src=torch.tensor([0, 1, 1, 2, 2, 3, 4], dtype=torch.long),
        dst=torch.tensor([1, 1, 2, 2, 3, 4, 5], dtype=torch.long),
        t=torch.arange(7, dtype=torch.float32),
        msg=torch.arange(28, dtype=torch.float32).view(7, 4),
        y=torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long),
        attack_type=torch.tensor([0, 2, 0, 2, 0, 2, 0], dtype=torch.long),
        stream_id=torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.long),
        reset_after=torch.tensor([False, False, True, False, True, False, True]),
        event_id=torch.arange(7, dtype=torch.long),
        is_scored=torch.tensor([True, True, False, True, True, True, True]),
    )


class _Source:
    cache_key = "temporal-dm-test"
    attack_type_names = {0: "benign", 2: "fuzzing"}

    def build(self):
        return type(
            "State",
            (),
            {
                "train": _temporal([0, 1, 0]),
                "val": _temporal([0, 1]),
                "test": {"holdout": _temporal([0, 1, 1])},
            },
        )()


def test_temporal_datamodule_exposes_event_schema_and_named_tests():
    clear_cache()
    dm = TemporalDataModule(_Source(), batch_size=2)
    dm.setup(None)

    assert dm.in_channels == 4
    assert dm.num_ids == 4
    assert dm.num_classes == 2
    assert list(dm.test_data) == ["holdout"]
    assert list(dm.test_datasets) == ["holdout"]
    assert dm.attack_type_names == {0: "benign", 2: "fuzzing"}

    batch = next(iter(dm.train_dataloader()))
    assert batch.y.numel() == 2
    assert tuple(batch.msg.shape) == (2, 4)

    test_loaders = dm.test_dataloader()
    assert len(test_loaders) == 1
    assert sum(int(batch.y.numel()) for batch in test_loaders[0]) == 3


def test_temporal_datamodule_passes_temporal_loader_resource_kwargs(monkeypatch):
    clear_cache()
    captured = []

    class _FakeTemporalDataLoader:
        def __init__(self, data, **kwargs):
            self.data = data
            self.kwargs = kwargs
            captured.append(kwargs)

    monkeypatch.setattr("torch_geometric.loader.TemporalDataLoader", _FakeTemporalDataLoader)
    dm = TemporalDataModule(
        _Source(),
        batch_size=7,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
    )
    dm.setup(None)

    loader = dm.train_dataloader()

    assert loader.kwargs == {
        "batch_size": 7,
        "num_workers": 3,
        "pin_memory": True,
        "persistent_workers": True,
    }
    assert captured == [loader.kwargs]


def test_temporal_stream_lane_loader_preserves_stream_order_and_events_once():
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    loader = TemporalStreamLaneLoader(_multi_stream_temporal(), stream_lanes=2, chunk_size=2)
    batches = list(loader)

    seen = torch.cat([batch.event_id[batch.valid_mask] for batch in batches]).tolist()
    assert sorted(seen) == list(range(7))

    by_stream = {}
    for batch in batches:
        for sid, event_id in zip(batch.stream_id[batch.valid_mask].tolist(), batch.event_id[batch.valid_mask].tolist(), strict=True):
            by_stream.setdefault(sid, []).append(event_id)
    assert by_stream == {0: [0, 1, 2], 1: [3, 4], 2: [5, 6]}


def test_temporal_stream_lane_loader_masks_padding_and_marks_lane_resets():
    from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneLoader

    batches = list(TemporalStreamLaneLoader(_multi_stream_temporal(), stream_lanes=2, chunk_size=2))

    assert batches[0].lane_reset.tolist() == [True, True]
    assert batches[1].lane_reset.tolist() == [False, True]
    assert batches[-1].valid_mask.tolist() == [[True, False], [True, True]]
    assert batches[-1].is_scored[batches[-1].valid_mask.logical_not()].any().item() is False


def test_temporal_datamodule_stream_lanes_only_changes_train_loader():
    clear_cache()
    dm = TemporalDataModule(_Source(), batch_mode="stream_lanes", stream_lanes=2, chunk_size=2)
    dm.setup(None)

    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    assert tuple(train_batch.valid_mask.shape) == (2, 2)
    assert val_batch.y.ndim == 1
