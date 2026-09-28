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
