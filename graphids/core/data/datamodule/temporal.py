"""Lightning data module for temporal PyG event streams."""

from __future__ import annotations

import lightning.pytorch as pl

from graphids.core.data.state import get_or_build


class TemporalDataModule(pl.LightningDataModule):
    """Serve temporal event streams with PyG's TemporalDataLoader."""

    def __init__(self, dataset, batch_size: int = 256):
        super().__init__()
        self.source = dataset
        self.batch_size = batch_size
        self._train = None
        self._val = None
        self._tests: dict[str, object] = {}

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        st = get_or_build(self.source)
        self._train, self._val, self._tests = st.train, st.val, st.test

    def _loader(self, data):
        from torch_geometric.loader import TemporalDataLoader

        return TemporalDataLoader(data, batch_size=self.batch_size)

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

    def train_dataloader(self):
        return self._loader(self._train)

    def val_dataloader(self):
        return self._loader(self._val)

    def test_dataloader(self):
        if self._tests:
            return [self._loader(td) for td in self._tests.values()]
        return self.val_dataloader()
