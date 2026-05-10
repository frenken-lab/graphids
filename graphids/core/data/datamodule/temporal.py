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

    def train_dataloader(self):
        return self._loader(self._train)

    def val_dataloader(self):
        return self._loader(self._val)

    def test_dataloader(self):
        if self._tests:
            return [self._loader(td) for td in self._tests.values()]
        return self.val_dataloader()
