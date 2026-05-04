"""GraphDataModule v2: minimal Lightning + PyG composition.

Built from the primitives we already pay for: ``pl.LightningDataModule``,
``torch_geometric.loader.DataLoader``, ``torch_geometric.loader.PrefetchLoader``,
``torch_geometric.data.Batch``. No private loader factories, no
``_loader_cache`` keyed on ``id()``, no fixed-vs-dynamic dual paths, no
unreachable live-sampler branch.

What the DM actually does:
1. ``setup`` → ``get_or_build`` → optional curriculum attribute attach.
2. Train: pre-pack via FFD (``pack_offline``) into a list of ``Batch``,
   shuffle batch order each epoch through a plain ``torch.utils.data.DataLoader``.
3. Val/Test on CUDA: same pre-pack against the trainer's budget probe.
4. Val/Test on CPU: PyG ``DataLoader`` at fixed ``batch_size``.
5. ``label_filter='benign'`` is a PyG ``ds[idx]`` index-select view.

The live ``NodeBudgetBatchSampler`` path is gone — it was unreachable
in v1 (always shadowed by the prebatch branch) and the prebatch packer
gives ~10-20% tighter batches anyway.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch, InMemoryDataset
from torch_geometric.loader import DataLoader, PrefetchLoader

from graphids.core.budget import node_budget
from graphids.core.data.datamodule.sampler import pack_offline
from graphids.core.data.state import get_or_build


def _file_system_worker(_id: int) -> None:
    """spawn workers must call set_sharing_strategy at module level (lambdas
    aren't picklable). Used by every PyG DataLoader below.
    """
    mp.set_sharing_strategy("file_system")


def _clone(x):
    """Picklable collate for prebatched lists under spawn."""
    return x.clone() if hasattr(x, "clone") else x


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        dynamic_batching: bool = True,
        label_filter: str | None = None,
        difficulty: dict | None = None,
        scope_label: int = 0,
    ):
        super().__init__()
        self.source = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dynamic_batching = dynamic_batching
        self.label_filter = label_filter
        self.difficulty = difficulty
        self.scope_label = scope_label
        self._train: InMemoryDataset | None = None
        self._val: InMemoryDataset | None = None
        self._tests: dict[str, InMemoryDataset] = {}
        self._train_graphs: list | None = None
        self._prebatched: list[Batch] | None = None
        self._budget = None
        self._autosize_info: dict | None = None

    # ── Lightning lifecycle ─────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        st = get_or_build(self.source)
        self._train, self._val, self._tests = st.train, st.val, st.test
        if self.difficulty is not None:
            self._attach_curriculum()

    def _attach_curriculum(self) -> None:
        import importlib

        if self.label_filter is not None:
            raise ValueError("label_filter and difficulty are mutually exclusive")
        spec = self.difficulty
        mod, _, fn = spec["class_path"].rpartition(".")
        score_fn = getattr(importlib.import_module(mod), fn)
        graphs = list(self._train)
        scores = score_fn(graphs, **spec.get("init_args", {}))
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float)
        if scores.numel() != len(graphs):
            raise ValueError(f"got {scores.numel()} scores for {len(graphs)} graphs")
        in_scope = torch.tensor(
            [int(g.y[0]) == int(self.scope_label) for g in graphs], dtype=torch.bool
        )
        for i, g in enumerate(graphs):
            g.difficulty = scores[i].view(1)
            g.in_scope = in_scope[i].view(1)
        self._train_graphs = graphs

    # ── Properties (after setup) ────────────────────────────────────────

    def _any_ds(self) -> InMemoryDataset:
        ds = self._train or next(iter(self._tests.values()), None)
        assert ds is not None, "call setup() first"
        return ds

    @property
    def train_dataset(self) -> InMemoryDataset:
        assert self._train is not None
        return self._train

    @property
    def val_dataset(self) -> InMemoryDataset:
        assert self._val is not None
        return self._val

    @property
    def test_datasets(self) -> dict[str, InMemoryDataset]:
        return self._tests

    @property
    def num_ids(self) -> int:
        return self._any_ds().num_ids

    @property
    def in_channels(self) -> int:
        return self._any_ds()[0].x.shape[1]

    @property
    def num_classes(self) -> int:
        return max(2, int(self._any_ds()._data.y.unique().numel()))

    @property
    def edge_dim(self) -> int:
        return self._any_ds()[0].edge_attr.shape[1]

    # ── Helpers ─────────────────────────────────────────────────────────

    def _device(self):
        if not torch.cuda.is_available() or self.trainer is None:
            return None
        return self.trainer.strategy.root_device

    def _budget_result(self):
        if self._budget is not None:
            return self._budget
        model = self.trainer.lightning_module if self.trainer is not None else None
        if torch.cuda.is_available() and model is None:
            raise RuntimeError("budget probe needs trainer + lightning_module")
        self._budget = (
            model.compute_budget(self._train, self.source.name)
            if model is not None
            else node_budget(self.source.name, train_dataset=self._train)
        )
        return self._budget

    def _train_view(self) -> InMemoryDataset:
        ds = self._train
        if self.label_filter != "benign":
            return ds
        full_y = ds._data.y.view(-1)
        y = full_y if ds._indices is None else full_y[torch.as_tensor(ds._indices)]
        idx = (y == 0).nonzero(as_tuple=False).flatten().tolist()
        if not idx:
            raise RuntimeError("label_filter='benign' yielded empty train set")
        return ds[idx]

    def _pack(self, graphs, sizes, edge_sizes) -> list[Batch]:
        b = self._budget_result()
        plans = pack_offline(sizes, max_num=b.budget, edge_sizes=edge_sizes, max_edges=b.edge_budget)
        return [Batch.from_data_list([graphs[i] for i in p]) for p in plans]

    def _wrap(self, loader):
        dev = self._device()
        return PrefetchLoader(loader, device=dev) if dev is not None else loader

    def _fixed_loader(self, ds, *, shuffle: bool):
        nw = self.num_workers if self.num_workers is not None else 2
        kw = dict(num_workers=nw, pin_memory=self._device() is None)
        if nw > 0:
            kw.update(
                persistent_workers=True,
                multiprocessing_context="spawn",
                worker_init_fn=_file_system_worker,
                prefetch_factor=self.prefetch_factor,
            )
        return self._wrap(DataLoader(ds, batch_size=max(8, self.batch_size), shuffle=shuffle, **kw))

    def _prebatched_dl(self, batches: list[Batch], *, shuffle: bool):
        return self._wrap(
            TorchDataLoader(batches, batch_size=None, shuffle=shuffle, collate_fn=_clone)
        )

    # ── DataLoader hooks ────────────────────────────────────────────────

    def train_dataloader(self):
        view = self._train_view()
        if not self.dynamic_batching:
            return self._fixed_loader(view, shuffle=True)
        if self._prebatched is None:
            graphs = self._train_graphs if self._train_graphs is not None else view
            self._prebatched = self._pack(graphs, view.num_nodes_per_graph, view.num_edges_per_graph)
        return self._prebatched_dl(self._prebatched, shuffle=True)

    def val_dataloader(self):
        return self._eval_loader(self._val)

    def test_dataloader(self):
        return [self._eval_loader(ds) for ds in self._tests.values()]

    def train_eval_dataloader(self):
        """Train split with eval-style fixed-batch loader (no probe). Used
        by SVDD calibration / centroid stats — batch-boundary-invariant.
        """
        return self._fixed_loader(self._train_view(), shuffle=False)

    def _eval_loader(self, ds):
        if self.dynamic_batching and torch.cuda.is_available():
            return self._prebatched_dl(
                self._pack(ds, ds.num_nodes_per_graph, ds.num_edges_per_graph),
                shuffle=False,
            )
        return self._fixed_loader(ds, shuffle=False)
