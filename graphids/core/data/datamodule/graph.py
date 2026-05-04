"""Dataset-agnostic graph DataModule.

Accepts any object with a ``cache_key: str`` + ``build() -> DatasetState``
(the ``Dataset`` protocol consumed by ``graphids.core.data.state``).
Preprocessing and split logic live in the dataset class; the datamodule
only wraps DataLoaders and batching.

Curriculum learning lives in :class:`graphids.core.data.datamodule.curriculum.CurriculumDataModule`
— a subclass that owns scorer/tier state. Plain ``GraphDataModule`` has
no curriculum branches.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch_geometric.data import Batch, InMemoryDataset

from graphids.core.budget import autosize_workers, node_budget
from graphids.core.data.state import get_or_build
from graphids.core.data.sampler import NodeBudgetBatchSampler, pack_offline


def _prefetch(loader, device: torch.device | None):
    if device is not None:
        from torch_geometric.loader import PrefetchLoader

        return PrefetchLoader(loader, device=device)
    return loader


def _worker_init_file_system(_worker_id: int) -> None:
    # Module-level — local lambdas can't be pickled for ``spawn`` workers.
    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")


def _clone_collate(x):
    # Module-level — prebatched loaders need a picklable collate under spawn.
    return x.clone() if hasattr(x, "clone") else x


def _spawn_loader(
    dataset,
    *,
    batch_size=1,
    batch_sampler=None,
    shuffle=False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    device: torch.device | None = None,
):
    """PyGDataLoader with spawn/persistent_workers defaults + PrefetchLoader."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    kw: dict = dict(num_workers=num_workers, pin_memory=device is None)
    if num_workers > 0:
        kw.update(
            persistent_workers=True,
            multiprocessing_context="spawn",
            worker_init_fn=_worker_init_file_system,
            prefetch_factor=prefetch_factor,
        )
    if batch_sampler is not None:
        loader = PyGDataLoader(dataset, batch_sampler=batch_sampler, **kw)
    else:
        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kw)
    return _prefetch(loader, device)


def _prebatched_loader(batches, *, shuffle: bool = True, device: torch.device | None = None):
    """TorchDataLoader for pre-collated Batch lists + PrefetchLoader."""
    from torch.utils.data import DataLoader as TorchDataLoader

    loader = TorchDataLoader(
        batches,
        batch_size=None,
        shuffle=shuffle,
        collate_fn=_clone_collate,
    )
    return _prefetch(loader, device)


class GraphDataModule(pl.LightningDataModule):
    """Graph DataModule that wraps a Dataset source.

    ``dataset`` is any object satisfying the Dataset protocol consumed
    by ``graphids.core.data.state.get_or_build``: ``cache_key: str`` +
    ``build() -> DatasetState``. The datamodule owns loader/batching
    policy; the dataset owns preprocessing and splits.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        dynamic_batching: bool = True,
        # --- label-scope toggle ---
        # "benign": restrict train loader to y == 0 graphs (unsupervised
        #   reconstruction stages — VGAE/DGI — must see normal traffic only;
        #   attack rows pollute the reconstruction prior).
        # None: full train set (supervised stages).
        label_filter: str | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        # Init args dict for downstream kwargs access. Stored under ``_hp``
        # (not ``hparams``) because ``pl.LightningDataModule.hparams`` is a
        # read-only property owned by Lightning's HyperparametersMixin.
        self._hp = {k: v for k, v in locals().items() if k != "self"}
        self._train_ds: InMemoryDataset | None = None
        self._val_ds: InMemoryDataset | None = None
        self._test_datasets: dict[str, InMemoryDataset] = {}
        self._prebatched_train: list[Batch] | None = None
        # Worker count actually used for the train loader (autosize result OR
        # explicit hp). Read at epoch 0 by
        # :class:`graphids._mlflow.MLflowTrainingCallback._stamp_run_config`
        # → ``params.graphids.{num_workers,prefetch_factor}`` +
        # ``tags.graphids.num_workers_source``. Without this dict the autosize
        # path is unfalsifiable post-hoc.
        self._autosize_info: dict | None = None
        # Memoize (dataset_id, shuffle) → loader; val/test used to rebuild
        # sampler + re-run probe + re-time workers every epoch.
        self._loader_cache: dict[tuple[int, bool], object] = {}

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return
        state = get_or_build(self.dataset)
        self._train_ds = state.train
        self._val_ds = state.val
        self._test_datasets = state.test

    # -- Properties (available after setup) -----------------------------------

    @property
    def train_dataset(self) -> InMemoryDataset:
        assert self._train_ds is not None, "call setup('fit') first"
        return self._train_ds

    @property
    def val_dataset(self) -> InMemoryDataset:
        assert self._val_ds is not None, "call setup('fit') first"
        return self._val_ds

    @property
    def test_datasets(self) -> dict[str, InMemoryDataset]:
        return self._test_datasets

    @property
    def num_ids(self) -> int:
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds.num_arb_ids

    @property
    def in_channels(self) -> int:
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].x.shape[1]

    @property
    def num_classes(self) -> int:
        # Clamp to 2: benign-only train splits (VGAE/DGI reconstruction stages)
        # have y.unique().numel() == 1, but downstream classifiers need at
        # least 2 logits to size their output heads.
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return max(2, int(ds._data.y.unique().numel()))

    @property
    def edge_dim(self) -> int:
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].edge_attr.shape[1]

    # -- Shared helpers -------------------------------------------------------

    def _ensure_budget(self):
        """Route to ``model.compute_budget`` — the model owns the probe.

        Model is read off ``self.trainer.lightning_module`` — Lightning sets
        ``self.trainer`` automatically when the DM is passed via
        ``trainer.fit(model, datamodule=dm)``. When CUDA is available the
        trainer must be present; otherwise the static-bpn fallback path
        runs without the actual conv_type/heads and underperforms by
        orders of magnitude.
        """
        model = self.trainer.lightning_module if self.trainer is not None else None
        if torch.cuda.is_available() and model is None:
            raise RuntimeError(
                "_ensure_budget called on a CUDA device without a wired trainer. "
                "Pass the DM via trainer.fit(model, datamodule=dm) so Lightning "
                "wires self.trainer before the first dataloader request."
            )
        if model is None:
            # CPU fallback path (tests / cache rebuild) — call node_budget
            # directly with the linear-probe defaults; no model means no probe
            # anyway, so conv_type/heads only affect the fallback formula.
            return node_budget(self.dataset.name, train_dataset=self._train_ds)
        return model.compute_budget(self._train_ds, self.dataset.name)

    def _prebatch(self, graphs, sizes, edge_sizes) -> list[Batch]:
        """Pre-collate graphs into Batches via first-fit-decreasing packing.

        Uses ``pack_offline`` instead of the live sampler: prebatch doesn't
        need epoch-to-epoch randomness (``_prebatched_loader`` shuffles
        batch order separately), so FFD's tighter packing is a pure win.

        Invariant: every emitted plan stays within the probed (node, edge)
        envelope. Asserted here so a future packer regression surfaces
        immediately instead of waiting for a CUDA OOM.

        Caller owns ``sizes`` / ``edge_sizes`` — both are required. The
        dataset exposes ``num_nodes_per_graph`` / ``num_edges_per_graph``
        as precomputed tensors; callers pass those directly.
        """
        result = self._ensure_budget()
        plans = pack_offline(
            sizes,
            max_num=result.budget,
            edge_sizes=edge_sizes,
            max_edges=result.edge_budget,
        )
        for plan in plans:
            tot_n = sum(int(sizes[i]) for i in plan)
            tot_e = sum(int(edge_sizes[i]) for i in plan)
            if tot_n > result.budget or (
                result.edge_budget is not None and tot_e > result.edge_budget
            ):
                raise RuntimeError(
                    "packer produced an oversized batch: "
                    f"nodes={tot_n}/{result.budget}, "
                    f"edges={tot_e}/{result.edge_budget}"
                )
        return [Batch.from_data_list([graphs[i] for i in plan]) for plan in plans]

    def _prefetch_device(self):
        if not torch.cuda.is_available() or self.trainer is None:
            return None
        return self.trainer.strategy.root_device

    # -- DataLoaders ----------------------------------------------------------

    def _effective_train_ds(self):
        """Train dataset with ``label_filter`` applied (view over _train_ds).

        For VGAE/DGI reconstruction stages, ``label_filter="benign"`` drops
        y != 0 graphs from the training view. val/test loaders see the
        unfiltered splits. The subset is a PyG index_select view — cheap
        to construct, no tensor copies.
        """
        ds = self._train_ds
        if self._hp["label_filter"] != "benign":
            return ds
        full_y = ds._data.y.view(-1)
        if ds._indices is None:
            y = full_y
        else:
            y = full_y[torch.as_tensor(ds._indices, dtype=torch.long)]
        benign_local = (y == 0).nonzero(as_tuple=False).flatten().tolist()
        if not benign_local:
            raise RuntimeError(
                "label_filter='benign' produced an empty training set — "
                "check that the train tensor contains y == 0 graphs."
            )
        return ds[benign_local]

    def train_dataloader(self):
        if self._hp["dynamic_batching"]:
            return self._prebatched_train_dataloader()
        return self._build_train_loader(self._effective_train_ds(), shuffle=True)

    def val_dataloader(self):
        return self._build_eval_loader(self._val_ds)

    def test_dataloader(self):
        return [self._build_eval_loader(ds) for ds in self._test_datasets.values()]

    def train_eval_dataloader(self):
        """Train split with eval-style fixed-batch loader (no budget probe).

        Used by SVDD calibration / centroid statistics that iterate the
        effective (label-filtered) train split once under no_grad. Bypassing
        dynamic batching means CPU test jobs without CUDA can still run the
        calibration — the budget probe is a training-throughput optimization
        and the centroid is batch-boundary-invariant.
        """
        return self._build_eval_loader(self._effective_train_ds())

    def _make_fixed_batch_loader(self, dataset, *, shuffle: bool):
        """Build a fixed-batch-size PyG loader from hparams. No budget probe.

        Shared by the non-dynamic-batching train path and every eval path
        (val/test) — both want the same hp-derived defaults (``batch_size``
        floor of 8, ``num_workers`` fallback of 2, prefetch factor + device
        from hp / ``_prefetch_device``). The dynamic-batching train path
        uses ``_spawn_loader`` directly with a ``batch_sampler``.
        """
        hp = self._hp
        nw = hp["num_workers"]
        return _spawn_loader(
            dataset,
            batch_size=max(8, hp["batch_size"]),
            shuffle=shuffle,
            num_workers=nw if nw is not None else 2,
            device=self._prefetch_device(),
            prefetch_factor=hp["prefetch_factor"],
        )

    def _build_train_loader(self, dataset, shuffle: bool):
        """Training loader. Uses dynamic batching (budget probe) when enabled."""
        key = (id(dataset), shuffle)
        cached = self._loader_cache.get(key)
        if cached is not None:
            return cached

        hp = self._hp
        if not hp["dynamic_batching"]:
            loader = self._make_fixed_batch_loader(dataset, shuffle=shuffle)
            self._loader_cache[key] = loader
            return loader

        nw = hp["num_workers"]
        pf = hp["prefetch_factor"]
        result = self._ensure_budget()
        if nw is None:
            model = self.trainer.lightning_module if self.trainer is not None else None
            nw, pf, diag = autosize_workers(model, dataset, result, default_prefetch=pf)
            self._autosize_info = {
                "num_workers": nw,
                "prefetch_factor": pf,
                "source": "autosize",
                **diag,
            }
        else:
            self._autosize_info = {"num_workers": nw, "prefetch_factor": pf, "source": "explicit"}
        # MLflow tagging of ``_autosize_info`` happens in
        # :class:`graphids._mlflow.MLflowTrainingCallback._stamp_run_config`
        # at epoch 0 — after the dataloader has been built but before any
        # training-step metrics are logged.

        sampler = NodeBudgetBatchSampler(
            dataset.num_nodes_per_graph,
            max_num=result.budget,
            edge_sizes=dataset.num_edges_per_graph,
            max_edges=result.edge_budget,
            shuffle=shuffle,
        )
        loader = _spawn_loader(
            dataset,
            batch_sampler=sampler,
            num_workers=nw,
            device=self._prefetch_device(),
            prefetch_factor=pf,
        )
        self._loader_cache[key] = loader
        return loader

    def _build_eval_loader(self, dataset):
        """Val / test loader.

        GPU path (``dynamic_batching=true`` AND CUDA available): reuses the
        train-time budget probe to dynamically pack val/test batches at the
        same VRAM-saturating granularity as training. Val runs every epoch
        during fit — without this, val pays ~36× more small forwards through
        a 2-worker NFS chain than train does, dragging GPU util down to
        ~30% on V100 (jid 47126749, set_01 seed=43, 30225-graph val set →
        ~945 batches/epoch at batch_size=32 vs ~26 at the probed budget).

        CPU path (no CUDA): falls back to fixed ``batch_size`` (defaulted
        to 32, ``num_workers`` to 2) — the budget probe needs CUDA + a
        wired model, which is what motivated commit a224f8c. SVDD
        calibration via ``train_eval_dataloader`` runs CPU-side test jobs
        and depends on this fallback.

        Per-example metrics (AUROC, accuracy) and centroid statistics are
        batch-boundary-invariant, so dynamic vs fixed batching only
        affects throughput, not numerics.
        """
        key = (id(dataset), False)
        cached = self._loader_cache.get(key)
        if cached is not None:
            return cached

        if self._hp["dynamic_batching"] and torch.cuda.is_available():
            prebatched = self._prebatch(
                dataset,
                dataset.num_nodes_per_graph,
                dataset.num_edges_per_graph,
            )
            loader = _prebatched_loader(
                prebatched,
                shuffle=False,
                device=self._prefetch_device(),
            )
        else:
            loader = self._make_fixed_batch_loader(dataset, shuffle=False)
        self._loader_cache[key] = loader
        return loader

    def _prebatched_train_dataloader(self):
        """Pre-batched training: collate once, shuffle batch order per epoch."""
        if self._prebatched_train is None:
            train_ds = self._effective_train_ds()
            self._prebatched_train = self._prebatch(
                train_ds,
                train_ds.num_nodes_per_graph,
                train_ds.num_edges_per_graph,
            )
        return _prebatched_loader(
            self._prebatched_train,
            shuffle=True,
            device=self._prefetch_device(),
        )

