"""Dataset-agnostic graph DataModule.

Accepts any object with a ``cache_key: str`` + ``build() -> DatasetState``
(the ``Dataset`` protocol consumed by ``graphids.core.data.cache``).
Preprocessing and split logic live in the dataset class; the datamodule
only wraps DataLoaders, batching, and the curriculum toggle.

Curriculum learning is a ``sampler="curriculum"`` toggle. When enabled,
``setup()`` scores graphs via VGAE and buckets normals into difficulty
tiers. A ``CurriculumEpochCallback`` selects active tiers each epoch.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, InMemoryDataset

from graphids.core.data.budget import autosize_workers, node_budget
from graphids.core.data.cache import get_or_build
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


class GraphDataModule:
    """Graph DataModule that wraps a Dataset source.

    ``dataset`` is any object satisfying the Dataset protocol consumed
    by ``graphids.core.data.cache.get_or_build``: ``cache_key: str`` +
    ``build() -> DatasetState``. The datamodule owns loader/batching
    policy; the dataset owns preprocessing and splits.

    Curriculum toggle:
        Pass ``sampler="curriculum"`` + ``vgae_ckpt_path`` to enable
        VGAE-scored difficulty gating. Requires a
        ``CurriculumEpochCallback`` in the trainer's callback list.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        dynamic_batching: bool = True,
        conv_type: str = "gatv2",
        heads: int = 4,
        # --- label-scope toggle ---
        # "benign": restrict train loader to y == 0 graphs (unsupervised
        #   reconstruction stages — VGAE/DGI — must see normal traffic only;
        #   attack rows pollute the reconstruction prior).
        # None: full train set (supervised stages).
        label_filter: str | None = None,
        # --- curriculum toggle ---
        sampler: str = "standard",
        scorer: dict | None = None,  # {class_path, init_args} — see core.data.curriculum
        curriculum_start_ratio: float = 1.0,
        curriculum_end_ratio: float = 10.0,
        max_epochs: int = 300,
        num_tiers: int = 10,
    ):
        self.dataset = dataset
        # Store init args as a dict for downstream kwargs access.
        self.hparams = {k: v for k, v in locals().items() if k != "self"}
        self._train_ds: InMemoryDataset | None = None
        self._val_ds: InMemoryDataset | None = None
        self._test_datasets: dict[str, InMemoryDataset] = {}
        self._prebatched_train: list[Batch] | None = None
        # Curriculum: populated by _setup_curriculum, pre-batched on first epoch
        self._tier_graphs: list[list] | None = None  # per-tier graph lists
        self._tier_sizes: list[torch.Tensor] | None = None  # per-tier node counts
        self._tier_batches: list[list[Batch]] | None = None  # pre-batched tiers
        self._active_batches: list[Batch] | None = None
        # Device for PrefetchLoader — set by trainer via _set_device()
        self._device: torch.device | None = None
        # Model handle — set by trainer via _set_model() so _budget_result
        # can run a real probe (otherwise node_budget falls back to a static bpn).
        self._model = None

    def _set_device(self, device: torch.device | None) -> None:
        """Called by Trainer to tell the datamodule which device to prefetch to."""
        self._device = device

    def _set_model(self, model) -> None:
        """Called by Trainer so dynamic-batching probes can measure this model."""
        self._model = model

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return
        state = get_or_build(self.dataset)
        self._train_ds = state.train
        self._val_ds = state.val
        self._test_datasets = state.test
        if self.hparams["sampler"] == "curriculum":
            self._setup_curriculum()

    def _setup_curriculum(self) -> None:
        """Score graphs via the configured strategy, bucket into tiers + attack tier.

        Batching divergence from the dynamic-batching path: each tier is
        pre-packed INDEPENDENTLY (``_curriculum_train_dataloader`` calls
        ``_prebatch`` once per tier). The node/edge budgets returned by
        the probe apply *per tier*, not across tiers. Because
        ``_active_batches`` is built by concatenating pre-packed tier
        batches (no re-packing), the system-wide VRAM peak is still
        bounded by a single-batch budget — the contract the callback /
        prebatch guard depend on.

        DO NOT add a cross-tier packer (e.g. "mix tiers in a single
        batch" for gradient smoothing) without revisiting memory sizing:
        packing across tiers would require a fresh probe because the
        budget result was computed against tier-slice graph populations.
        """
        from graphids.core.data.curriculum import build_curriculum_tiers, make_scorer

        hp = self.hparams
        scorer = make_scorer(hp["scorer"])
        scores, normal_tiers, attack_indices, full_dataset, dataset_sizes = build_curriculum_tiers(
            self._train_ds,
            scorer,
            num_tiers=hp.get("num_tiers", 10),
        )
        # Build per-tier graph lists + size tensors for _prebatch
        self._tier_graphs = []
        self._tier_sizes = []
        for tier_idx in normal_tiers:
            if not tier_idx:
                continue  # empty tier — scorer may produce this at the extremes
            self._tier_graphs.append([full_dataset[i] for i in tier_idx])
            self._tier_sizes.append(dataset_sizes[tier_idx])
        # Attack tier (always active)
        if attack_indices:
            self._tier_graphs.append([full_dataset[i] for i in attack_indices])
            self._tier_sizes.append(dataset_sizes[attack_indices])

        # Invariant: every stored tier has graphs AND its sizes aligned.
        # Empty tiers or mismatched lengths would build a silently dead
        # dataloader entry; fail loud so callers see it at setup() time.
        assert len(self._tier_graphs) == len(self._tier_sizes), (
            f"tier bookkeeping mismatch: "
            f"{len(self._tier_graphs)} graph lists vs "
            f"{len(self._tier_sizes)} size tensors"
        )
        for i, (g, s) in enumerate(zip(self._tier_graphs, self._tier_sizes)):
            assert len(g) == len(s) > 0, f"tier {i} is empty or length-mismatched"

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
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return max(2, int(ds._data.y.unique().numel()))

    @property
    def edge_dim(self) -> int:
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].edge_attr.shape[1]

    # -- Shared helpers -------------------------------------------------------

    def _budget_result(self, dataset=None):
        """Probe VRAM budget for the given dataset (or train_ds).

        When ``_set_model`` has supplied a model AND CUDA is available,
        ``node_budget`` runs a one-shot fwd+bwd profile to derive the
        per-node / per-edge byte cost. Otherwise it falls back to the
        static ``_FALLBACK_BPN`` constant (binding="fallback").
        """
        hp = self.hparams
        return node_budget(
            self.dataset.name,
            self.dataset.resolved_lake_root(),
            conv_type=hp.get("conv_type", "gatv2"),
            heads=hp.get("heads", 4),
            model=self._model,
            train_dataset=dataset or self._train_ds,
        )

    def _prebatch(self, graphs, sizes, edge_sizes=None) -> list[Batch]:
        """Pre-collate graphs into Batches via first-fit-decreasing packing.

        Uses ``pack_offline`` instead of the live sampler: prebatch doesn't
        need epoch-to-epoch randomness (``_prebatched_loader`` shuffles
        batch order separately), so FFD's tighter packing is a pure win.

        Invariant: every emitted plan stays within the probed (node, edge)
        envelope. Asserted here so a future packer regression surfaces
        immediately instead of waiting for a CUDA OOM.
        """
        result = self._budget_result(graphs)
        if edge_sizes is None:
            edge_sizes = torch.tensor(
                [int(g.num_edges) for g in graphs],
                dtype=torch.long,
            )
        plans = pack_offline(
            sizes,
            max_num=result.budget,
            edge_sizes=edge_sizes,
            max_edges=result.edge_budget,
            skip_too_big=True,
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
        return self._device if torch.cuda.is_available() else None

    # -- DataLoaders ----------------------------------------------------------

    def _effective_train_ds(self):
        """Train dataset with ``label_filter`` applied (view over _train_ds).

        For VGAE/DGI reconstruction stages, ``label_filter="benign"`` drops
        y != 0 graphs from the training view. val/test loaders see the
        unfiltered splits. The subset is a PyG index_select view — cheap
        to construct, no tensor copies.
        """
        ds = self._train_ds
        if self.hparams.get("label_filter") != "benign":
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
        if self.hparams["sampler"] == "curriculum":
            return self._curriculum_train_dataloader()
        if self.hparams["dynamic_batching"]:
            return self._prebatched_train_dataloader()
        return self._build_loader(self._effective_train_ds(), shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self._val_ds, shuffle=False)

    def test_dataloader(self):
        return [self._build_loader(ds, shuffle=False) for ds in self._test_datasets.values()]

    def _build_loader(self, dataset, shuffle: bool):
        hp = self.hparams
        nw = hp.get("num_workers")
        pf = hp.get("prefetch_factor", 2)
        device = self._prefetch_device()

        if not hp["dynamic_batching"]:
            return _spawn_loader(
                dataset,
                batch_size=max(8, hp["batch_size"]),
                shuffle=shuffle,
                num_workers=nw if nw is not None else 2,
                device=device,
                prefetch_factor=pf,
            )

        result = self._budget_result(dataset)
        if nw is None:
            nw, pf = autosize_workers(self._model, dataset, result, default_prefetch=pf)

        sampler = NodeBudgetBatchSampler(
            dataset.num_nodes_per_graph,
            max_num=result.budget,
            edge_sizes=dataset.num_edges_per_graph,
            max_edges=result.edge_budget,
            shuffle=shuffle,
            skip_too_big=True,
        )
        return _spawn_loader(
            dataset,
            batch_sampler=sampler,
            num_workers=nw,
            device=device,
            prefetch_factor=pf,
        )

    def _prebatched_train_dataloader(self):
        """Pre-batched training: collate once, shuffle batch order per epoch."""
        if self._prebatched_train is None:
            train_ds = self._effective_train_ds()
            self._prebatched_train = self._prebatch(
                train_ds,
                train_ds.num_nodes_per_graph,
            )
        return _prebatched_loader(
            self._prebatched_train,
            shuffle=True,
            device=self._prefetch_device(),
        )

    def _curriculum_train_dataloader(self):
        """Tier-based curriculum: pre-batch each tier once, select active per epoch."""
        if self._tier_batches is None:
            self._tier_batches = [
                self._prebatch(graphs, sizes)
                for graphs, sizes in zip(self._tier_graphs, self._tier_sizes)
            ]
            self._select_active_tiers(0)
        return _prebatched_loader(
            self._active_batches,
            shuffle=True,
            device=self._prefetch_device(),
        )

    def _select_active_tiers(self, epoch: int) -> None:
        """Assemble ``self._active_batches`` for ``epoch``.

        Tier 0 = easiest, last tier = attacks (always active). The active
        count comes from :func:`curriculum.active_tier_count`; this method
        only handles the concatenation.
        """
        from graphids.core.data.curriculum import active_tier_count

        hp = self.hparams
        n_normal = len(self._tier_batches) - 1  # last tier is attacks
        count = active_tier_count(
            epoch,
            n_normal,
            start_ratio=hp["curriculum_start_ratio"],
            end_ratio=hp["curriculum_end_ratio"],
            max_epochs=hp["max_epochs"],
        )
        active: list[Batch] = []
        for i in range(count):
            active.extend(self._tier_batches[i])
        active.extend(self._tier_batches[-1])  # attacks always active
        self._active_batches = active
