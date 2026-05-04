"""Curriculum-batching DataModule.

Subclass of :class:`GraphDataModule` that scores graphs via a configured
scorer at setup time, buckets normals into difficulty tiers (attacks form
a tier of their own that's always active), and pre-batches each tier
independently. A :class:`graphids.core.data.curriculum.CurriculumEpochCallback`
selects active tiers each epoch.

Per-tier independent packing is load-bearing for the budget contract:
batches are concatenated from pre-packed tier batches (no re-packing), so
the system-wide VRAM peak stays bounded by a single-batch budget. Adding
a cross-tier packer would require re-running the probe against the mixed
population.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from .graph import GraphDataModule, _prebatched_loader


class CurriculumDataModule(GraphDataModule):
    def __init__(
        self,
        dataset,
        *,
        scorer: dict | None = None,  # {class_path, init_args}
        curriculum_start_ratio: float = 1.0,
        curriculum_end_ratio: float = 10.0,
        max_epochs: int = 300,
        num_tiers: int = 10,
        **graph_kwargs,
    ):
        super().__init__(dataset, **graph_kwargs)
        self._hp.update(
            scorer=scorer,
            curriculum_start_ratio=curriculum_start_ratio,
            curriculum_end_ratio=curriculum_end_ratio,
            max_epochs=max_epochs,
            num_tiers=num_tiers,
        )
        self._tier_graphs: list[list] | None = None
        self._tier_sizes: list[torch.Tensor] | None = None
        self._tier_edge_sizes: list[torch.Tensor] | None = None
        self._tier_batches: list[list[Batch]] | None = None
        self._active_batches: list[Batch] | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return
        super().setup(stage)
        self._setup_curriculum()

    def train_dataloader(self):
        if self._tier_batches is None:
            self._tier_batches = [
                self._prebatch(graphs, sizes, edge_sizes)
                for graphs, sizes, edge_sizes in zip(
                    self._tier_graphs, self._tier_sizes, self._tier_edge_sizes
                )
            ]
            self._select_active_tiers(0)
        return _prebatched_loader(
            self._active_batches,
            shuffle=True,
            device=self._prefetch_device(),
        )

    def _setup_curriculum(self) -> None:
        from graphids.core.data.curriculum import build_curriculum_tiers, make_scorer

        hp = self._hp
        scorer = make_scorer(hp["scorer"])
        scores, normal_tiers, attack_indices, full_dataset, dataset_sizes = build_curriculum_tiers(
            self._train_ds,
            scorer,
            num_tiers=hp["num_tiers"],
        )
        dataset_edge_sizes = torch.tensor(
            [int(g.num_edges) for g in full_dataset], dtype=torch.long
        )
        self._tier_graphs = []
        self._tier_sizes = []
        self._tier_edge_sizes = []
        for tier_idx in normal_tiers:
            if not tier_idx:
                continue  # empty tier — scorer may produce this at the extremes
            self._tier_graphs.append([full_dataset[i] for i in tier_idx])
            self._tier_sizes.append(dataset_sizes[tier_idx])
            self._tier_edge_sizes.append(dataset_edge_sizes[tier_idx])
        if attack_indices:
            self._tier_graphs.append([full_dataset[i] for i in attack_indices])
            self._tier_sizes.append(dataset_sizes[attack_indices])
            self._tier_edge_sizes.append(dataset_edge_sizes[attack_indices])

        # Invariant: every stored tier has graphs AND its sizes aligned.
        assert len(self._tier_graphs) == len(self._tier_sizes) == len(self._tier_edge_sizes), (
            f"tier bookkeeping mismatch: {len(self._tier_graphs)} graphs / "
            f"{len(self._tier_sizes)} node-sizes / {len(self._tier_edge_sizes)} edge-sizes"
        )
        for i, (g, s, e) in enumerate(
            zip(self._tier_graphs, self._tier_sizes, self._tier_edge_sizes)
        ):
            assert len(g) == len(s) == len(e) > 0, f"tier {i} empty or length-mismatched"

    def _select_active_tiers(self, epoch: int) -> None:
        """Assemble ``self._active_batches`` for ``epoch``.

        Tier 0 = easiest, last tier = attacks (always active).
        """
        from graphids.core.data.curriculum import active_tier_count

        hp = self._hp
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
