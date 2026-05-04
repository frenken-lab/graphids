"""End-to-end wiring: setup → prebatch → loss reads difficulty + in_scope.

This is the integration test for steps 1-5 combined: the per-graph
attributes attached at ``setup`` time must survive prebatching and
reach the curriculum loss as ``batch.difficulty`` / ``batch.in_scope``.
Without step 5's prebatch wiring fix, the attrs were stranded on
``_train_graphs`` and never reached the dataloader.
"""

from __future__ import annotations

import torch
from conftest import make_graph
from torch_geometric.data import Batch

from graphids.core.data.datamodule.graph import GraphDataModule
from graphids.core.data.preprocessing.curriculum import LinearRampSchedule
from graphids.core.losses import CrossEntropyLoss, CurriculumWeightedLoss


def score_synthetic(graphs, *, base: float = 0.0) -> torch.Tensor:
    return torch.tensor([base + i for i in range(len(graphs))], dtype=torch.float)


class _FakeDataset:
    def __init__(self, train, val=(), test=None):
        self._state = _FakeState(train, val, test or {})
        self.cache_key = f"fake-{id(self)}"
        self.name = "fake"

    def build(self):
        return self._state


class _FakeState:
    def __init__(self, train, val, test):
        # Preserve input types — train may be a wrapped list with
        # num_nodes_per_graph / num_edges_per_graph attributes that the
        # prebatch path reads.
        self.train = train
        self.val = val
        self.test = test


def _make_train(n_normal: int, n_attack: int) -> list:
    out = []
    for _ in range(n_normal):
        g = make_graph()
        g.y = torch.tensor([0])
        out.append(g)
    for _ in range(n_attack):
        g = make_graph()
        g.y = torch.tensor([1])
        out.append(g)
    return out


def _wrap_with_num_per_graph(train_list):
    """Attach `num_nodes_per_graph` / `num_edges_per_graph` so the prebatch
    path's size lookups succeed without the real preprocessing layer."""
    sizes = torch.tensor([g.num_nodes for g in train_list], dtype=torch.long)
    edge_sizes = torch.tensor([int(g.num_edges) for g in train_list], dtype=torch.long)
    # Wrap as a list-like with the required attrs.
    class _Wrapped(list):
        pass
    w = _Wrapped(train_list)
    w.num_nodes_per_graph = sizes
    w.num_edges_per_graph = edge_sizes
    return w


def test_prebatched_train_carries_curriculum_attrs():
    # CONTRACT: when curriculum is configured, every batch yielded by the
    # train dataloader has batch.difficulty and batch.in_scope populated.
    # This is the wire-format guarantee the loss depends on.
    train = _wrap_with_num_per_graph(_make_train(n_normal=8, n_attack=2))
    ds = _FakeDataset(train)
    dm = GraphDataModule(
        ds,
        num_workers=0,
        dynamic_batching=False,  # avoid the budget probe (no CUDA in tests)
        difficulty={
            "class_path": f"{__name__}.score_synthetic",
            "init_args": {"base": 5.0},
        },
        scope_label=0,
    )
    # dynamic_batching=False uses _build_train_loader; force the prebatch
    # path explicitly via _hp toggle.
    dm._hp["dynamic_batching"] = True
    dm.setup("fit")

    # Drive the prebatch path. Need a stub for _ensure_budget since CUDA isn't
    # available — the prebatch path calls it via self.trainer.lightning_module.
    from graphids.core.budget import BudgetResult
    dm._ensure_budget = lambda: BudgetResult(
        budget=10_000, edge_budget=20_000, binding="opted_in_fallback",
        backward_multiplier=None, t_fwd=0.0, target_bytes=0,
    )

    loader = dm.train_dataloader()
    batches = list(loader)
    assert len(batches) > 0
    for b in batches:
        assert hasattr(b, "difficulty"), "prebatched batch missing difficulty"
        assert hasattr(b, "in_scope"), "prebatched batch missing in_scope"
        assert b.difficulty.shape == (b.num_graphs,)
        assert b.in_scope.shape == (b.num_graphs,)
        # Difficulty values should be from score_synthetic (5.0 + index).
        assert b.difficulty.min().item() >= 5.0
        assert b.difficulty.max().item() <= 5.0 + 9


def test_curriculum_loss_consumes_prebatched_attrs():
    # END-TO-END: an actual forward pass through the curriculum loss using
    # a batch produced by the datamodule. Catches any drift between what
    # setup attaches and what the loss reads.
    train = _wrap_with_num_per_graph(_make_train(n_normal=6, n_attack=2))
    ds = _FakeDataset(train)
    dm = GraphDataModule(
        ds,
        num_workers=0,
        dynamic_batching=False,
        difficulty={
            "class_path": f"{__name__}.score_synthetic",
            "init_args": {},
        },
        scope_label=0,
    )
    dm.setup("fit")
    # Build a batch the way the prebatch loader will produce one.
    batch = Batch.from_data_list(dm._train_graphs)
    logits = torch.randn(batch.num_graphs, 2, requires_grad=True)

    sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
    loss = CurriculumWeightedLoss(CrossEntropyLoss(reduction="none"), sched)
    loss.set_epoch(0)
    out = loss(logits, batch.y, batch)
    out.backward()
    assert out.shape == ()
    # At epoch 0: 1 of 6 in-scope normals unlocked + 2 out-of-scope attacks
    # always on → 3 examples receive nonzero gradient.
    nonzero = (logits.grad.abs().sum(dim=1) > 0).sum().item()
    assert nonzero == 3, f"expected 3 nonzero-grad rows (1 normal + 2 attacks), got {nonzero}"
