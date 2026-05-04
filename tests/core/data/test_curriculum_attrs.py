"""GraphDataModule attaches difficulty + in_scope when curriculum is configured.

Step 1 of the curriculum-primitives loss-masking redesign: the datamodule
materializes train graphs, calls a free-function difficulty scorer, and
attaches per-graph ``difficulty`` + ``in_scope`` attributes that PyG
collates automatically into ``batch.difficulty`` / ``batch.in_scope``.
"""

from __future__ import annotations

import pytest
import torch
from conftest import make_graph
from torch_geometric.data import Batch

from graphids.core.data.datamodule.graph import GraphDataModule


def score_synthetic(graphs, *, base: float = 0.0) -> torch.Tensor:
    """Test-only free-function scorer. Score = base + index, deterministic."""
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
        self.train = list(train)
        self.val = list(val)
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


_DIFFICULTY_SPEC = {
    "class_path": "test_curriculum_attrs.score_synthetic",
    "init_args": {"base": 10.0},
}


class TestCurriculumAttrs:
    def test_setup_populates_difficulty_and_scope(self):
        # CONTRACT: setup attaches difficulty + in_scope on every train graph.
        ds = _FakeDataset(_make_train(n_normal=4, n_attack=2))
        dm = GraphDataModule(
            ds,
            num_workers=0,
            difficulty=_DIFFICULTY_SPEC,
            scope_label=0,
        )
        dm.setup("fit")
        assert dm._train_graphs is not None
        assert len(dm._train_graphs) == 6
        for i, g in enumerate(dm._train_graphs):
            assert hasattr(g, "difficulty")
            assert hasattr(g, "in_scope")
            assert g.difficulty.shape == (1,)
            assert g.in_scope.shape == (1,)
            assert g.difficulty.item() == 10.0 + i

    def test_in_scope_matches_label(self):
        # CONTRACT: in_scope[i] == (y[i] == scope_label).
        ds = _FakeDataset(_make_train(n_normal=3, n_attack=2))
        dm = GraphDataModule(ds, num_workers=0, difficulty=_DIFFICULTY_SPEC, scope_label=0)
        dm.setup("fit")
        flags = [bool(g.in_scope.item()) for g in dm._train_graphs]
        assert flags == [True, True, True, False, False]

    def test_collation_produces_batch_attributes(self):
        # CONTRACT: Batch.from_data_list collates difficulty + in_scope into
        # per-graph tensors of shape [N_graphs] — this is the wire format the
        # curriculum-weighted loss will read at training-step time.
        ds = _FakeDataset(_make_train(n_normal=4, n_attack=1))
        dm = GraphDataModule(ds, num_workers=0, difficulty=_DIFFICULTY_SPEC, scope_label=0)
        dm.setup("fit")
        batch = Batch.from_data_list(dm._train_graphs[:3])
        assert batch.difficulty.shape == (3,)
        assert batch.in_scope.shape == (3,)
        assert torch.allclose(batch.difficulty, torch.tensor([10.0, 11.0, 12.0]))
        assert batch.in_scope.tolist() == [True, True, True]

    def test_curriculum_off_leaves_attrs_none(self):
        # INVARIANT: without a difficulty config, no curriculum work runs.
        ds = _FakeDataset(_make_train(n_normal=3, n_attack=1))
        dm = GraphDataModule(ds, num_workers=0)
        dm.setup("fit")
        assert dm._train_graphs is None
        assert dm._train_difficulty is None
        assert dm._train_in_scope is None

    def test_label_filter_with_difficulty_rejected(self):
        # REGRESSION: label_filter='benign' would drop attacks from the train
        # view, which contradicts curriculum's need for the full label
        # distribution to define scope. Fail fast.
        ds = _FakeDataset(_make_train(n_normal=3, n_attack=1))
        dm = GraphDataModule(
            ds,
            num_workers=0,
            label_filter="benign",
            difficulty=_DIFFICULTY_SPEC,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            dm.setup("fit")

    def test_score_length_mismatch_rejected(self):
        # CONTRACT: scorer must return one score per graph.
        def bad_scorer(graphs):
            return torch.zeros(len(graphs) - 1)

        # Resolve via class_path so we exercise the real importlib path.
        import sys

        module = sys.modules[__name__]
        module.bad_scorer = bad_scorer

        ds = _FakeDataset(_make_train(n_normal=3, n_attack=0))
        dm = GraphDataModule(
            ds,
            num_workers=0,
            difficulty={"class_path": f"{__name__}.bad_scorer", "init_args": {}},
        )
        with pytest.raises(ValueError, match="returned 2 scores for 3"):
            dm.setup("fit")
