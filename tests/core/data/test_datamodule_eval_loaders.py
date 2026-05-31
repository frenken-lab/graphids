"""GraphDataModule val/test loader invariants.

REGRESSION: CPU test jobs crashed because ``test_dataloader`` called
``_ensure_budget`` which requires CUDA for a backward-pass probe. CPU
eval must continue to work without touching the budget path — the
fallback fixed-batch loader is the legitimate path there.

(GPU eval, by contrast, now reuses the train-time probe to dynamically
pack val/test batches. Tested by the val-throughput SLURM smoke, not
here — these tests are explicitly login-node / CPU-partition.)
"""

from __future__ import annotations

from conftest import make_graph

from graphids.core.data.datamodule.graph import GraphDataModule


class _FakeDataset:
    """Minimal Dataset-protocol stand-in: holds pre-built train/val/test lists."""

    def __init__(self, train, val, test):
        self._state = _FakeState(train, val, test)
        self.cache_key = f"fake-{id(self)}"
        self.name = "fake"

    def build(self):
        return self._state


class _FakeState:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test


def _make_dm():
    train = [make_graph() for _ in range(4)]
    val = [make_graph() for _ in range(3)]
    test_a = [make_graph() for _ in range(2)]
    test_b = [make_graph() for _ in range(2)]
    ds = _FakeDataset(train, val, {"a": test_a, "b": test_b})
    # num_workers=0 avoids spawn overhead for an import-only test.
    dm = GraphDataModule(ds, batch_size=2, num_workers=0, dynamic_batching=True)
    dm.setup("test")
    return dm


class TestEvalLoadersCpuOnly:
    """INVARIANT: val / test loaders build on CPU without a wired model.

    ``_ensure_budget`` requires CUDA + a model. On the CPU branch
    (``torch.cuda.is_available() == False``) the eval loader must skip
    the probe and use fixed batches — model is intentionally unwired
    (``bind`` not called) to assert the CPU path does not reach
    ``_ensure_budget``.
    """

    def test_test_dataloader_builds_without_cuda_or_model(self):
        dm = _make_dm()
        loaders = dm.test_dataloader()
        assert len(loaders) == 2
        # Loader must iterate — if the budget probe were on the path,
        # it would have raised before returning.
        assert sum(1 for _ in loaders[0]) > 0

    def test_val_dataloader_builds_without_cuda_or_model(self):
        dm = _make_dm()
        loader = dm.val_dataloader()
        assert sum(1 for _ in loader) > 0

    def test_eval_loaders_do_not_populate_budget(self):
        """CONTRACT: eval path must not call ``_ensure_budget``."""
        dm = _make_dm()
        dm.test_dataloader()
        dm.val_dataloader()
        assert dm._budget is None


def test_require_cache_fails_before_building():
    import pytest

    class MissingCacheSource:
        name = "dummy"

        def cache_ready(self):
            return False

        def cache_root_path(self):
            return "/tmp/missing-cache"

        def build(self):
            raise AssertionError("build should not run when require_cache=True")

    dm = GraphDataModule(MissingCacheSource(), require_cache=True)

    with pytest.raises(RuntimeError, match="required graph cache is missing"):
        dm.setup(None)
