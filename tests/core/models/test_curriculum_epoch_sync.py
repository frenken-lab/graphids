"""``_ModelBase.on_train_epoch_start`` forwards epoch to curriculum losses.

Step 4 of the curriculum-primitives loss-masking redesign. Lightning's
``self.current_epoch`` is read each epoch and pushed into ``loss_fn``
when the loss exposes ``set_epoch`` — duck-typed so non-curriculum
losses are a clean no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import lightning.pytorch as pl
import torch.nn as nn

from graphids.core.models.base import _ModelBase


class _NonCurriculumLoss(nn.Module):
    """Loss without ``set_epoch`` — must remain untouched by the hook."""

    def forward(self, *_, **__):
        raise NotImplementedError


class _CurriculumLoss(nn.Module):
    """Loss exposing ``set_epoch``. The base hook should call it."""

    def __init__(self):
        super().__init__()
        self.received_epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.received_epochs.append(int(epoch))

    def forward(self, *_, **__):
        raise NotImplementedError


def _bare_model_base() -> _ModelBase:
    """Bare _ModelBase instance — only the attributes the hook reads.

    We exercise ``on_train_epoch_start`` plus attribute reads, so we
    skip pl.LightningModule.__init__'s heavy state but DO need
    ``nn.Module.__init__`` so ``self.loss_fn = ...`` doesn't trip over
    Module's submodule registration.
    """
    obj = _ModelBase.__new__(_ModelBase)
    pl.LightningModule.__init__(obj)
    return obj


class TestEpochSync:
    def test_curriculum_loss_receives_epoch(self):
        # CONTRACT: loss.set_epoch(self.current_epoch) is called when the
        # loss exposes it. This is the load-bearing wiring for curriculum.
        m = _bare_model_base()
        m.loss_fn = _CurriculumLoss()
        # current_epoch is a Lightning property; on a bare instance we can
        # set it directly as a class-level shadow via __dict__.
        m._trainer = MagicMock(current_epoch=7)
        m.on_train_epoch_start()
        assert m.loss_fn.received_epochs == [7]

    def test_set_epoch_called_each_invocation(self):
        # INVARIANT: every epoch start forwards a fresh value.
        m = _bare_model_base()
        m.loss_fn = _CurriculumLoss()
        for e in [0, 1, 5, 9]:
            m._trainer = MagicMock(current_epoch=e)
            m.on_train_epoch_start()
        assert m.loss_fn.received_epochs == [0, 1, 5, 9]

    def test_non_curriculum_loss_untouched(self):
        # CONTRACT: losses without set_epoch are a clean no-op — non-
        # curriculum runs must not pay any cost or risk attribute errors.
        m = _bare_model_base()
        m.loss_fn = _NonCurriculumLoss()
        m._trainer = MagicMock(current_epoch=3)
        m.on_train_epoch_start()  # must not raise

    def test_no_loss_fn_attribute_no_op(self):
        # EDGE: model without a loss_fn (eval-only setups) is also fine.
        m = _bare_model_base()
        m._trainer = MagicMock(current_epoch=0)
        m.on_train_epoch_start()  # must not raise

    def test_set_epoch_not_callable_no_op(self):
        # REGRESSION: a non-curriculum loss that happens to expose a
        # ``set_epoch`` attribute (e.g. as a string config field) must not
        # crash — the hook checks callable() before invoking.
        m = _bare_model_base()
        bogus = MagicMock(spec=nn.Module)
        bogus.set_epoch = "not callable"
        m.loss_fn = bogus
        m._trainer = MagicMock(current_epoch=2)
        m.on_train_epoch_start()  # must not raise
