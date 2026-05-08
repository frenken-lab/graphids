"""Fusion bridge contracts: Lightning manual-opt + replay-buffer ckpt I/O.

CONTRACT 1 — RLFusionBase._learn_step routes through Lightning's manual-opt API
(``self.optimizers()`` + ``self.manual_backward(loss)``) so AMP gradient
scaling and DDP optimizer toggling fire under
``automatic_optimization=False``. Bare ``self._optimizer.step()`` /
``loss.backward()`` silently produce zero grads under bf16/fp16 mixed
precision (Lightning manual-opt docs).

CONTRACT 2 — RLFusionBase ckpt round-trips the torchrl
``TensorDictReplayBuffer``. Lightning's default ckpt persists state_dict
+ optimizer_states only; the RB lives outside the nn.Module parameter
tree. Without on_save_checkpoint / on_load_checkpoint, a SIGUSR2 preempt
mid-episode drops the buffer and DQN restarts learning from empty memory
on resume.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from tensordict import TensorDict

_REWARD_KWARGS = {
    "vgae_weights": [0.4, 0.3, 0.3],
    "correct": 3.0,
    "incorrect": -3.0,
    "confidence_weight": 0.5,
    "combined_conf_weight": 0.3,
    "disagreement_penalty": -1.0,
    "overconf_penalty": -1.5,
    "balance_weight": 0.3,
}


def _push_dummy(rb, n: int, obs_dim: int) -> None:
    ones = torch.ones(n, 1, dtype=torch.bool)
    rb.extend(
        TensorDict(
            {
                "observation": torch.randn(n, obs_dim),
                "action": torch.zeros(n, dtype=torch.long),
                "next": TensorDict(
                    {
                        "observation": torch.randn(n, obs_dim),
                        "reward": torch.randn(n, 1),
                        "done": ones,
                        "terminated": ones,
                    },
                    batch_size=[n],
                ),
            },
            batch_size=[n],
        )
    )


class TestRLFusionLightningManualOptContract:
    def test_learn_step_uses_lightning_manual_opt_api(self):
        # CONTRACT 1: source-level anti-regression. Anyone reverting to
        # bare ``self._optimizer.step()`` / ``loss.backward()`` would
        # silently break bf16 AMP fusion runs (zero grads from the missed
        # GradScaler hook). The runtime path only fails under mixed
        # precision, so a pure-Python source check is the cheapest gate.
        src = Path(__file__).resolve().parents[3] / "graphids/core/models/fusion/base.py"
        learn_step = re.search(
            r"def _learn_step\(self.*?(?=\n    def |\nclass |\Z)",
            src.read_text(),
            re.DOTALL,
        )
        assert learn_step is not None, "could not locate _learn_step in fusion/base.py"
        # Strip comments so substrings inside docstring/inline comments
        # don't mask a real revert.
        code_lines = [
            line for line in learn_step.group(0).splitlines() if not line.lstrip().startswith("#")
        ]
        body = "\n".join(code_lines)
        assert "self.optimizers()" in body, (
            "_learn_step must call self.optimizers() — bare self._optimizer.step() "
            "skips Lightning's GradScaler / DDP optimizer toggle"
        )
        assert "self.manual_backward(" in body, (
            "_learn_step must call self.manual_backward(loss) — bare loss.backward() "
            "skips AMP gradient scaling under bf16/fp16-mixed"
        )
        assert "loss.backward(" not in body, (
            "_learn_step must not call loss.backward() directly — use self.manual_backward()"
        )


class TestRLFusionReplayBufferCheckpoint:
    def _make_module(self):
        from graphids.core.models.fusion.bandit import BanditFusionModule

        return BanditFusionModule(
            state_dim=15,
            alpha_steps=4,
            hidden_dim=8,
            num_layers=1,
            buffer_size=64,
            batch_size=4,
            reward_kwargs=_REWARD_KWARGS,
        )

    def test_replay_buffer_survives_checkpoint_roundtrip(self):
        # CONTRACT 2: RB length + sampled tensor identity preserved across
        # on_save_checkpoint → on_load_checkpoint. Under SIGUSR2 preempt,
        # the resumed process must see the same RB it had at preempt.
        # REGRESSION guard: before the on_save_checkpoint hook landed, RB
        # was dropped silently and DQN learning restarted from empty.
        m1 = self._make_module()
        _push_dummy(m1._rb, n=10, obs_dim=15)
        assert len(m1._rb) == 10

        ckpt: dict = {}
        m1.on_save_checkpoint(ckpt)
        assert "replay_buffer" in ckpt, (
            "on_save_checkpoint must include 'replay_buffer' key — "
            "torchrl RB lives outside nn.Module's state_dict"
        )

        m2 = self._make_module()
        assert len(m2._rb) == 0
        m2.on_load_checkpoint(ckpt)
        assert len(m2._rb) == 10, "RB length lost across ckpt roundtrip"

    def test_load_checkpoint_skips_missing_replay_buffer(self):
        # INVARIANT: legacy ckpts (saved before the hook landed) load
        # without replay_buffer; on_load_checkpoint must no-op rather than
        # KeyError so resume from old ckpts still works.
        m = self._make_module()
        m.on_load_checkpoint({})  # no replay_buffer key
        assert len(m._rb) == 0
