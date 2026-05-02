"""Fusion model bases.

All fusion modules consume a feature **TensorDict** (output of
an ``ExtractRow`` in a fusion plan), not a flat state vector. Modules that need a
flat input (Q-network for DQN/Bandit, MLP) call ``flatten_features(td)``
to concatenate every leaf tensor along the feature dim.

- ``FusionModuleBase`` — predict / training_step / validation_step /
  test_step. Branches on ``automatic_optimization``: supervised path
  (MLP, WeightedAvg) implements ``forward_scores(td) -> probs``; RL path
  comes from ``RLFusionBase``.

- ``RLFusionBase`` — torchrl replay buffer + act → reward → push → learn.
  Subclasses provide a torchrl ``LossModule`` plus three hooks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler

from ..base import _ModelBase
from .reward import FusionRewardCalculator

__all__ = [
    "FusionModuleBase",
    "FusionRewardCalculator",
    "RLFusionBase",
    "build_mlp_body",
    "flatten_features",
]


def build_mlp_body(state_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """[Linear → LayerNorm → ReLU → Dropout(0.2)] x N."""
    layers: list[nn.Module] = []
    in_dim = state_dim
    for _ in range(num_layers):
        layers.extend(
            [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        )
        in_dim = hidden_dim
    return nn.Sequential(*layers)


def flatten_features(td: TensorDict) -> torch.Tensor:
    """Concatenate every leaf tensor along the last dim. Stable order:
    sorted nested-key path so the same TD always yields the same layout."""
    leaves = sorted(td.keys(include_nested=True, leaves_only=True))
    return torch.cat([td[k] for k in leaves], dim=-1)


def feature_dim(td: TensorDict) -> int:
    """Total flat feature dim implied by ``flatten_features(td)``."""
    return sum(td[k].size(-1) for k in td.keys(include_nested=True, leaves_only=True))


class FusionModuleBase(_ModelBase):
    automatic_optimization = False

    def __init__(
        self,
        *,
        state_dim: int,
        alpha_steps: int = 21,
        batch_size: int = 128,
        decision_threshold: float = 0.5,
        reward_kwargs: dict | None = None,
    ):
        super().__init__()
        self._store_init_kwargs(locals())
        self.register_buffer("alpha_values", torch.linspace(0, 1, alpha_steps))
        if reward_kwargs is not None:
            self.reward_calc = FusionRewardCalculator(**reward_kwargs)
        from ..base import classification_test_metrics

        self.test_metrics = classification_test_metrics(2)

    def build_optimizers(self, max_epochs: int):
        return None, None

    # -- shared prediction / training / validation / test --------------------

    def predict(self, td: TensorDict) -> dict:
        if self.automatic_optimization:
            scores = self.forward_scores(td)
            return {"fused_scores": scores, "preds": (scores > self.decision_threshold).long()}
        # RL path: greedy action → fused score
        actions, alphas, td_norm = self.select_action_batch(td, training=False)
        anomaly, gat = self.reward_calc.derive_scores(td_norm)
        fused = (1 - alphas) * anomaly + alphas * gat
        return {
            "fused_scores": fused,
            "preds": (fused > self.decision_threshold).long(),
            "alphas": alphas,
            "td_norm": td_norm,
        }

    def _supervised_loss(self, td, labels):
        scores = self.forward_scores(td)
        loss = nn.functional.binary_cross_entropy(
            scores.clamp(1e-7, 1 - 1e-7), labels.float()
        )
        return scores, loss

    def training_step(self, batch, batch_idx):
        td, labels = batch
        if self.automatic_optimization:
            _, loss = self._supervised_loss(td, labels)
            self.log("train_loss", loss)
            return loss
        for k, v in self.train_episode(td, labels).items():
            if v is not None:
                self.log(k, float(v), prog_bar=(k in ("avg_reward", "accuracy")))

    def validation_step(self, batch, batch_idx):
        td, labels = batch
        if self.automatic_optimization:
            scores, loss = self._supervised_loss(td, labels)
            preds = (scores > self.decision_threshold).long()
            self.log("val_loss", loss)
            self.log("val_acc", (preds == labels).float().mean(), prog_bar=True)
            return
        result = self.predict(td)
        preds, td_norm, alphas = result["preds"], result["td_norm"], result["alphas"]
        rewards = self.reward_calc.compute(td_norm, preds, labels, alphas)
        self.log("val_acc", (preds == labels).float().mean().item(), prog_bar=True)
        self.log("avg_reward", rewards.mean().item())
        self.log("avg_alpha", alphas.mean().item())

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        td, labels = batch
        fused = self.predict(td)["fused_scores"].float()
        self.test_metrics.update(torch.stack([1.0 - fused, fused], dim=1), labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())


class RLFusionBase(FusionModuleBase):
    """torchrl replay buffer + unified act/learn flow.

    Subclass sets in ``__init__``:
    - ``self.loss_module`` — a torchrl ``LossModule`` (callable
      ``td → {"loss": tensor}``).
    - ``self._optimizer`` — optimizer over ``loss_module.parameters()``.

    Hooks:
    - ``_score_actions(td, training)`` — write ``td['action']``.
    - ``_after_act(actions, obs, rewards)`` — online update.
    - ``_should_learn()`` — gate the optim step (default: every step).
    - ``_after_optim_step()`` — post-step (DQN target sync).
    - ``_after_learn()`` — post-batch (Bandit ridge reset).
    - ``_extra_metrics()`` — extra log fields.
    """

    automatic_optimization = False

    def __init__(self, *, buffer_size: int, batch_size: int, **kw):
        super().__init__(batch_size=batch_size, **kw)
        self._rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=torch.device("cpu")),
            sampler=RandomSampler(),
            batch_size=batch_size,
        )

    def build_optimizers(self, max_epochs: int):
        return self._optimizer, None

    # -- subclass hooks ------------------------------------------------------

    def _score_actions(self, td: TensorDict, training: bool) -> None:
        raise NotImplementedError

    def _after_act(self, actions, obs, rewards) -> None:
        return None

    def _should_learn(self) -> bool:
        return True

    def _after_optim_step(self) -> None:
        return None

    def _after_learn(self) -> None:
        return None

    def _extra_metrics(self) -> dict:
        return {}

    # -- concrete RL flow ----------------------------------------------------

    def select_action_batch(self, features_td: TensorDict, training: bool = True):
        """Returns ``(actions[N], alphas[N], normalized_features_td[N])``."""
        td_norm = self.reward_calc.normalize(features_td).to(self.device)
        obs = flatten_features(td_norm)
        inner = TensorDict({"observation": obs}, batch_size=[obs.size(0)], device=self.device)
        with torch.no_grad():
            self._score_actions(inner, training=training)
        actions = inner["action"].detach().cpu()
        return actions, self.alpha_values[actions], td_norm.cpu()

    def train_episode(self, features_td: TensorDict, labels: torch.Tensor) -> dict:
        actions, alphas, td_norm = self.select_action_batch(features_td, training=True)
        preds = (alphas > self.decision_threshold).long()
        rewards = self.reward_calc.compute(td_norm, preds, labels, alphas)

        obs = flatten_features(td_norm)  # CPU flat tensor for buffer
        self._after_act(actions, obs, rewards)

        n = obs.size(0)
        ones = torch.ones(n, 1, dtype=torch.bool)
        self._rb.extend(
            TensorDict(
                {
                    "observation": obs,
                    "action": actions,
                    "next": TensorDict(
                        {
                            "observation": obs,
                            "reward": rewards.float().unsqueeze(-1),
                            "done": ones,
                            "terminated": ones,
                        },
                        batch_size=[n],
                    ),
                },
                batch_size=[n],
            )
        )
        return {
            "avg_reward": rewards.mean().item(),
            "avg_alpha": alphas.mean().item(),
            "loss": self._learn_step(),
            **self._extra_metrics(),
        }

    def _learn_step(self) -> float | None:
        if not self._should_learn() or len(self._rb) < self.batch_size:
            return None
        last: float | None = None
        for _ in range(self.gpu_training_steps):
            sample = self._rb.sample().to(self.device)
            loss = self.loss_module(sample)["loss"]
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_norm=1.0)
            self._optimizer.step()
            self._after_optim_step()
            last = float(loss.detach().item())
        self._after_learn()
        return last
