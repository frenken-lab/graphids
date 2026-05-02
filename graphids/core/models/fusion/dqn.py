"""DQN fusion: torchrl ``DQNLoss`` + ``EGreedyModule`` over ``QValueActor``.

Subclasses ``RLFusionBase`` and contributes only the DQN-specific math:
the Q-actor + epsilon-greedy explorer, the ``DQNLoss`` (with ``double_dqn``
toggle and ``delay_value`` target net), and ``SoftUpdate`` Polyak sync.
``gamma=0`` because each graph is an independent context.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Categorical
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss
from torchrl.objectives.utils import SoftUpdate

from .base import RLFusionBase


class DQNFusionModule(RLFusionBase):
    def __init__(
        self,
        state_dim: int = 15,
        alpha_steps: int = 21,
        lr: float = 1e-3,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        weight_decay: float = 1e-5,
        decision_threshold: float = 0.5,
        gpu_training_steps: int = 1,
        double_dqn: bool = True,
        target_eps: float = 0.995,
        reward_kwargs: dict | None = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            batch_size=batch_size,
            state_dim=state_dim,
            alpha_steps=alpha_steps,
            decision_threshold=decision_threshold,
            reward_kwargs=reward_kwargs,
        )
        self._store_init_kwargs(locals())

        action_spec = Categorical(n=alpha_steps, shape=(), dtype=torch.long)

        trunk = MLP(
            in_features=state_dim,
            out_features=alpha_steps,
            num_cells=[hidden_dim] * num_layers,
            activation_class=nn.ReLU,
        )
        self.q_value_module = TensorDictModule(
            trunk, in_keys=["observation"], out_keys=["action_value"]
        )
        self.q_actor = QValueActor(
            module=self.q_value_module,
            spec=action_spec,
            in_keys=["observation"],
            action_space="categorical",
        )

        ratio = max(min_epsilon / max(epsilon, 1e-9), 1e-9)
        annealing_num_steps = max(1, int(math.log(ratio) / math.log(epsilon_decay)))
        self._egreedy = EGreedyModule(
            spec=action_spec,
            eps_init=epsilon,
            eps_end=min_epsilon,
            annealing_num_steps=annealing_num_steps,
            action_key="action",
        )
        self.explore_policy = TensorDictSequential(self.q_actor, self._egreedy)

        self.loss_module = DQNLoss(
            value_network=self.q_actor,
            loss_function="smooth_l1",
            delay_value=True,
            double_dqn=double_dqn,
            action_space="categorical",
        )
        self.loss_module.make_value_estimator(gamma=0.0)
        self._target_updater = SoftUpdate(self.loss_module, eps=target_eps)

        self._optimizer = optim.AdamW(
            self.loss_module.parameters(), lr=lr, weight_decay=weight_decay
        )

    # -- RLFusionBase hooks --------------------------------------------------

    def _compute_loss(self, sample: TensorDict) -> torch.Tensor:
        return self.loss_module(sample)["loss"]

    def _score_actions(self, td: TensorDict, training: bool) -> None:
        (self.explore_policy if training else self.q_actor)(td)

    def _after_act(self, actions, norm_states, rewards) -> None:
        self._egreedy.step(int(norm_states.size(0)))

    def _after_optim_step(self) -> None:
        self._target_updater.step()

    def _extra_metrics(self) -> dict:
        return {"epsilon": float(self._egreedy.eps.item())}

    # -- analysis ------------------------------------------------------------

    def q_values(self, norm_states: torch.Tensor) -> torch.Tensor:
        td = TensorDict(
            {"observation": norm_states.to(self.device)},
            batch_size=[norm_states.size(0)],
            device=self.device,
        )
        with torch.no_grad():
            self.q_value_module(td)
        return td["action_value"].detach().cpu()
