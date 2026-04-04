"""Shared NN infrastructure for fusion agents.

Generic building blocks used by both ``DQNFusionModule`` and ``BanditFusionModule``.
Kept out of each agent's file so retiring one agent doesn't break the other
via hidden cross-file imports.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp_body(state_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """Build MLP trunk: [Linear → LayerNorm → ReLU → Dropout(0.2)] × N."""
    layers: list[nn.Module] = []
    in_dim = state_dim
    for _ in range(num_layers):
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])
        in_dim = hidden_dim
    return nn.Sequential(*layers)


class TensorReplayBuffer:
    """Fixed-size circular buffer backed by contiguous tensors.

    Stores (state, action, reward) triples only — next_state is always
    identical to state in the current fusion formulation.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity)
        self._pos = 0
        self._size = 0

    def add_batch(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        """Add a batch of experiences. Wraps around when full."""
        n = len(states)
        if n >= self.capacity:
            # Keep only the last `capacity` items
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            n = self.capacity

        end = self._pos + n
        if end <= self.capacity:
            self.states[self._pos : end] = states
            self.actions[self._pos : end] = actions
            self.rewards[self._pos : end] = rewards
        else:
            first = self.capacity - self._pos
            self.states[self._pos :] = states[:first]
            self.actions[self._pos :] = actions[:first]
            self.rewards[self._pos :] = rewards[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]

        self._pos = (self._pos + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random sample without replacement (or with, if batch_size > size)."""
        idx = torch.randint(0, self._size, (batch_size,))
        return self.states[idx], self.actions[idx], self.rewards[idx]

    def __len__(self):
        return self._size
