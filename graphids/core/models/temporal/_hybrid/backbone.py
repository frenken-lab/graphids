"""Sequential backbones for temporal hybrid event streams."""

from __future__ import annotations

import torch
import torch.nn as nn


class _SsmLiteCell(nn.Module):
    """Small gated recurrent state-space-like block with causal linear-time scans."""

    def __init__(self, hidden: int, dropout: float, *, ssm_kernel: str = "torch"):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden)
        self.decay_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.ssm_kernel = ssm_kernel

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros_like(x)
        from .kernels.ssm import ssm_recurrence
        next_state = ssm_recurrence(self.decay_proj(x), self.gate_proj(x), self.in_proj(x), state,
                                    kernel=self.ssm_kernel)
        out = self.out_proj(next_state)
        return self.norm(x + self.dropout(out)), next_state


class TemporalStreamBackbone(nn.Module):
    """Configurable causal stream backbone."""

    def __init__(self, *, backbone_type: str, hidden: int, layers: int, dropout: float,
                 ssm_kernel: str = "torch"):
        super().__init__()
        self.backbone_type = str(backbone_type)
        self.hidden = int(hidden)
        self.layers = max(1, int(layers))
        self.dropout = nn.Dropout(float(dropout))

        if self.backbone_type == "none":
            self.cells = nn.ModuleList()
        elif self.backbone_type == "gru":
            self.cells = nn.ModuleList([nn.GRUCell(self.hidden, self.hidden) for _ in range(self.layers)])
        elif self.backbone_type == "ssm_lite":
            self.cells = nn.ModuleList([_SsmLiteCell(self.hidden, float(dropout), ssm_kernel=ssm_kernel) for _ in range(self.layers)])
        elif self.backbone_type == "mamba":
            try:
                import mamba_ssm  # noqa: F401
            except ImportError as exc:  # pragma: no cover - depends on optional external package.
                raise ImportError(
                    "backbone.type='mamba' requires the optional 'mamba_ssm' package. "
                    "Install it explicitly or use backbone.type='ssm_lite'."
                ) from exc
            raise NotImplementedError("backbone.type='mamba' is reserved for the optional backend.")
        else:
            raise ValueError("backbone.type must be one of: none, gru, ssm_lite, mamba")

    def reset_state(self) -> torch.Tensor | None:
        return None

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.backbone_type == "none":
            return x, None
        if state is None:
            shape = (self.layers, self.hidden) if x.ndim == 1 else (self.layers, x.size(0), self.hidden)
            state = x.new_zeros(shape)
        h = x
        next_layers: list[torch.Tensor] = []
        for layer_idx, cell in enumerate(self.cells):
            if self.backbone_type == "gru":
                h = cell(h, state[layer_idx])
                h = self.dropout(h)
                next_layers.append(h)
            else:
                h, next_state = cell(h, state[layer_idx])
                next_layers.append(next_state)
        return h, torch.stack(next_layers, dim=0)

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.backbone_type == "none":
            return x, None
        if self.backbone_type == "gru":
            return self._forward_gru_sequence(x, state, valid_mask)
        return self._forward_ssm_lite_sequence(x, state, valid_mask)

    def _initial_sequence_state(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is not None:
            return state
        return x.new_zeros((self.layers, x.size(0), self.hidden))

    def _forward_gru_sequence(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = self._initial_sequence_state(x, state)
        h = x
        next_layers: list[torch.Tensor] = []
        for layer_idx, cell in enumerate(self.cells):
            prev = state[layer_idx]
            outputs: list[torch.Tensor] = []
            for step_idx in range(h.size(1)):
                proposed = cell(h[:, step_idx], prev)
                proposed = self.dropout(proposed)
                if valid_mask is not None:
                    active = valid_mask[:, step_idx].view(-1, 1)
                    proposed = torch.where(active, proposed, prev)
                outputs.append(proposed)
                prev = proposed
            h = torch.stack(outputs, dim=1)
            next_layers.append(prev)
        return h, torch.stack(next_layers, dim=0)

    def _forward_ssm_lite_sequence(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = self._initial_sequence_state(x, state)
        h = x
        next_layers: list[torch.Tensor] = []
        for layer_idx, cell in enumerate(self.cells):
            prev = state[layer_idx]
            decay = torch.sigmoid(cell.decay_proj(h))
            gate = torch.sigmoid(cell.gate_proj(h))
            candidate = torch.tanh(cell.in_proj(h))
            states: list[torch.Tensor] = []
            for step_idx in range(h.size(1)):
                proposed = decay[:, step_idx] * prev + gate[:, step_idx] * candidate[:, step_idx]
                if valid_mask is not None:
                    active = valid_mask[:, step_idx].view(-1, 1)
                    proposed = torch.where(active, proposed, prev)
                states.append(proposed)
                prev = proposed
            state_sequence = torch.stack(states, dim=1)
            h = cell.norm(h + cell.dropout(cell.out_proj(state_sequence)))
            next_layers.append(prev)
        return h, torch.stack(next_layers, dim=0)
