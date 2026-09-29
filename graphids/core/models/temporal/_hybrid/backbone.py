"""Sequential backbones for temporal hybrid event streams."""

from __future__ import annotations

import torch
import torch.nn as nn


class _SsmLiteCell(nn.Module):
    """Small gated recurrent state-space-like block with causal linear-time scans."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden)
        self.decay_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros_like(x)
        decay = torch.sigmoid(self.decay_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))
        candidate = torch.tanh(self.in_proj(x))
        next_state = decay * state + gate * candidate
        out = self.out_proj(next_state)
        return self.norm(x + self.dropout(out)), next_state


class TemporalStreamBackbone(nn.Module):
    """Configurable causal stream backbone."""

    def __init__(self, *, backbone_type: str, hidden: int, layers: int, dropout: float):
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
            self.cells = nn.ModuleList([_SsmLiteCell(self.hidden, float(dropout)) for _ in range(self.layers)])
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
            state = x.new_zeros((self.layers, self.hidden))
        h = x
        next_layers: list[torch.Tensor] = []
        for layer_idx, cell in enumerate(self.cells):
            if self.backbone_type == "gru":
                h = cell(h.unsqueeze(0), state[layer_idx].unsqueeze(0)).squeeze(0)
                h = self.dropout(h)
                next_layers.append(h)
            else:
                h, next_state = cell(h, state[layer_idx])
                next_layers.append(next_state)
        return h, torch.stack(next_layers, dim=0)
