"""Explicit FP32 diagnostic Mamba-1 recurrence. Never used as a CUDA fallback.

Parameter layout follows state-spaces/mamba's Mamba class so identical weights
can exercise sequence/step correctness without compiling CUDA on CPU workers.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner, self.d_state, self.d_conv = d_model * expand, d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        with torch.no_grad():
            self.dt_proj.bias.fill_(-4.0)
        self.A_log = nn.Parameter(
            torch.arange(1, d_state + 1).float().log().repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
        p = self.in_proj.weight
        return (
            p.new_zeros(batch_size, self.d_inner, self.d_conv),
            p.new_zeros(batch_size, self.d_inner, self.d_state),
        )

    def _update(self, x, z, state):
        parts = self.x_proj(x)
        delta, B, C = torch.split(
            parts, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        A = -self.A_log.float().exp()
        state = torch.exp(delta.unsqueeze(-1) * A) * state + delta.unsqueeze(
            -1
        ) * B.unsqueeze(1) * x.unsqueeze(-1)
        y = (state * C.unsqueeze(1)).sum(-1) + self.D * x
        return self.out_proj(y * F.silu(z)), state

    def forward(self, x):
        projected, z = self.in_proj(x).chunk(2, -1)
        convolved = F.silu(
            self.conv1d(projected.transpose(1, 2))[..., : x.shape[1]]
        ).transpose(1, 2)
        state = x.new_zeros(x.shape[0], self.d_inner, self.d_state)
        outputs = []
        for t in range(x.shape[1]):
            out, state = self._update(convolved[:, t], z[:, t], state)
            outputs.append(out)
        return torch.stack(outputs, 1)

    def step(self, x, conv_state, ssm_state):
        projected, z = self.in_proj(x[:, 0]).chunk(2, -1)
        conv_state.copy_(torch.roll(conv_state, -1, -1))
        conv_state[:, :, -1] = projected
        convolved = F.silu(
            (conv_state * self.conv1d.weight[:, 0]).sum(-1) + self.conv1d.bias
        )
        y, next_state = self._update(convolved, z, ssm_state)
        ssm_state.copy_(next_state)
        return y.unsqueeze(1), conv_state, ssm_state
