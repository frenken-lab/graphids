"""State-carrying Mamba-1 scan using the pinned official CUDA operators."""

import torch
from torch.nn import functional as F

_WORKSPACE_BYTES = 64 * 1024**2


def mamba_chunk(block, x, conv, state, *, official, differentiable_state=True):
    """Return outputs and both final caches without resetting incoming history.

    The official scan starts at zero. Linearity of the SSM recurrence lets us add
    exp(A * sum(delta)) * incoming_state before gating, without dividing by decay.
    The correction is tiled to bound temporary [streams, channels, time, state].
    """
    if not official:
        projected, z = block.in_proj(x).chunk(2, -1)
        history = torch.cat((conv, projected.transpose(1, 2)), -1)
        convolved = F.silu(
            F.conv1d(
                history[:, :, -(x.shape[1] + block.d_conv - 1) :],
                block.conv1d.weight,
                block.conv1d.bias,
                groups=block.d_inner,
            )
        ).transpose(1, 2)
        outputs = []
        for ti in range(x.shape[1]):
            y, state = block._update(convolved[:, ti], z[:, ti], state)
            outputs.append(y)
        return torch.stack(outputs, 1), history[:, :, -block.d_conv :], state
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    streams, length, _ = x.shape
    inner, size = block.d_inner, block.d_state
    max_streams = max(1, _WORKSPACE_BYTES // (3 * inner * size * 4))
    if streams > max_streams:
        parts = [
            mamba_chunk(
                block,
                x[i : i + max_streams],
                conv[i : i + max_streams],
                state[i : i + max_streams],
                official=True,
                differentiable_state=differentiable_state,
            )
            for i in range(0, streams, max_streams)
        ]
        return tuple(torch.cat([part[j] for part in parts]) for j in range(3))
    projected, z = block.in_proj(x).transpose(1, 2).chunk(2, dim=1)
    history = torch.cat((conv, projected), -1)
    convolution_input = history[:, :, -(length + block.d_conv - 1) :].contiguous()
    u = causal_conv1d_fn(
        convolution_input,
        block.conv1d.weight[:, 0],
        block.conv1d.bias,
        activation=block.activation,
    )[:, :, -length:].contiguous()
    final_conv = history[:, :, -block.d_conv :].contiguous()
    parts = block.x_proj(u.transpose(1, 2))
    dt, B, C = torch.split(parts, [block.dt_rank, size, size], dim=-1)
    dt = F.linear(dt, block.dt_proj.weight).transpose(1, 2).contiguous()
    B, C = B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous()
    A = -block.A_log.float().exp()
    y, final_state = selective_scan_fn(
        u,
        dt,
        A,
        B,
        C,
        block.D.float(),
        z=None,
        delta_bias=block.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=True,
    )
    with torch.profiler.record_function("mamba.incoming_state_correction"):
        cumulative = F.softplus(
            dt.float() + block.dt_proj.bias.float()[None, :, None]
        ).cumsum(-1)
        tile = max(1, _WORKSPACE_BYTES // (3 * streams * inner * size * 4))
        corrections = []
        final_parts = []
        delta = F.softplus(dt.float() + block.dt_proj.bias.float()[None, :, None])
        for lo in range(0, length, tile):
            hi = min(length, lo + tile)
            decay = torch.exp(cumulative[:, :, lo:hi, None] * A[None, :, None, :])
            correction = (
                decay
                * state.float()[:, :, None, :]
                * C[:, None, :, lo:hi].transpose(-1, -2).float()
            ).sum(-1)
            corrections.append(correction.to(y.dtype))
            # The official operator intentionally omits gradients for last_state.
            # Reconstruct it functionally when differentiable outgoing state is needed.
            if torch.is_grad_enabled() and differentiable_state:
                end_decay = torch.exp(
                    (cumulative[:, :, -1:] - cumulative[:, :, lo:hi])[:, :, :, None]
                    * A[None, :, None, :]
                )
                updates = (delta[:, :, lo:hi] * u[:, :, lo:hi].float())[:, :, :, None]
                final_parts.append(
                    (end_decay * updates * B[:, None, :, lo:hi].transpose(-1, -2)).sum(
                        2
                    )
                )
        y = y + torch.cat(corrections, -1)
        if final_parts:
            final_state = torch.stack(final_parts).sum(0)
        final_state = (
            final_state
            + torch.exp(cumulative[:, :, -1, None] * A[None]) * state.float()
        )
    y = y * F.silu(z)
    return block.out_proj(y.transpose(1, 2)), final_conv, final_state.to(state.dtype)
