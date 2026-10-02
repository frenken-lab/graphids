"""Optional first-order float32 SSM pointwise recurrence and analytic backward.

Linear projections, dropout, normalization and recurrent ownership stay in PyTorch.
CPU, other dtypes, and strided inputs use the reference expression.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .rhythm import tl, triton

if triton is not None:
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _forward(D, G, C, S, Out, Jacobian, N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = index < N
        decay = tl.sigmoid(tl.load(D + index, mask, 0))
        gate = tl.sigmoid(tl.load(G + index, mask, 0))
        candidate = libdevice.tanh(tl.load(C + index, mask, 0))
        state = tl.load(S + index, mask, 0)
        tl.store(Out + index, decay * state + gate * candidate, mask)
        tl.store(Jacobian + index, state * decay * (1 - decay), mask)
        tl.store(Jacobian + N + index, candidate * gate * (1 - gate), mask)
        tl.store(Jacobian + 2 * N + index, gate * (1 - candidate * candidate), mask)
        tl.store(Jacobian + 3 * N + index, decay, mask)

    @triton.jit
    def _backward(Grad, Jacobian, Out, N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = index < N
        grad = tl.load(Grad + index, mask, 0)
        for part in tl.static_range(4):
            derivative = tl.load(Jacobian + part * N + index, mask, 0)
            tl.store(Out + part * N + index, grad * derivative, mask)


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay, gate, candidate, state):
        n = decay.numel()
        output = torch.empty_like(decay)
        jacobian = decay.new_empty((4, n))
        if n:
            _forward[(triton.cdiv(n, 256),)](decay, gate, candidate, state, output, jacobian,
                                            n, 256, enable_fp_fusion=False)
        ctx.save_for_backward(jacobian)
        ctx.shape = decay.shape
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (jacobian,) = ctx.saved_tensors
        n = grad.numel()
        derivatives = torch.empty_like(jacobian)
        if n:
            _backward[(triton.cdiv(n, 256),)](grad.contiguous(), jacobian, derivatives, n, 256)
        return tuple(part.view(ctx.shape) for part in derivatives.unbind(0))


def ssm_recurrence(decay, gate, candidate, state, *, kernel='torch'):
    tensors = (decay, gate, candidate, state)
    if kernel == 'triton' and triton is not None and all(
        value.is_cuda and value.dtype == torch.float32 and value.is_contiguous()
        and value.shape == decay.shape and value.device == decay.device for value in tensors
    ):
        return _Recurrence.apply(*tensors)
    return torch.sigmoid(decay) * state + torch.sigmoid(gate) * torch.tanh(candidate)
