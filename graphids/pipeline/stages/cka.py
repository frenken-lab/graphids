"""CKA (Centered Kernel Alignment) math for teacher-student comparison.

Extracted from storage/mapper.py — domain logic, not I/O.
"""

from __future__ import annotations

import numpy as np
import torch


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    """Unbiased HSIC estimator (Song et al. 2012)."""
    n = K.shape[0]
    ones = torch.ones(n, 1, device=K.device)
    result = torch.trace(K @ L)
    result += (
        (ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((n - 1) * (n - 2))
    ).item()
    result -= ((ones.t() @ K @ L @ ones) * 2 / (n - 2)).item()
    return (1 / (n * (n - 3)) * result).item()


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA with unbiased HSIC estimator."""
    X_t = torch.from_numpy(X - X.mean(axis=0)).float()
    Y_t = torch.from_numpy(Y - Y.mean(axis=0)).float()

    K = X_t @ X_t.T
    L = Y_t @ Y_t.T

    hsic_xy = _unbiased_hsic(K, L)
    hsic_xx = _unbiased_hsic(K, K)
    hsic_yy = _unbiased_hsic(L, L)

    denom = (hsic_xx * hsic_yy) ** 0.5
    return hsic_xy / denom if denom > 0 else 0.0


def _collect_layer_representations(
    model, data, device, max_samples: int = 500
) -> list[np.ndarray]:
    """Collect per-layer representations from a GAT model."""
    all_layers: list[list] | None = None
    count = 0
    with torch.no_grad():
        for g in data:
            if count >= max_samples:
                break
            g = g.clone().to(device)
            xs = model(g, return_intermediate=True)
            layer_reps = [x.mean(dim=0).cpu().numpy() for x in xs]
            if all_layers is None:
                all_layers = [[] for _ in range(len(layer_reps))]
            for i, rep in enumerate(layer_reps):
                all_layers[i].append(rep)
            count += 1
    if all_layers is None:
        return []
    return [np.array(layer) for layer in all_layers]
