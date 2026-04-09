"""CKA (Centered Kernel Alignment) for teacher-student layer comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from graphids.core.models.base import safe_load_checkpoint
from graphids._otel import get_logger

log = get_logger(__name__)


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    n = K.shape[0]
    ones = torch.ones(n, 1, device=K.device)
    result = torch.trace(K @ L)
    result += ((ones.T @ K @ ones @ ones.T @ L @ ones) / ((n - 1) * (n - 2))).item()
    result -= ((ones.T @ K @ L @ ones) * 2 / (n - 2)).item()
    return (result / (n * (n - 3))).item()


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X_t = torch.from_numpy(X - X.mean(axis=0)).float()
    Y_t = torch.from_numpy(Y - Y.mean(axis=0)).float()
    K, L = X_t @ X_t.T, Y_t @ Y_t.T
    denom = (_unbiased_hsic(K, K) * _unbiased_hsic(L, L)) ** 0.5
    return _unbiased_hsic(K, L) / denom if denom > 0 else 0.0


def compute_and_save_cka(
    student: torch.nn.Module,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    *,
    teacher_ckpt: str,
    max_samples: int = 500,
) -> None:
    """Load teacher, compute layer-wise CKA vs student, save to JSON."""
    log.info("artifact_start", artifact="cka")
    teacher_module = safe_load_checkpoint("gat", teacher_ckpt, map_location=device)
    teacher_module.eval()
    try:
        student_reps = _collect_reps(student, val_data, device, max_samples=max_samples)
        teacher_reps = _collect_reps(
            teacher_module.model, val_data, device, max_samples=max_samples
        )

        n_layers = min(len(teacher_reps), len(student_reps))
        scores = {
            f"layer_{i}": _linear_cka(teacher_reps[i], student_reps[i]) for i in range(n_layers)
        }
        (output_dir / "cka.json").write_text(json.dumps(scores, indent=2))
    finally:
        del teacher_module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _collect_reps(model, data, device, max_samples: int = 500) -> list[np.ndarray]:
    layers: list[list] | None = None
    count = 0
    with torch.no_grad():
        for g in data:
            if count >= max_samples:
                break
            g = g.clone().to(device, non_blocking=True)
            xs = model(g, return_intermediate=True)
            reps = [x.mean(dim=0).cpu().numpy() for x in xs]
            if layers is None:
                layers = [[] for _ in reps]
            for i, r in enumerate(reps):
                layers[i].append(r)
            count += 1
    return [np.array(l) for l in layers] if layers else []
