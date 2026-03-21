"""CKA (Centered Kernel Alignment) for teacher-student layer comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


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


def compute_and_save_cka(cfg, val_data, device, output_dir: Path) -> None:
    """Compute layer-wise CKA between teacher (large) and student (current scale), save to JSON."""
    from graphids.config import resolve
    from omegaconf import open_dict
    from .trainer_factory import load_model

    # Student = current config's GAT
    student = load_model(cfg, "gat", "curriculum", device)
    # Teacher = large-scale GAT (inherit num_ids/in_channels from cfg)
    teacher_cfg = resolve(f"model_type=gat", f"scale=large", f"dataset={cfg.dataset}", f"seed={cfg.seed}")
    with open_dict(teacher_cfg):
        teacher_cfg.num_ids = cfg.num_ids
        teacher_cfg.in_channels = cfg.in_channels
    teacher = load_model(teacher_cfg, "gat", "curriculum", device)

    student_reps = _collect_reps(student, val_data, device)
    teacher_reps = _collect_reps(teacher, val_data, device)

    n_layers = min(len(teacher_reps), len(student_reps))
    scores = {f"layer_{i}": _linear_cka(teacher_reps[i], student_reps[i]) for i in range(n_layers)}
    Path(output_dir / "cka.json").write_text(json.dumps(scores, indent=2))


def _collect_reps(model, data, device, max_samples: int = 500) -> list[np.ndarray]:
    layers: list[list] | None = None
    count = 0
    with torch.no_grad():
        for g in data:
            if count >= max_samples:
                break
            g = g.clone().to(device)
            xs = model(g, return_intermediate=True)
            reps = [x.mean(dim=0).cpu().numpy() for x in xs]
            if layers is None:
                layers = [[] for _ in reps]
            for i, r in enumerate(reps):
                layers[i].append(r)
            count += 1
    return [np.array(l) for l in layers] if layers else []
