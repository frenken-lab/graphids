"""All filesystem I/O for the artifact pipeline.

Loaders return torch / numpy structures; savers consume the dataclasses
defined in :mod:`compute` and write to ``output_dir``. Compute primitives
never touch the filesystem — split here so the pure pipeline can be
exercised against in-memory fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from structlog import get_logger

from graphids.core.models.base import safe_load_checkpoint

from .compute import (
    AttentionResult,
    EmbeddingsResult,
    LandscapeResult,
    PolicyResult,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_val_data(
    *,
    lake_root: str,
    dataset: str,
    vocab_scope: str,
    seed: int,
    window_size: int,
    stride: int,
) -> list:
    """Load the val split through the same Source → ``get_or_build`` path
    training/eval uses. ``val_fraction``, scaler strategy, and cache
    digest live on ``CANBusSource``; there is no parallel declaration
    for the analyzer to drift against.
    """
    from graphids.core.data.datasets.can_bus import CANBusSource
    from graphids.core.data.state import get_or_build

    state = get_or_build(
        CANBusSource(
            name=dataset,
            lake_root=lake_root,
            window_size=window_size,
            stride=stride,
            seed=seed,
            vocab_scope=vocab_scope,
        )
    )
    val = list(state.val)
    log.info("data_loaded", n_val=len(val))
    return val


def load_teacher(model_type: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load and ``.eval()`` a teacher checkpoint for CKA."""
    teacher = safe_load_checkpoint(model_type, ckpt_path, map_location=device)
    teacher.eval()
    return teacher


def load_fusion_eval(
    *,
    dataset: str,
    lake_root: str,
    seed: int,
    vgae_ckpt_path: str,
    gat_ckpt_path: str,
    window_size: int,
    stride: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the FusionDataModule eval cache and return ``(states, labels)``."""
    from graphids.core.data.datamodule.fusion import FusionDataModule

    dm = FusionDataModule(
        dataset=dataset,
        lake_root=lake_root,
        seed=seed,
        vgae_ckpt_path=vgae_ckpt_path,
        gat_ckpt_path=gat_ckpt_path,
        window_size=window_size,
        stride=stride,
    )
    dm.setup("test")
    return dm.val_cache["states"].to(device), dm.val_cache["labels"]


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------


def save_embeddings(output_dir: Path, r: EmbeddingsResult) -> None:
    path = output_dir / "embeddings.npz"
    np.savez_compressed(path, embeddings=r.embeddings, labels=r.labels, model_type=r.model_type)
    log.info("embeddings_saved", path=str(path), n_samples=len(r.labels), model_type=r.model_type)


def save_attention(output_dir: Path, r: AttentionResult) -> None:
    if r.n_samples == 0:
        return
    path = output_dir / "attention_weights.npz"
    np.savez_compressed(path, n_samples=np.array(r.n_samples), **r.weights)
    log.info("attention_weights_saved", samples=r.n_samples, path=str(path))


def save_cka(output_dir: Path, scores: dict[str, float]) -> None:
    path = output_dir / "cka.json"
    path.write_text(json.dumps(scores, indent=2))
    log.info("cka_saved", path=str(path), n_layers=len(scores))


def save_landscape(output_dir: Path, r: LandscapeResult) -> None:
    table = pa.table({
        "x": r.x,
        "y": r.y,
        "loss": r.loss,
        "model_type": [r.model_type] * len(r.x),
        "dataset": [r.dataset] * len(r.x),
    })
    path = output_dir / f"loss_landscape_{r.model_type}.parquet"
    pq.write_table(table, path)
    log.info("loss_landscape_saved", model=r.model_type, points=len(r.x), path=str(path))


def save_fusion_policy(output_dir: Path, r: PolicyResult) -> None:
    alpha_list = r.alphas.tolist()
    label_list = r.labels.tolist()
    by_label: dict[str, list] = {"normal": [], "attack": []}
    for a, lbl in zip(alpha_list, label_list):
        by_label["normal" if lbl == 0 else "attack"].append(a)
    path = output_dir / "dqn_policy.json"
    path.write_text(
        json.dumps(
            {
                "alphas": alpha_list,
                "labels": label_list,
                "alpha_by_label": by_label,
                "q_values": r.q_values.tolist(),
            },
            indent=2,
        )
    )
    log.info("dqn_policy_saved", path=str(path), n_samples=len(label_list))
