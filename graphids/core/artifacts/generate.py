"""Generate all derived artifacts from evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import structlog
import torch

log = structlog.get_logger()


def generate_all(
    cfg,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    artifacts: dict,
    *,
    load_model_fn: Callable | None = None,
) -> None:
    """Generate all derived artifacts. Best-effort — failures are logged, never fatal.

    Args:
        cfg: Resolved config namespace.
        val_data: List of PyG Data objects (validation set).
        device: Compute device.
        output_dir: Directory to write artifact files.
        artifacts: Dict of per-model artifact objects (GATResult, VGAEResult, FusionResult).
        load_model_fn: Callable(cfg, model_type, device) -> nn.Module.
    """
    from .embeddings import save_embeddings, save_attention
    from .fusion_policy import save_fusion_policy

    save_embeddings(output_dir, artifacts.get("vgae"), artifacts.get("gat"))
    save_attention(output_dir, artifacts.get("gat"))
    save_fusion_policy(output_dir, artifacts.get("fusion"))

    # CKA (KD runs only)
    if any(a.type == "kd" for a in cfg.get("auxiliaries", [])) and load_model_fn:
        try:
            from .cka import compute_and_save_cka
            compute_and_save_cka(
                cfg, val_data, device, output_dir,
                load_model_fn=load_model_fn,
                max_samples=cfg.evaluation.cka_max_samples,
            )
        except Exception as e:
            log.warning("cka_failed", error=str(e))

    # Loss landscape (opt-in)
    if cfg.evaluation.get("loss_landscape", False) and load_model_fn:
        try:
            from .loss_landscape import compute_and_save_loss_landscape
            compute_and_save_loss_landscape(
                cfg, val_data, device, output_dir,
                load_model_fn=load_model_fn,
                resolution=cfg.evaluation.get("landscape_resolution", 51),
                scale=cfg.evaluation.get("landscape_scale", 1.0),
            )
        except Exception as e:
            log.warning("loss_landscape_failed", error=str(e))
