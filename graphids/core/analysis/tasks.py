"""Task-level helpers for generating analyzer artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from graphids.core.models._training import safe_load_checkpoint
from graphids.log import get_logger

log = get_logger(__name__)


def run_embeddings(
    *,
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    model_type: str,
    max_samples: int,
    batch_size: int,
) -> None:
    from .embeddings import collect_and_save_embeddings

    log.info("artifact_start", artifact="embeddings")
    collect_and_save_embeddings(
        model,
        val_data,
        device,
        output_dir,
        model_type=model_type,
        max_samples=max_samples,
        batch_size=batch_size,
    )


def run_attention(
    *,
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    max_samples: int,
) -> None:
    from .embeddings import collect_and_save_attention

    log.info("artifact_start", artifact="attention")
    collect_and_save_attention(
        model,
        val_data,
        device,
        output_dir,
        max_samples=max_samples,
    )


def run_cka(
    *,
    model: torch.nn.Module,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    teacher_ckpt: str,
    max_samples: int,
) -> None:
    from .cka import compute_and_save_cka

    log.info("artifact_start", artifact="cka")
    teacher_module = safe_load_checkpoint("gat", teacher_ckpt, map_location=device)
    teacher_module.eval()
    compute_and_save_cka(
        model,
        teacher_module.model,
        val_data,
        device,
        output_dir,
        max_samples=max_samples,
    )
    del teacher_module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_landscape(
    *,
    model: torch.nn.Module,
    model_type: str,
    val_data: list,
    device: torch.device,
    output_dir: Path,
    hparams,
    resolution: int,
    scale: float,
    max_graphs: int,
    dataset: str,
    seed: int,
) -> None:
    from .loss_landscape import compute_and_save_loss_landscape

    log.info("artifact_start", artifact="landscape")
    compute_and_save_loss_landscape(
        model,
        model_type,
        val_data,
        device,
        output_dir,
        hparams,
        resolution=resolution,
        scale=scale,
        max_graphs=max_graphs,
        dataset=dataset,
        seed=seed,
    )


def run_fusion_policy(
    *,
    module,
    dataset: str,
    lake_root: str,
    seed: int,
    vgae_ckpt_path: str,
    gat_ckpt_path: str,
    window_size: int,
    stride: int,
    output_dir: Path,
    device: torch.device,
) -> None:

    from graphids.core.data.datamodule.fusion import FusionDataModule

    from .fusion_policy import save_fusion_policy

    log.info("artifact_start", artifact="fusion_policy")
    agent = module.agent
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
    states = dm.val_cache["states"].to(device)
    labels = dm.val_cache["labels"]
    result = agent.predict(states)
    save_fusion_policy(
        output_dir,
        alphas=result["alphas"].cpu().numpy(),
        labels=labels.numpy(),
        q_values=agent.q_values(result["norm_states"]).cpu().numpy(),
    )
