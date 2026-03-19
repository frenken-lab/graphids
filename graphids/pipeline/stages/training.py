"""Training stages: autoencoder, curriculum, normal."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from graphids.config import PipelineConfig, checkpoint_path, config_path, run_id

from .batch_sizing import resolve_batch_config
from .data_loading import training_preamble
from .modules import CurriculumDataModule, GATModule, VGAEModule
from .trainer_factory import prepare_kd
from .utils import (
    cleanup,
    graph_label,
    load_data,
    load_model,
    make_dataloader,
    make_trainer,
)

log = logging.getLogger(__name__)


def _extract_training_metrics(trainer: pl.Trainer) -> dict:
    """Extract metrics from trainer's callback state after training.

    Returns a dict with val_loss, epochs_run, and any scalar callback metrics.
    """
    metrics: dict = {}

    # Extract best score from ModelCheckpoint callback
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_score is not None:
            metrics["val_loss"] = float(cb.best_model_score)
            break

    # Add final logged scalar metrics
    if trainer.callback_metrics:
        for k, v in trainer.callback_metrics.items():
            if k not in metrics:
                try:
                    metrics[k] = float(v) if hasattr(v, "item") else v
                except (TypeError, ValueError):
                    pass  # skip non-scalar values

    metrics["epochs_run"] = trainer.current_epoch + 1
    return metrics


def _resume_ckpt_path(cfg: PipelineConfig, stage: str) -> str | None:
    """Find a checkpoint to resume from.

    Resolution order:
    1. ``KD_GAT_CKPT_PATH`` env var — explicit override from orchestrator
       (set when Dagster retries a timed-out stage).
    2. Lightning auto-save — ``.pl_auto_save.ckpt`` in persistent_root,
       written by ``SLURMEnvironment(auto_requeue=True)`` on SIGUSR1.
    """
    # 1. Explicit override from orchestrator
    path = os.environ.get("KD_GAT_CKPT_PATH")
    try:
        del os.environ["KD_GAT_CKPT_PATH"]
    except KeyError:
        pass
    if path and Path(path).exists():
        log.info("Resuming from orchestrator checkpoint: %s", path)
        return path
    if path:
        log.warning("Checkpoint path set but not found: %s", path)

    # 2. Lightning auto-save from SLURMEnvironment (timeout requeue)
    persistent_root = Path(cfg.lake_root) / run_id(cfg, stage)
    auto_save = persistent_root / ".pl_auto_save.ckpt"
    if auto_save.exists():
        log.info("Resuming from Lightning auto-save: %s", auto_save)
        return str(auto_save)

    return None


def _save_and_cleanup(module, trainer, cfg, stage: str, label: str | None = None) -> dict:
    """Save checkpoint and config. Returns result dict with checkpoint path and metrics.

    Metrics extraction is non-fatal — returns empty metrics on failure.
    """
    try:
        metrics = _extract_training_metrics(trainer)
    except Exception as e:
        log.warning("Failed to extract training metrics: %s", e)
        metrics = {}

    ckpt = checkpoint_path(cfg, stage)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, stage))
    log.info("Saved %s: %s", label or stage, ckpt)
    cleanup()
    return {"checkpoint": str(ckpt), "metrics": metrics}


def train_autoencoder(cfg: PipelineConfig) -> dict:
    """Train VGAE on graph reconstruction. Returns result dict with checkpoint and metrics."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "AUTOENCODER")

    teacher, projection = prepare_kd(cfg, "vgae", num_ids, in_ch, device)
    module = VGAEModule(cfg, num_ids, in_ch, teacher=teacher, projection=projection)
    bs, max_nodes = resolve_batch_config(cfg)

    train_dl = make_dataloader(train_data, cfg, bs, shuffle=True, max_num_nodes=max_nodes)
    val_dl = make_dataloader(val_data, cfg, bs, shuffle=False, max_num_nodes=max_nodes)

    trainer = make_trainer(cfg, "autoencoder")
    trainer.fit(module, train_dl, val_dl, ckpt_path=_resume_ckpt_path(cfg, "autoencoder"))
    return _save_and_cleanup(module, trainer, cfg, "autoencoder", "VGAE")


def train_curriculum(cfg: PipelineConfig) -> dict:
    """Train GAT with VGAE-guided curriculum learning. Returns result dict with checkpoint and metrics."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "CURRICULUM")

    # Load VGAE for difficulty scoring
    vgae = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, device)

    # Split and score
    normals = [g for g in train_data if graph_label(g) == 0]
    attacks = [g for g in train_data if graph_label(g) == 1]
    scores = _score_difficulty(vgae, normals, device, canid_weight=cfg.vgae.canid_weight)
    del vgae
    cleanup()

    teacher, _ = prepare_kd(cfg, "gat", num_ids, in_ch, device)
    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)
    trainer = make_trainer(cfg, "curriculum")

    dm = CurriculumDataModule(normals, attacks, scores, val_data, cfg)
    trainer.fit(module, datamodule=dm, ckpt_path=_resume_ckpt_path(cfg, "curriculum"))
    return _save_and_cleanup(module, trainer, cfg, "curriculum", "GAT")


def train_normal(cfg: PipelineConfig) -> dict:
    """Train GAT with standard cross-entropy (no curriculum). Returns result dict with checkpoint and metrics."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "NORMAL")

    teacher, _ = prepare_kd(cfg, "gat", num_ids, in_ch, device)
    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)
    bs, max_nodes = resolve_batch_config(cfg)

    train_dl = make_dataloader(train_data, cfg, bs, shuffle=True, max_num_nodes=max_nodes)
    val_dl = make_dataloader(val_data, cfg, bs, shuffle=False, max_num_nodes=max_nodes)

    trainer = make_trainer(cfg, "normal")
    trainer.fit(module, train_dl, val_dl, ckpt_path=_resume_ckpt_path(cfg, "normal"))
    return _save_and_cleanup(module, trainer, cfg, "normal", "GAT (normal)")


def _score_difficulty(
    vgae_model, graphs, device, chunk_size: int = 500, canid_weight: float = 0.1
) -> list[float]:
    """Score each graph's reconstruction difficulty using trained VGAE.

    Memory optimization: Processes graphs in chunks and clears GPU cache between
    chunks to prevent memory accumulation on large datasets.
    """
    from graphids.core.preprocessing import get_batch_index

    scores = []
    vgae_model.eval()
    total_chunks = (len(graphs) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(graphs))
        chunk_graphs = graphs[start:end]

        with torch.no_grad():
            for g in chunk_graphs:
                g = g.clone().to(device)
                batch_idx = get_batch_index(g, device)
                edge_attr = getattr(g, "edge_attr", None)
                cont, canid_logits, _, _, _ = vgae_model(
                    g.x, g.edge_index, batch_idx, edge_attr=edge_attr
                )
                recon = F.mse_loss(cont, g.x[:, 1:]).item()
                canid = F.cross_entropy(canid_logits, g.x[:, 0].long()).item()
                scores.append(recon + canid_weight * canid)
                del g

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (chunk_idx + 1) % 10 == 0:
            log.info("Difficulty scoring: %d/%d chunks complete", chunk_idx + 1, total_chunks)

    return scores
