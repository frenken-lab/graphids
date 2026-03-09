"""Training stages: autoencoder, curriculum, normal."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from graphids.config import PipelineConfig, checkpoint_path, config_path, stage_dir
from graphids.config.constants import get_batch_index

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


def _save_training_metrics(trainer: pl.Trainer, cfg: PipelineConfig, stage: str) -> None:
    """Write metrics.json from trainer's callback state after training.

    Non-fatal: exceptions are logged but do not crash training.
    """
    try:
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

        out = stage_dir(cfg, stage) / "metrics.json"
        out.write_text(json.dumps(metrics, indent=2))
        log.info("Saved training metrics: %s", out)
    except Exception as e:
        log.warning("Failed to save training metrics: %s", e)


def _resume_ckpt_path() -> str | None:
    """Read and consume the resume checkpoint path from environment.

    Set by the coordinator (via CLI --ckpt-path) when resubmitting a
    timed-out stage that saved a Lightning auto-checkpoint.
    """
    path = os.environ.pop("KD_GAT_CKPT_PATH", None)
    if path and Path(path).exists():
        log.info("Resuming from Lightning checkpoint: %s", path)
        return path
    if path:
        log.warning("Checkpoint path set but not found: %s", path)
    return None


def _save_and_cleanup(module, trainer, cfg, stage: str, label: str | None = None) -> Path:
    """Save checkpoint, config, metrics. Returns checkpoint path."""
    _save_training_metrics(trainer, cfg, stage)
    ckpt = checkpoint_path(cfg, stage)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), ckpt)
    cfg.save(config_path(cfg, stage))
    log.info("Saved %s: %s", label or stage, ckpt)
    cleanup()
    return ckpt


def train_autoencoder(cfg: PipelineConfig) -> Path:
    """Train VGAE on graph reconstruction. Returns checkpoint path."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "AUTOENCODER")

    teacher, projection = prepare_kd(cfg, "vgae", num_ids, in_ch, device)
    module = VGAEModule(cfg, num_ids, in_ch, teacher=teacher, projection=projection)
    bs, max_nodes = resolve_batch_config(cfg)

    train_dl = make_dataloader(train_data, cfg, bs, shuffle=True, max_num_nodes=max_nodes)
    val_dl = make_dataloader(val_data, cfg, bs, shuffle=False, max_num_nodes=max_nodes)

    trainer = make_trainer(cfg, "autoencoder")
    trainer.fit(module, train_dl, val_dl, ckpt_path=_resume_ckpt_path())
    return _save_and_cleanup(module, trainer, cfg, "autoencoder", "VGAE")


def train_curriculum(cfg: PipelineConfig) -> Path:
    """Train GAT with VGAE-guided curriculum learning. Returns checkpoint path."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "CURRICULUM")

    # Load VGAE for difficulty scoring
    vgae = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, device)

    # Split and score
    normals = [g for g in train_data if graph_label(g) == 0]
    attacks = [g for g in train_data if graph_label(g) == 1]
    scores = _score_difficulty(vgae, normals, device)
    del vgae
    cleanup()

    teacher, _ = prepare_kd(cfg, "gat", num_ids, in_ch, device)
    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)
    trainer = make_trainer(cfg, "curriculum")

    dm = CurriculumDataModule(normals, attacks, scores, val_data, cfg)
    trainer.fit(module, datamodule=dm, ckpt_path=_resume_ckpt_path())
    return _save_and_cleanup(module, trainer, cfg, "curriculum", "GAT")


def train_normal(cfg: PipelineConfig) -> Path:
    """Train GAT with standard cross-entropy (no curriculum). Returns checkpoint path."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "NORMAL")

    teacher, _ = prepare_kd(cfg, "gat", num_ids, in_ch, device)
    module = GATModule(cfg, num_ids, in_ch, teacher=teacher)
    bs, max_nodes = resolve_batch_config(cfg)

    train_dl = make_dataloader(train_data, cfg, bs, shuffle=True, max_num_nodes=max_nodes)
    val_dl = make_dataloader(val_data, cfg, bs, shuffle=False, max_num_nodes=max_nodes)

    trainer = make_trainer(cfg, "normal")
    trainer.fit(module, train_dl, val_dl, ckpt_path=_resume_ckpt_path())
    return _save_and_cleanup(module, trainer, cfg, "normal", "GAT (normal)")


def _score_difficulty(vgae_model, graphs, device, chunk_size: int = 500) -> list[float]:
    """Score each graph's reconstruction difficulty using trained VGAE.

    Memory optimization: Processes graphs in chunks and clears GPU cache between
    chunks to prevent memory accumulation on large datasets.
    """
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
                scores.append(recon + 0.1 * canid)
                del g

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (chunk_idx + 1) % 10 == 0:
            log.info("Difficulty scoring: %d/%d chunks complete", chunk_idx + 1, total_chunks)

    return scores
