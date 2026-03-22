"""Training stages: autoencoder, curriculum, normal."""

from __future__ import annotations

import structlog
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from graphids.core.preprocessing import CANBusDataModule

from .data_loading import cleanup
from .eval_inference import graph_label
from .modules import CurriculumDataModule, GATModule, VGAEModule
from .trainer_factory import load_model, make_trainer, prepare_kd

log = structlog.get_logger()


def _resume_ckpt_path(cfg, stage: str) -> str | None:
    """Find a checkpoint to resume from.

    Resolution order:
    1. ``KD_GAT_CKPT_PATH`` env var — explicit override from orchestrator
       (set by orchestrator when retrying a timed-out stage).
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
        log.info("resume_from_orchestrator_checkpoint", path=path)
        return path
    if path:
        log.warning("checkpoint_path_not_found", path=path)

    # 2. Lightning auto-save from SLURMEnvironment (timeout requeue)
    auto_save = Path.cwd() / ".pl_auto_save.ckpt"
    if auto_save.exists():
        log.info("resume_from_auto_save", path=str(auto_save))
        return str(auto_save)

    return None


def _save_and_cleanup(module, trainer, cfg, stage: str, label: str | None = None) -> dict:
    """Extract results after training. ModelCheckpoint already saved the model."""
    ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "")
    metrics = {}
    if trainer.callback_metrics:
        metrics = {k: v.item() if hasattr(v, "item") else v
                   for k, v in trainer.callback_metrics.items()}
    log.info("training_complete", label=label or stage, checkpoint=ckpt)
    cleanup()
    return {"checkpoint": ckpt, "metrics": metrics}


def _training_setup(cfg) -> tuple[CANBusDataModule, torch.device]:
    """Common setup: seed, datamodule, populate config, resolve device."""
    pl.seed_everything(cfg.seed)
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup("fit")
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    return dm, device


def train_autoencoder(cfg) -> dict:
    """Train VGAE on graph reconstruction. Returns result dict with checkpoint and metrics."""
    dm, device = _training_setup(cfg)

    teacher, projection = prepare_kd(cfg, "vgae", device)
    module = VGAEModule(cfg, teacher=teacher, projection=projection)

    trainer = make_trainer(cfg, "autoencoder")
    trainer.fit(module, datamodule=dm, ckpt_path=_resume_ckpt_path(cfg, "autoencoder"))
    return _save_and_cleanup(module, trainer, cfg, "autoencoder", "VGAE")


def train_curriculum(cfg) -> dict:
    """Train GAT with VGAE-guided curriculum learning. Returns result dict with checkpoint and metrics."""
    dm, device = _training_setup(cfg)

    # Load VGAE for difficulty scoring
    vgae = load_model(cfg, "vgae", "autoencoder", device)

    # Split and score
    normals = [g for g in dm.train_dataset if graph_label(g) == 0]
    attacks = [g for g in dm.train_dataset if graph_label(g) == 1]
    scores = _score_difficulty(vgae, normals, device, canid_weight=cfg.vgae.canid_weight)
    del vgae
    cleanup()

    teacher, _ = prepare_kd(cfg, "gat", device)
    module = GATModule(cfg, teacher=teacher)
    trainer = make_trainer(cfg, "curriculum")

    cdm = CurriculumDataModule(normals, attacks, scores, list(dm.val_dataset), cfg)
    trainer.fit(module, datamodule=cdm, ckpt_path=_resume_ckpt_path(cfg, "curriculum"))
    return _save_and_cleanup(module, trainer, cfg, "curriculum", "GAT")


def train_normal(cfg) -> dict:
    """Train GAT with standard cross-entropy (no curriculum). Returns result dict with checkpoint and metrics."""
    dm, device = _training_setup(cfg)

    teacher, _ = prepare_kd(cfg, "gat", device)
    module = GATModule(cfg, teacher=teacher)

    trainer = make_trainer(cfg, "normal")
    trainer.fit(module, datamodule=dm, ckpt_path=_resume_ckpt_path(cfg, "normal"))
    return _save_and_cleanup(module, trainer, cfg, "normal", "GAT (normal)")


def _score_difficulty(
    vgae_model, graphs, device, chunk_size: int = 500, canid_weight: float = 0.1
) -> list[float]:
    """Score each graph's reconstruction difficulty using trained VGAE.

    Memory optimization: Processes graphs in chunks and clears GPU cache between
    chunks to prevent memory accumulation on large datasets.
    """
    from torch_geometric.data import Batch
    from torch_geometric.utils import scatter

    scores: list[float] = []
    was_training = vgae_model.training
    vgae_model.eval()
    try:
        total_chunks = (len(graphs) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(graphs))
            chunk_graphs = graphs[start:end]

            with torch.no_grad():
                batch = Batch.from_data_list([g.clone() for g in chunk_graphs]).to(device)
                edge_attr = getattr(batch, "edge_attr", None)
                cont, canid_logits, _, _, _, _ = vgae_model(
                    batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr
                )
                # Per-graph losses via scatter reduction
                node_mse = (cont - batch.x[:, 1:]).pow(2).mean(dim=1)
                graph_mse = scatter(node_mse, batch.batch, reduce="mean")
                node_ce = F.cross_entropy(canid_logits, batch.x[:, 0].long(), reduction="none")
                graph_ce = scatter(node_ce, batch.batch, reduce="mean")
                scores.extend((graph_mse + canid_weight * graph_ce).tolist())
                del batch

            if (chunk_idx + 1) % 10 == 0:
                log.info("difficulty_scoring_progress", chunks_done=chunk_idx + 1, total=total_chunks)

        return scores
    finally:
        vgae_model.train(was_training)
