"""Trainer factory, KD preparation, and model loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
import torch.nn as nn

from graphids.config import STAGE_MODEL_MAP

log = structlog.get_logger()

# model_type → canonical stage that produces the teacher checkpoint.
_TEACHER_STAGE: dict[str, str] = {}
for _stage, _model in STAGE_MODEL_MAP.items():
    _TEACHER_STAGE.setdefault(_model, _stage)


def make_trainer(cfg, stage: str, **overrides) -> pl.Trainer:
    """Create a Lightning Trainer from config with optional overrides.

    Stages like fusion pass overrides for callbacks, max_epochs, logger, etc.
    while inheriting precision, gradient_clip, deterministic, etc. from cfg.training.
    """
    if "callbacks" not in overrides:
        from hydra.utils import instantiate
        overrides["callbacks"] = [cb for cb in instantiate(cfg.callbacks).values() if cb is not None]

    kwargs = dict(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.training.gradient_clip,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.cudnn_benchmark,
        enable_progress_bar=not bool(os.environ.get("SLURM_JOB_ID")),
    )
    kwargs.update(overrides)
    return pl.Trainer(**kwargs)


def build_datamodule(cfg, stage: str) -> tuple[pl.LightningDataModule, torch.device]:
    """Build DataModule + resolve device for any training stage."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if stage == "fusion":
        from .fusion import FusionDataModule
        dm = FusionDataModule(cfg)
        dm.setup("fit")
        return dm, dm.device

    # All graph-based stages start from CANBusDataModule
    from graphids.core.preprocessing import CANBusDataModule
    raw_dm = CANBusDataModule.from_cfg(cfg)
    raw_dm.setup("fit")
    raw_dm.populate_config(cfg)

    if stage == "curriculum":
        dm = _build_curriculum_dm(raw_dm, cfg, device)
        return dm, device

    return raw_dm, device


def _build_curriculum_dm(raw_dm, cfg, device):
    """Score difficulty with VGAE, build CurriculumDataModule."""
    import gc

    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from torch_geometric.utils import scatter

    from .eval_inference import graph_label
    from .modules import CurriculumDataModule

    vgae = load_model(cfg, "vgae", "autoencoder", device)
    normals = [g for g in raw_dm.train_dataset if graph_label(g) == 0]
    attacks = [g for g in raw_dm.train_dataset if graph_label(g) == 1]

    # Score difficulty via VGAE reconstruction error
    scores: list[float] = []
    was_training = vgae.training
    vgae.eval()
    try:
        chunk_size = 500
        canid_weight = cfg.vgae.canid_weight
        for start in range(0, len(normals), chunk_size):
            chunk = normals[start : start + chunk_size]
            with torch.no_grad():
                batch = Batch.from_data_list([g.clone() for g in chunk]).to(device, non_blocking=True)
                edge_attr = getattr(batch, "edge_attr", None)
                cont, canid_logits, _, _, _, _ = vgae(
                    batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr,
                    node_id=batch.node_id,
                )
                node_mse = (cont - batch.x).pow(2).mean(dim=1)
                graph_mse = scatter(node_mse, batch.batch, reduce="mean")
                node_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
                graph_ce = scatter(node_ce, batch.batch, reduce="mean")
                scores.extend((graph_mse + canid_weight * graph_ce).tolist())
                del batch
                torch.cuda.empty_cache()
    finally:
        vgae.train(was_training)

    del vgae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return CurriculumDataModule(normals, attacks, scores, list(raw_dm.val_dataset), cfg)


def build_module(cfg, stage: str, device: torch.device) -> pl.LightningModule:
    """Build the Lightning module for any training stage."""
    if stage == "autoencoder":
        from .modules import DGIModule, VGAEModule
        if cfg.model_type == "dgi":
            return DGIModule(cfg)
        teacher, projection = prepare_kd(cfg, "vgae", device)
        return VGAEModule(cfg, teacher=teacher, projection=projection)
    elif stage in ("normal", "curriculum"):
        from .modules import GATModule
        teacher, _ = prepare_kd(cfg, "gat", device)
        return GATModule(cfg, teacher=teacher)
    elif stage == "fusion":
        return _build_fusion_module(cfg, device)
    else:
        raise ValueError(f"Unknown training stage: {stage}")


def _build_fusion_module(cfg, device: torch.device) -> pl.LightningModule:
    """Build fusion module per method."""
    method = cfg.fusion.method
    if method == "dqn":
        from .fusion import RLFusionModule
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        agent = EnhancedDQNFusionAgent.from_config(cfg, device=str(device))
        return RLFusionModule(agent, "optimizer")
    elif method == "bandit":
        from .fusion import RLFusionModule
        from graphids.core.models.bandit import NeuralLinUCBAgent
        agent = NeuralLinUCBAgent.from_config(cfg, device=str(device))
        return RLFusionModule(agent, "backbone_optimizer")
    elif method == "mlp":
        from graphids.core.models.fusion_baselines import MLPFusionModule
        from graphids.core.models.registry import fusion_state_dim
        return MLPFusionModule(state_dim=fusion_state_dim(), hidden_dims=cfg.fusion.mlp_hidden_dims, lr=cfg.fusion.lr)
    elif method == "weighted_avg":
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        return WeightedAvgModule(lr=cfg.fusion.lr, decision_threshold=cfg.fusion.decision_threshold)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def prepare_kd(
    cfg, model_type: str, device: torch.device,
) -> tuple[nn.Module | None, nn.Linear | None]:
    """Resolve teacher path, load & freeze teacher, create projection if needed.

    Reads num_ids and in_channels from cfg (populated by DataModule.populate_config).
    Returns (teacher, projection) when KD is active, or (None, None) otherwise.
    """
    if not any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        return None, None

    num_ids, in_channels = cfg.num_ids, cfg.in_channels

    # --- Resolve teacher checkpoint path ---
    kd = next(a for a in cfg.get("auxiliaries", []) if a.type == "kd")
    if kd.get("model_path"):
        teacher_path = Path(kd.model_path)
    else:
        teacher_scale = kd.get("teacher_scale", "large")
        stage = _TEACHER_STAGE.get(model_type)
        if stage is None:
            raise ValueError(f"No teacher stage mapping for model_type '{model_type}'")
        from graphids.config import resolve
        teacher_cfg = resolve(f"model_type={model_type}", f"scale={teacher_scale}",
                              f"dataset={cfg.dataset}", f"seed={cfg.seed}")
        teacher_path = Path(teacher_cfg.checkpoints[model_type])
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_path}. "
                f"Train {model_type}/{teacher_scale} first, or set model_path explicitly."
            )

    # --- Load and freeze teacher ---
    from graphids.core.models.registry import get as registry_get
    from omegaconf import OmegaConf

    checkpoint = torch.load(str(teacher_path), map_location="cpu", weights_only=True)
    sd = checkpoint
    if isinstance(sd, dict) and "state_dict" in sd:
        raw = sd["state_dict"]
        sd = {k.replace("model.", ""): v for k, v in raw.items() if k.startswith("model.")} or raw

    tcfg_path = teacher_path.parent / "config.yaml"
    if not tcfg_path.exists():
        raise FileNotFoundError(f"Teacher config not found: {tcfg_path}")
    tcfg = OmegaConf.load(tcfg_path)

    t_num_ids = tcfg.get("num_ids", num_ids)

    teacher = registry_get(model_type)(tcfg, t_num_ids, in_channels)

    if model_type == "dqn":
        teacher.load_state_dict(sd.get("q_network") or sd.get("q_network_state_dict") or sd)
    else:
        teacher.load_state_dict(sd)

    log.info("loaded_teacher", model_type=model_type, path=str(teacher_path), num_ids=t_num_ids)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Projection layer (VGAE latent space alignment) ---
    projection = None
    if model_type == "vgae":
        s_dim = cfg.vgae.latent_dim
        t_dim = tcfg.vgae.latent_dim
        if s_dim != t_dim:
            log.info("projection_layer", student_dim=s_dim, teacher_dim=t_dim)
            projection = nn.Linear(s_dim, t_dim).to(device)

    return teacher, projection


def load_frozen_cfg(cfg, stage: str, model_type: str | None = None):
    """Load the frozen config.yaml saved by a prior training stage."""
    mt = model_type or STAGE_MODEL_MAP.get(stage, stage)
    p = Path(cfg.checkpoints.get(mt, "")).parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Frozen config not found: {p}\n"
            f"The '{mt}/{stage}' stage must be trained first."
        )
    from omegaconf import OmegaConf
    return OmegaConf.load(p)


def load_model(
    cfg, model_type: str, stage: str, device: torch.device,
) -> nn.Module:
    """Load a trained model using its frozen config and the registry.

    Reads num_ids and in_channels from cfg (populated by DataModule.populate_config).
    """
    from graphids.core.models.registry import get as registry_get

    ckpt_path = Path(cfg.checkpoints[model_type])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"The '{model_type}/{stage}' stage must be trained first."
        )
    frozen_cfg = load_frozen_cfg(cfg, stage, model_type=model_type)
    model = registry_get(model_type)(frozen_cfg, cfg.num_ids, cfg.in_channels)
    model.load_state_dict(torch.load(cfg.checkpoints[model_type], map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model


def build_optimizer_dict(optimizer, cfg):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    if not cfg.training.use_scheduler or cfg.training.scheduler is None:
        return optimizer

    from hydra.utils import instantiate
    sched = instantiate(cfg.training.scheduler, optimizer=optimizer)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "monitor": cfg.training.monitor_metric}}
