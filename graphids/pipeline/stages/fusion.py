"""Fusion stage: combines VGAE + GAT predictions via configurable method (DQN, MLP, weighted_avg)."""

from __future__ import annotations

import structlog
from pathlib import Path

import torch

from graphids.config import PipelineConfig

from .data_loading import training_preamble
from .data_loading import cache_predictions, cleanup
from .trainer_factory import load_model

log = structlog.get_logger()


def _save_dqn_ckpt(agent) -> None:
    torch.save({
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "epsilon": agent.epsilon,
    }, "best_model.pt")


def _train_dqn_fusion(cfg, train_cache, val_cache, device, out) -> float:
    """Vectorized DQN RL fusion training loop. Returns best validation accuracy.

    Each episode: batch forward pass -> batch reward -> store in tensor buffer
    -> gradient steps from buffer. No Python-level per-sample loop.
    """
    from graphids.core.models.dqn import EnhancedDQNFusionAgent

    agent = EnhancedDQNFusionAgent.from_config(cfg, device=str(device))

    best_acc = 0.0
    val_states = val_cache["states"][: min(5000, len(val_cache["states"]))]
    val_labels = val_cache["labels"][: min(5000, len(val_cache["labels"]))]

    for ep in range(cfg.fusion.episodes):
        idx = torch.randperm(len(train_cache["states"]))[: cfg.fusion.episode_sample_size]
        batch_states = train_cache["states"][idx]
        batch_labels = train_cache["labels"][idx]

        # Vectorized: one forward pass for all samples, batch reward, batch store
        actions, alphas, norm_states = agent.select_action_batch(batch_states, training=True)
        # TODO(open-question): Training uses (alpha > 0.5) as prediction, but
        # validation uses the proper fused score. See dqn.py top-level comment.
        preds = (alphas > 0.5).long()
        rewards = agent.compute_fusion_reward_batch(preds, batch_labels, norm_states, alphas)
        agent.store_experiences_batch(norm_states, actions, rewards)

        # Gradient steps from replay buffer
        if agent.buffer_size_current >= cfg.dqn.batch_size:
            for _ in range(cfg.fusion.gpu_training_steps):
                agent.train_step()

        # Epsilon decay
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

        if (ep + 1) % 50 == 0:
            metrics = agent.validate_batch(val_states, val_labels)
            acc = metrics.get("accuracy", 0)
            log.info(
                "dqn_episode",
                episode=ep + 1,
                total_episodes=cfg.fusion.episodes,
                avg_reward=round(rewards.mean().item(), 2),
                val_acc=round(acc, 4),
                epsilon=round(agent.epsilon, 3),
            )

            if acc > best_acc:
                best_acc = acc
                _save_dqn_ckpt(agent)

    # Ensure we always save something
    if not Path("best_model.pt").exists():
        _save_dqn_ckpt(agent)

    return best_acc


def _make_fusion_trainer(cfg):
    """Create a lightweight Lightning Trainer for fusion baselines (MLP/WeightedAvg)."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    return pl.Trainer(
        default_root_dir=".",
        max_epochs=cfg.fusion.mlp_max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=".", filename="best_model",
                monitor="val_loss", mode="min", save_top_k=1,
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        logger=pl.loggers.CSVLogger(save_dir=".", name="", version=""),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )


def _make_fusion_dataloaders(train_cache, val_cache, batch_size: int):
    """Build train/val DataLoaders from cached prediction tensors."""
    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(train_cache["states"], train_cache["labels"])
    val_ds = TensorDataset(val_cache["states"], val_cache["labels"])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _train_mlp_fusion(cfg, train_cache, val_cache, device) -> float:
    """MLP supervised fusion via Lightning Trainer. Returns best validation accuracy."""
    from graphids.core.models.dqn import MLPFusionModule
    from graphids.core.models.registry import fusion_state_dim

    module = MLPFusionModule(
        state_dim=fusion_state_dim(),
        hidden_dims=cfg.fusion.mlp_hidden_dims,
        lr=cfg.fusion.lr,
    )
    train_dl, val_dl = _make_fusion_dataloaders(
        train_cache, val_cache, cfg.dqn.batch_size,
    )
    trainer = _make_fusion_trainer(cfg)
    trainer.fit(module, train_dl, val_dl)

    torch.save({"model": module.model.state_dict()}, "best_model.pt")
    best_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    return best_acc


def _train_weighted_avg_fusion(cfg, train_cache, val_cache, device) -> float:
    """Weighted average fusion via Lightning Trainer. Returns best validation accuracy."""
    from graphids.core.models.dqn import WeightedAvgModule

    module = WeightedAvgModule(lr=cfg.fusion.lr)
    train_dl, val_dl = _make_fusion_dataloaders(
        train_cache, val_cache, cfg.dqn.batch_size,
    )
    trainer = _make_fusion_trainer(cfg)
    trainer.fit(module, train_dl, val_dl)

    torch.save(module.state_dict_for_save(), "best_model.pt")
    best_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    return best_acc


def train_fusion(cfg: PipelineConfig) -> dict:
    """Train fusion agent on cached VGAE+GAT predictions. Returns result dict with checkpoint and metrics."""
    train_data, val_data, num_ids, in_ch, device = training_preamble(
        cfg, f"FUSION ({cfg.fusion.method})"
    )

    # Load frozen VGAE + GAT
    vgae = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, device)
    gat = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)

    # Cache predictions
    log.info("Caching VGAE + GAT predictions ...")
    models = {"vgae": vgae, "gat": gat}
    train_cache = cache_predictions(models, train_data, device, cfg.fusion.max_samples)
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)
    del vgae, gat
    cleanup()

    # Dispatch on fusion method
    method = cfg.fusion.method
    if method == "dqn":
        best_acc = _train_dqn_fusion(cfg, train_cache, val_cache, device, Path.cwd())
    elif method == "mlp":
        best_acc = _train_mlp_fusion(cfg, train_cache, val_cache, device)
    elif method == "weighted_avg":
        best_acc = _train_weighted_avg_fusion(cfg, train_cache, val_cache, device)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    ckpt = Path("best_model.pt")
    cfg.save(Path("config.json"))

    metrics = {"best_acc": best_acc, "val_loss": 1.0 - best_acc, "fusion_method": method}
    log.info("saved_fusion", method=method, checkpoint=str(ckpt), best_acc=round(best_acc, 4))
    cleanup()
    return {"checkpoint": str(ckpt), "metrics": metrics}
