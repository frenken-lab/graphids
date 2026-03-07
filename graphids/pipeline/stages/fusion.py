"""Fusion stage: combines VGAE + GAT predictions via configurable method (DQN, MLP, weighted_avg)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from graphids.config import PipelineConfig, checkpoint_path, config_path, stage_dir

from .data_loading import training_preamble
from .utils import cache_predictions, cleanup, load_model

log = logging.getLogger(__name__)


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
                "Episode %d/%d  avg_reward=%.2f  val_acc=%.4f  eps=%.3f",
                ep + 1,
                cfg.fusion.episodes,
                rewards.mean().item(),
                acc,
                agent.epsilon,
            )

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {
                        "q_network": agent.q_network.state_dict(),
                        "target_network": agent.target_network.state_dict(),
                        "epsilon": agent.epsilon,
                    },
                    checkpoint_path(cfg, "fusion"),
                )

    # Ensure we always save something
    ckpt = checkpoint_path(cfg, "fusion")
    if not ckpt.exists():
        torch.save(
            {
                "q_network": agent.q_network.state_dict(),
                "target_network": agent.target_network.state_dict(),
                "epsilon": agent.epsilon,
            },
            ckpt,
        )

    return best_acc


def _train_mlp_fusion(cfg, train_cache, val_cache, device) -> float:
    """MLP supervised fusion. Returns best validation accuracy."""
    from graphids.core.models.dqn import MLPFusionAgent
    from graphids.core.models.registry import fusion_state_dim

    agent = MLPFusionAgent(
        state_dim=fusion_state_dim(),
        hidden_dims=cfg.fusion.mlp_hidden_dims,
        lr=cfg.fusion.lr,
        device=str(device),
    )
    best_acc = agent.train_on_cache(
        train_cache["states"],
        train_cache["labels"],
        val_cache["states"],
        val_cache["labels"],
        cfg,
    )
    torch.save(agent.state_dict(), checkpoint_path(cfg, "fusion"))
    return best_acc


def _train_weighted_avg_fusion(cfg, train_cache, val_cache, device) -> float:
    """Weighted average fusion. Returns best validation accuracy."""
    from graphids.core.models.dqn import WeightedAvgFusionAgent

    agent = WeightedAvgFusionAgent(device=str(device))
    best_acc = agent.train_on_cache(
        train_cache["states"],
        train_cache["labels"],
        val_cache["states"],
        val_cache["labels"],
        cfg,
    )
    torch.save(agent.state_dict(), checkpoint_path(cfg, "fusion"))
    return best_acc


def train_fusion(cfg: PipelineConfig) -> Path:
    """Train fusion agent on cached VGAE+GAT predictions. Returns checkpoint path."""
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

    out = stage_dir(cfg, "fusion")
    out.mkdir(parents=True, exist_ok=True)

    # Dispatch on fusion method
    method = cfg.fusion.method
    if method == "dqn":
        best_acc = _train_dqn_fusion(cfg, train_cache, val_cache, device, out)
    elif method == "mlp":
        best_acc = _train_mlp_fusion(cfg, train_cache, val_cache, device)
    elif method == "weighted_avg":
        best_acc = _train_weighted_avg_fusion(cfg, train_cache, val_cache, device)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    ckpt = checkpoint_path(cfg, "fusion")
    cfg.save(config_path(cfg, "fusion"))

    # Write metrics.json (consistent with training.py stages, needed by tune trainable)
    import json

    metrics_out = out / "metrics.json"
    metrics_out.write_text(
        json.dumps(
            {"best_acc": best_acc, "val_loss": 1.0 - best_acc, "fusion_method": method},
            indent=2,
        )
    )
    log.info("Saved %s fusion: %s (best_acc=%.4f)", method, ckpt, best_acc)
    cleanup()
    return ckpt
