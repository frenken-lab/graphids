"""Module-level smoke tests for VGAE, GAT, and DQN training.

Fast tests (< 15s each) that verify models train for 2 epochs without crashing,
produce finite losses, and respect config parameters.

Run:  python -m pytest tests/test_training_smoke.py -v -m "not slow"
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from tests.conftest import (
    IN_CHANNELS,
    NUM_IDS,
    SMOKE_OVERRIDES,
    _make_dataset,
    _make_graph,
)


@pytest.mark.slurm
class TestVGAESmoke:
    """VGAE module trains and produces finite loss."""

    def test_trains(self):
        from graphids.config import resolve
        from graphids.pipeline.stages.modules import VGAEModule

        cfg = resolve("vgae", "large", **SMOKE_OVERRIDES)
        data = _make_dataset(20)
        module = VGAEModule(cfg, NUM_IDS, IN_CHANNELS)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(
            module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4)
        )

        assert trainer.callback_metrics.get("train_loss") is not None
        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_kd_trains(self):
        from graphids.config import resolve
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood
        from graphids.pipeline.stages.modules import VGAEModule
        from graphids.pipeline.stages.utils import make_projection

        teacher_cfg = resolve("vgae", "large", **SMOKE_OVERRIDES)
        student_cfg = resolve(
            "vgae",
            "small",
            auxiliaries="kd_standard",
            **SMOKE_OVERRIDES,
        )

        t_conv = teacher_cfg.vgae.conv_type
        teacher = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(teacher_cfg.vgae.hidden_dims),
            latent_dim=teacher_cfg.vgae.latent_dim,
            encoder_heads=teacher_cfg.vgae.heads,
            embedding_dim=teacher_cfg.vgae.embedding_dim,
            dropout=teacher_cfg.vgae.dropout,
            conv_type=t_conv,
            edge_dim=teacher_cfg.vgae.edge_dim if t_conv in ("gatv2", "transformer") else None,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        s_conv = student_cfg.vgae.conv_type
        student_model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(student_cfg.vgae.hidden_dims),
            latent_dim=student_cfg.vgae.latent_dim,
            encoder_heads=student_cfg.vgae.heads,
            embedding_dim=student_cfg.vgae.embedding_dim,
            dropout=student_cfg.vgae.dropout,
            conv_type=s_conv,
            edge_dim=student_cfg.vgae.edge_dim if s_conv in ("gatv2", "transformer") else None,
        )
        projection = make_projection(student_model, teacher, "vgae", torch.device("cpu"))
        del student_model

        module = VGAEModule(
            student_cfg, NUM_IDS, IN_CHANNELS, teacher=teacher, projection=projection
        )
        data = _make_dataset(20)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(
            module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4)
        )

        assert torch.isfinite(trainer.callback_metrics["train_loss"])


@pytest.mark.slurm
class TestGATSmoke:
    """GAT module trains and produces finite loss."""

    def test_trains(self):
        from graphids.config import resolve
        from graphids.pipeline.stages.modules import GATModule

        cfg = resolve("gat", "large", **SMOKE_OVERRIDES)
        data = _make_dataset(20)
        module = GATModule(cfg, NUM_IDS, IN_CHANNELS)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(
            module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4)
        )

        assert trainer.callback_metrics.get("train_loss") is not None
        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_kd_trains(self):
        from graphids.config import resolve
        from graphids.core.models.gat import GATWithJK
        from graphids.pipeline.stages.modules import GATModule

        teacher_cfg = resolve("gat", "large", **SMOKE_OVERRIDES)
        student_cfg = resolve(
            "gat",
            "small",
            auxiliaries="kd_standard",
            **SMOKE_OVERRIDES,
        )

        t_conv = teacher_cfg.gat.conv_type
        teacher = GATWithJK(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_channels=teacher_cfg.gat.hidden,
            out_channels=2,
            num_layers=teacher_cfg.gat.layers,
            heads=teacher_cfg.gat.heads,
            dropout=teacher_cfg.gat.dropout,
            num_fc_layers=teacher_cfg.gat.fc_layers,
            embedding_dim=teacher_cfg.gat.embedding_dim,
            conv_type=t_conv,
            edge_dim=teacher_cfg.gat.edge_dim if t_conv in ("gatv2", "transformer") else None,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        module = GATModule(student_cfg, NUM_IDS, IN_CHANNELS, teacher=teacher)
        data = _make_dataset(20)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(
            module, DataLoader(data[:15], batch_size=4), DataLoader(data[15:], batch_size=4)
        )

        assert torch.isfinite(trainer.callback_metrics["train_loss"])

    def test_fc_layers_config(self):
        """GATWithJK should work with different num_fc_layers values."""
        from graphids.core.models.gat import GATWithJK

        g = _make_graph()
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

        for fc_layers in [1, 2, 3]:
            model = GATWithJK(
                num_ids=NUM_IDS,
                in_channels=IN_CHANNELS,
                hidden_channels=24,
                out_channels=2,
                num_layers=2,
                heads=4,
                dropout=0.1,
                num_fc_layers=fc_layers,
                embedding_dim=8,
            )
            model.eval()
            with torch.no_grad():
                out = model(g)
            assert out.shape == (1, 2), f"fc_layers={fc_layers} gave wrong shape: {out.shape}"


@pytest.mark.slurm
class TestDQNSmoke:
    """DQN trains and produces finite loss."""

    def test_trains(self):
        import numpy as np

        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.registry import fusion_state_dim

        agent = EnhancedDQNFusionAgent(
            alpha_steps=21,
            lr=1e-3,
            gamma=0.99,
            epsilon=0.5,
            epsilon_decay=0.99,
            min_epsilon=0.01,
            buffer_size=500,
            batch_size=32,
            target_update_freq=10,
            device="cpu",
            state_dim=fusion_state_dim(),
            hidden_dim=64,
            num_layers=2,
        )

        # Fill replay buffer
        for _ in range(100):
            state = np.random.randn(15).astype(np.float32)
            alpha, action_idx, proc_state = agent.select_action(state, training=True)
            reward = 1.0 if np.random.random() > 0.5 else -1.0
            agent.store_experience(proc_state, action_idx, reward, proc_state, False)

        # Train a few steps
        losses = []
        for _ in range(10):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        assert len(losses) > 0, "DQN did not produce any training losses"
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), "DQN loss is not finite"
