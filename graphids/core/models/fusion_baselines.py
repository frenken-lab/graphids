"""Supervised fusion baselines: MLP and weighted average.

These consume the same 15-D state vector as the DQN agent but train with
standard supervised losses instead of RL episodes.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from ._training import binary_test_metrics


class MLPFusionNetwork(nn.Module):
    """Simple MLP for binary classification from fusion state vectors."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPFusionModule(pl.LightningModule):
    """Supervised MLP baseline: binary classification from fusion state vectors.

    Same state as DQN, but trained with BCE loss via Lightning instead of RL episodes.
    """

    def __init__(
        self,
        state_dim: int = 0,
        hidden_dims: tuple[int, ...] = (64, 32),
        lr: float = 0.001,
        # --- identity key metadata (for run directory hashing) ---
        scale: str = "small",
        gat_stage: str = "curriculum",
        loss_fn: str = "ce",
        conv_type: str = "gatv2",
        variational: bool = True,
    ):
        super().__init__()
        if state_dim == 0:
            from .registry import fusion_state_dim
            state_dim = fusion_state_dim()
        self.save_hyperparameters()
        self.model = MLPFusionNetwork(state_dim, hidden_dims)
        self._loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.test_metrics = binary_test_metrics()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self._loss_fn(logits, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        loss = self._loss_fn(logits, labels.float())
        preds = (logits > 0).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        logits = self(states)
        preds = (logits > 0).long()
        self.test_metrics.update(preds, labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self(t)
            return 1 if logit.item() > 0 else 0


class WeightedAvgModule(pl.LightningModule):
    """Simplest baseline: learns a single scalar alpha per model.

    If this matches DQN's F1, the RL approach is unjustified.
    Fusion: score = (1 - sigmoid(w)) * vgae_conf + sigmoid(w) * gat_conf
    """

    def __init__(
        self,
        lr: float = 0.01,
        decision_threshold: float = 0.5,
        # --- identity key metadata (for run directory hashing) ---
        scale: str = "small",
        gat_stage: str = "curriculum",
        loss_fn: str = "ce",
        conv_type: str = "gatv2",
        variational: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.weight = nn.Parameter(torch.zeros(1))
        self._loss_fn = nn.BCELoss()
        self.lr = lr
        self.decision_threshold = decision_threshold
        self.test_metrics = binary_test_metrics()

        from .registry import feature_layout

        layout = feature_layout()
        self._vgae_conf_idx = layout["vgae"].confidence_idx
        self._gat_conf_idx = layout["gat"].confidence_idx

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.weight)
        vgae_conf = states[:, self._vgae_conf_idx]
        gat_conf = states[:, self._gat_conf_idx]
        return torch.clamp((1 - alpha) * vgae_conf + alpha * gat_conf, 1e-7, 1 - 1e-7)

    def training_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self._loss_fn(scores, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        self.log("alpha", torch.sigmoid(self.weight).item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        loss = self._loss_fn(scores, labels.float())
        preds = (scores > self.decision_threshold).long()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        scores = self(states)
        preds = (scores > self.decision_threshold).long()
        self.test_metrics.update(preds, labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fuse(self, state_features: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            alpha = torch.sigmoid(self.weight).item()
            vgae_conf = state_features[self._vgae_conf_idx]
            gat_conf = state_features[self._gat_conf_idx]
            score = (1 - alpha) * vgae_conf + alpha * gat_conf
            return 1 if score > self.decision_threshold else 0


class RLFusionModule(pl.LightningModule):
    """Lightning wrapper for RL fusion agents (DQN, bandit).

    Uses manual optimization. Both agents implement ``train_episode(states, labels)``
    returning a metrics dict. All returned keys are logged automatically.

    Constructor accepts config + method so Lightning can round-trip through
    ``save_hyperparameters`` / ``load_from_checkpoint``.
    """

    def __init__(
        self,
        method: str = "bandit",
        # --- fusion ---
        episodes: int = 500,
        max_samples: int = 150_000,
        max_val_samples: int = 30_000,
        episode_sample_size: int = 20_000,
        training_step_interval: int = 32,
        gpu_training_steps: int = 16,
        lr: float = 0.001,
        alpha_steps: int = 21,
        decision_threshold: float = 0.5,
        # --- dqn ---
        dqn_hidden: int = 576,
        dqn_layers: int = 3,
        dqn_gamma: float = 0.99,
        dqn_epsilon: float = 0.1,
        dqn_epsilon_decay: float = 0.995,
        dqn_min_epsilon: float = 0.01,
        dqn_buffer_size: int = 100_000,
        dqn_batch_size: int = 128,
        dqn_target_update: int = 100,
        dqn_weight_decay: float = 1e-5,
        dqn_scheduler_patience: int = 1000,
        dqn_vgae_error_weights: list[float] | None = None,
        dqn_reward_correct: float = 3.0,
        dqn_reward_incorrect: float = -3.0,
        dqn_confidence_weight: float = 0.5,
        dqn_combined_conf_weight: float = 0.3,
        dqn_disagreement_penalty: float = -1.0,
        dqn_overconf_penalty: float = -1.5,
        dqn_balance_weight: float = 0.3,
        # --- bandit ---
        bandit_ucb_alpha: float = 1.0,
        bandit_lambda_reg: float = 1.0,
        bandit_backbone_retrain_freq: int = 50,
        bandit_backbone_lr: float = 0.001,
        bandit_backbone_epochs: int = 5,
        bandit_hidden: int = 576,
        bandit_layers: int = 3,
        bandit_buffer_size: int = 100_000,
        bandit_batch_size: int = 128,
        # --- identity key metadata (for run directory hashing) ---
        scale: str = "small",
        gat_stage: str = "curriculum",
        loss_fn: str = "ce",
        conv_type: str = "gatv2",
        variational: bool = True,
        # ---
        device: str = "cpu",
    ):
        super().__init__()
        if dqn_vgae_error_weights is None:
            dqn_vgae_error_weights = [0.4, 0.35, 0.25]
        self.save_hyperparameters()

        self.automatic_optimization = False

        if method == "dqn":
            from .dqn import EnhancedDQNFusionAgent
            agent = EnhancedDQNFusionAgent.from_config(self.hparams, device=device)
        elif method == "bandit":
            from .bandit import NeuralLinUCBAgent
            agent = NeuralLinUCBAgent.from_config(self.hparams, device=device)
        else:
            raise ValueError(f"RLFusionModule only handles dqn/bandit, got: {method}")

        self._optimizer_attr = "optimizer" if method == "dqn" else "backbone_optimizer"
        self.agent = agent
        self.test_metrics = binary_test_metrics()

    def training_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.train_episode(states, labels)
        for k, v in result.items():
            if v is not None:
                self.log(k, float(v), prog_bar=(k in ("avg_reward", "accuracy")))

    def validation_step(self, batch, batch_idx):
        states, labels = batch
        metrics = self.agent.validate_batch(states, labels)
        self.log("val_acc", metrics.get("accuracy", 0.0), prog_bar=True)

    def test_step(self, batch, batch_idx):
        states, labels = batch
        result = self.agent.predict(states)
        self.test_metrics.update(result["preds"], labels)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def on_save_checkpoint(self, checkpoint):
        checkpoint["agent_state"] = self.agent.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "agent_state" in checkpoint:
            self.agent.load_checkpoint(checkpoint["agent_state"])

    def configure_optimizers(self):
        return getattr(self.agent, self._optimizer_attr)



