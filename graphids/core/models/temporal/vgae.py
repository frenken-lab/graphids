"""Temporal variational event autoencoder for CAN streams."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TemporalModuleBase


class TemporalVGAE(TemporalModuleBase):
    """Recurrent variational autoencoder that scores event-level surprise."""

    _SCALES: dict[str, dict[str, int]] = {
        "small": {"hidden": 64, "layers": 1, "embedding_dim": 16, "latent_dim": 32},
        "large": {"hidden": 128, "layers": 2, "embedding_dim": 32, "latent_dim": 64},
    }

    def __init__(
        self,
        *,
        hidden: int | None = None,
        layers: int | None = None,
        embedding_dim: int | None = None,
        latent_dim: int | None = None,
        dropout: float = 0.1,
        kl_weight: float = 0.01,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scale: str = "small",
        model_type: str = "temporal_vgae",
        dataset: str = "",
        seed: int = 42,
        num_ids: int = 0,
        in_channels: int = 0,
        num_classes: int = 2,
    ):
        preset = self._SCALES.get(scale, {})
        hidden = hidden if hidden is not None else preset.get("hidden", 64)
        layers = layers if layers is not None else preset.get("layers", 1)
        embedding_dim = embedding_dim if embedding_dim is not None else preset.get("embedding_dim", 16)
        latent_dim = latent_dim if latent_dim is not None else preset.get("latent_dim", 32)
        super().__init__()
        self._init_post(locals())

    def _build(self) -> None:
        hp = self.hparams
        self.src_embedding = nn.Embedding(max(1, int(hp.num_ids)), int(hp.embedding_dim))
        self.dst_embedding = nn.Embedding(max(1, int(hp.num_ids)), int(hp.embedding_dim))
        input_dim = int(hp.in_channels) + (2 * int(hp.embedding_dim))
        hidden = int(hp.hidden)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            hidden,
            hidden,
            num_layers=int(hp.layers),
            dropout=float(hp.dropout) if int(hp.layers) > 1 else 0.0,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden, int(hp.latent_dim))
        self.logvar = nn.Linear(hidden, int(hp.latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(int(hp.latent_dim), hidden),
            nn.GELU(),
            nn.Dropout(float(hp.dropout)),
            nn.Linear(hidden, int(hp.in_channels)),
        )

    def _event_input(self, batch) -> torch.Tensor:
        src = batch.src.clamp_min(0).clamp_max(self.src_embedding.num_embeddings - 1).long()
        dst = batch.dst.clamp_min(0).clamp_max(self.dst_embedding.num_embeddings - 1).long()
        return torch.cat([batch.msg.float(), self.src_embedding(src), self.dst_embedding(dst)], dim=-1)

    def forward_temporal(self, batch, state=None) -> dict[str, torch.Tensor]:
        x = self.input_proj(self._event_input(batch)).unsqueeze(0)
        h, state = self.rnn(x, state)
        h = h.squeeze(0)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        recon = self.decoder(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)
        return {"recon": recon, "mu": mu, "logvar": logvar, "kl": kl, "state": state}

    def forward(self, batch) -> dict[str, torch.Tensor]:
        return self.forward_temporal(batch)

    def _components(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self(batch)
        recon_per_event = F.mse_loss(out["recon"], batch.msg.float(), reduction="none").mean(dim=-1)
        return recon_per_event, out["kl"], out["recon"]

    def score(self, batch) -> torch.Tensor:
        recon, kl, _decoded = self._components(batch)
        return recon + (float(self.hparams.kl_weight) * kl)

    def training_step(self, batch, _idx):
        mask = self.scored_mask(batch)
        recon, kl, _decoded = self._components(batch)
        if not mask.any():
            return recon.sum() * 0.0
        loss = recon[mask].mean() + (float(self.hparams.kl_weight) * kl[mask].mean())
        bs = int(mask.sum().item())
        self.log("train_loss", loss, batch_size=bs)
        self.log("train_recon", recon[mask].mean(), batch_size=bs)
        self.log("train_kl", kl[mask].mean(), batch_size=bs)
        return loss

    def validation_step(self, batch, _idx):
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        recon, kl, _decoded = self._components(batch)
        scores = recon + (float(self.hparams.kl_weight) * kl)
        labels = batch.y[mask].long()
        bs = int(labels.numel())
        self.log("val_loss", scores[mask].mean(), batch_size=bs)
        self.log("val_recon_mean", recon[mask].mean(), batch_size=bs)
        self.log("val_kl_mean", kl[mask].mean(), batch_size=bs)
        if labels.unique().numel() >= 2:
            from torchmetrics.functional.classification import binary_auroc

            self.log("val_auroc", binary_auroc(scores[mask], labels), batch_size=bs)
        return None

    def test_step(self, batch, _idx, dataloader_idx=0):
        mask = self.scored_mask(batch)
        if not mask.any():
            return None
        scores = self.score(batch)[mask]
        labels = batch.y[mask].long()
        attack_type = getattr(batch, "attack_type", None)
        self._record_test_batch(
            dataloader_idx,
            scores=scores,
            labels=labels,
            attack_type=attack_type[mask] if attack_type is not None else None,
        )
        return None

    def predict_step(self, batch, _idx):
        out = {"scores": self.score(batch), "labels": batch.y}
        event_id = getattr(batch, "event_id", None)
        if event_id is not None:
            out["event_id"] = event_id
        return out
