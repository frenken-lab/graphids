"""Lightning modules for VGAE and GAT training."""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphids.config import PipelineConfig

from .utils import (
    build_optimizer_dict,
    compute_node_budget,
    effective_batch_size,
    make_dataloader,
)

log = logging.getLogger(__name__)


def _teacher_to_device(teacher, device, on_cpu_flag, cfg):
    """Move teacher to device if offloaded, return updated on_cpu flag."""
    if cfg.training.offload_teacher_to_cpu and on_cpu_flag:
        teacher.to(device)
        return False
    return on_cpu_flag


def _teacher_offload(teacher, cfg):
    """Offload teacher to CPU after forward pass, return on_cpu=True."""
    if cfg.training.offload_teacher_to_cpu:
        teacher.to("cpu")
        torch.cuda.empty_cache()
        return True
    return False


class VGAEModule(pl.LightningModule):
    """VGAE training: reconstruct node features + CAN IDs + neighborhood.

    When teacher is provided, adds dual-signal KD loss:
      kd_loss = latent_w * MSE(project(z_s), z_t) + recon_w * MSE(recon_s, recon_t)
      total = alpha * kd_loss + (1-alpha) * task_loss

    Memory optimization: When cfg.training.offload_teacher_to_cpu is True, the teacher
    model is moved to CPU after each forward pass to free GPU memory.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        num_ids: int,
        in_channels: int,
        teacher: nn.Module | None = None,
        projection: nn.Linear | None = None,
    ):
        super().__init__()
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        self.cfg = cfg
        self.model = GraphAutoencoderNeighborhood.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.teacher = teacher
        self.projection = projection
        self._teacher_on_cpu = False

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        return self.model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)

    def _task_loss(self, batch):
        cont_out, canid_logits, nbr_logits, z, kl_loss = self(batch)
        recon = F.mse_loss(cont_out, batch.x[:, 1:])
        canid = F.cross_entropy(canid_logits, batch.x[:, 0].long())
        nbr_targets = self.model.create_neighborhood_targets(batch.x, batch.edge_index, batch.batch)
        nbr_loss = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)
        w = self.cfg.vgae
        task_loss = recon + w.canid_weight * canid + w.nbr_weight * nbr_loss + w.kl_weight * kl_loss
        return task_loss, cont_out, z

    def _step(self, batch):
        task_loss, cont_out, z = self._task_loss(batch)

        if self.teacher is not None:
            kd = self.cfg.kd
            self._teacher_on_cpu = _teacher_to_device(
                self.teacher, batch.x.device, self._teacher_on_cpu, self.cfg
            )

            with torch.no_grad():
                batch_idx = (
                    batch.batch
                    if batch.batch is not None
                    else torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
                )
                t_edge_attr = getattr(batch, "edge_attr", None)
                t_cont, _, _, t_z, _ = self.teacher(
                    batch.x, batch.edge_index, batch_idx, edge_attr=t_edge_attr
                )

            self._teacher_on_cpu = _teacher_offload(self.teacher, self.cfg)

            z_s = self.projection(z) if self.projection is not None else z
            min_n = min(z_s.size(0), t_z.size(0))
            latent_kd = F.mse_loss(z_s[:min_n], t_z[:min_n])

            min_r = min(cont_out.size(0), t_cont.size(0))
            recon_kd = F.mse_loss(cont_out[:min_r], t_cont[:min_r])

            kd_loss = kd.vgae_latent_weight * latent_kd + kd.vgae_recon_weight * recon_kd
            return kd.alpha * kd_loss + (1 - kd.alpha) * task_loss

        return task_loss

    def training_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.projection is not None:
            params += list(self.projection.parameters())
        opt = torch.optim.Adam(
            params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay
        )
        return build_optimizer_dict(opt, self.cfg)


class GATModule(pl.LightningModule):
    """GAT supervised classification (normal vs attack).

    When teacher is provided, adds soft-label KD:
      kd_loss = KL_div(student_logits/T, teacher_logits/T) * T^2
      total = alpha * kd_loss + (1-alpha) * task_loss

    Memory optimization: When cfg.training.offload_teacher_to_cpu is True, the teacher
    model is moved to CPU after each forward pass to free GPU memory.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        num_ids: int,
        in_channels: int,
        num_classes: int = 2,
        teacher: nn.Module | None = None,
    ):
        super().__init__()
        from graphids.core.models.gat import GATWithJK

        self.cfg = cfg
        self.model = GATWithJK.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.teacher = teacher
        self._teacher_on_cpu = False

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = F.cross_entropy(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()

        if self.teacher is not None:
            kd = self.cfg.kd
            self._teacher_on_cpu = _teacher_to_device(
                self.teacher, batch.x.device, self._teacher_on_cpu, self.cfg
            )

            with torch.no_grad():
                t_logits = self.teacher(batch)

            self._teacher_on_cpu = _teacher_offload(self.teacher, self.cfg)

            T = kd.temperature
            kd_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T**2)
            loss = kd.alpha * kd_loss + (1 - kd.alpha) * task_loss
        else:
            loss = task_loss

        return loss, acc

    def training_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_acc", acc, prog_bar=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        return build_optimizer_dict(opt, self.cfg)


class CurriculumDataModule(pl.LightningDataModule):
    """Resamples training data each epoch with increasing difficulty."""

    def __init__(self, normals, attacks, scores, val_data, cfg: PipelineConfig):
        super().__init__()
        self.normals = normals
        self.attacks = attacks
        self.scores = scores
        self.val_data = val_data
        self.cfg = cfg
        self._current_epoch = 0

    def train_dataloader(self):
        sampled = _curriculum_sample(
            self.normals,
            self.attacks,
            self.scores,
            self._current_epoch,
            self.cfg,
        )
        self._current_epoch += 1
        bs = effective_batch_size(self.cfg)
        max_nodes = None
        if self.cfg.training.dynamic_batching:
            max_nodes = compute_node_budget(bs, self.cfg)
        return make_dataloader(sampled, self.cfg, bs, shuffle=True, max_num_nodes=max_nodes)

    def val_dataloader(self):
        bs = effective_batch_size(self.cfg)
        max_nodes = None
        if self.cfg.training.dynamic_batching:
            max_nodes = compute_node_budget(bs, self.cfg)
        return make_dataloader(self.val_data, self.cfg, bs, shuffle=False, max_num_nodes=max_nodes)


def _curriculum_sample(normals, attacks, scores, epoch, cfg: PipelineConfig):
    """Sample training batch with curriculum ratio and difficulty-based selection."""
    progress = min(epoch / max(cfg.training.max_epochs, 1), 1.0)
    ratio = cfg.training.curriculum_start_ratio + progress * (
        cfg.training.curriculum_end_ratio - cfg.training.curriculum_start_ratio
    )
    percentile = cfg.training.difficulty_percentile + progress * (
        95 - cfg.training.difficulty_percentile
    )

    if scores:
        threshold = sorted(scores)[int(len(scores) * percentile / 100)]
        hard_normals = [n for n, s in zip(normals, scores) if s >= threshold]
        if not hard_normals:
            hard_normals = normals
    else:
        hard_normals = normals

    n_normals = min(int(len(attacks) * ratio), len(hard_normals))
    if n_normals and n_normals < len(hard_normals):
        # Use torch RNG for reproducible sampling controlled by pl.seed_everything()
        perm = torch.randperm(len(hard_normals))[:n_normals]
        sampled_normals = [hard_normals[i] for i in perm.tolist()]
    else:
        sampled_normals = hard_normals
    return sampled_normals + attacks
