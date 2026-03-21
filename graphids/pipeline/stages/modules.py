"""Lightning modules for VGAE and GAT: train + val + test."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from .data_loading import compute_node_budget, make_dataloader
from .trainer_factory import build_optimizer_dict



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


def soft_label_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Hinton soft-label knowledge distillation loss (F.kl_div).

    KL(softmax(student/T) || softmax(teacher/T)) * T^2
    """
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def _focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for class-imbalanced classification."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


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
        cfg,
        teacher: nn.Module | None = None,
        projection: nn.Linear | None = None,
    ):
        super().__init__()
        num_ids, in_channels = cfg.num_ids, cfg.in_channels
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg), "num_ids": num_ids, "in_channels": in_channels})
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        self.cfg = cfg
        self.model = GraphAutoencoderNeighborhood.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.teacher = teacher
        self.projection = projection
        self._teacher_on_cpu = False
        # Test mode
        self.test_threshold: float | None = None
        self.test_metrics = MetricCollection({
            "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
            "precision": BinaryPrecision(), "recall": BinaryRecall(),
            "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
        })
        self._test_errors: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        mask_ratio = self.cfg.vgae.mask_ratio if self.training else 0.0
        return self.model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=edge_attr, mask_ratio=mask_ratio,
        )

    def _task_loss(self, batch):
        cont_out, canid_logits, nbr_logits, z, kl_loss, mask = self(batch)
        target = batch.x[:, 1:]
        if mask is not None:
            # Reconstruction loss on masked positions only (GraphMAE-style)
            recon = F.mse_loss(cont_out[mask], target[mask])
        else:
            recon = F.mse_loss(cont_out, target)
        canid = F.cross_entropy(canid_logits, batch.x[:, 0].long())
        nbr_targets = self.model.create_neighborhood_targets(batch.x, batch.edge_index, batch.batch)
        nbr_loss = F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets)
        w = self.cfg.vgae
        task_loss = recon + w.canid_weight * canid + w.nbr_weight * nbr_loss + w.kl_weight * kl_loss
        return task_loss, cont_out, z

    def _step(self, batch):
        task_loss, cont_out, z = self._task_loss(batch)

        if self.teacher is not None:
            kd = next((a for a in self.cfg.get("auxiliaries", []) if a.type == "kd"), None)
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

    def test_step(self, batch, _idx):
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        cont, canid_logits, nbr_logits, _, _, _ = self.model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr,
        )
        # Per-node continuous reconstruction error
        per_node_se = (cont - batch.x[:, 1:]).pow(2).mean(dim=1)
        recon = scatter(per_node_se, batch.batch, dim=0, reduce="max")
        # Per-node CAN ID classification error
        canid_err = F.cross_entropy(canid_logits, batch.x[:, 0].long(), reduction="none")
        canid_per_graph = scatter(canid_err, batch.batch, dim=0, reduce="max")
        # Per-node neighborhood prediction error
        nbr_targets = self.model.create_neighborhood_targets(
            batch.x, batch.edge_index, batch.batch,
        )
        nbr_err = F.binary_cross_entropy_with_logits(
            nbr_logits, nbr_targets, reduction="none",
        ).mean(dim=1)
        nbr_per_graph = scatter(nbr_err, batch.batch, dim=0, reduce="max")
        # Composite score using config weights
        w = self.cfg.vgae
        errors = recon + w.canid_weight * canid_per_graph + w.nbr_weight * nbr_per_graph
        self._test_errors.append(errors)
        self._test_labels.append(batch.y)
        if self.test_threshold is not None:
            preds = (errors > self.test_threshold).long()
            self.test_metrics.update(preds, batch.y)
            self.log_dict(self.test_metrics, batch_size=batch.num_graphs)

    def get_test_errors(self) -> tuple:
        """Return accumulated (errors, labels) as numpy arrays after test."""
        import numpy as np
        return (torch.cat(self._test_errors).cpu().numpy(),
                torch.cat(self._test_labels).cpu().numpy())

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
        cfg,
        num_classes: int = 2,
        teacher: nn.Module | None = None,
    ):
        super().__init__()
        num_ids, in_channels = cfg.num_ids, cfg.in_channels
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg), "num_ids": num_ids, "in_channels": in_channels})
        from graphids.core.models.gat import GATWithJK

        self.cfg = cfg
        self.model = GATWithJK.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.teacher = teacher
        self._teacher_on_cpu = False
        self.test_metrics = MetricCollection({
            "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
            "precision": BinaryPrecision(), "recall": BinaryRecall(),
            "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
        })
        # Configurable loss for class-imbalance experiments
        loss_name = cfg.training.loss_fn
        if loss_name == "weighted_ce":
            w = torch.tensor([1.0, cfg.training.loss_weight])
            self.loss_fn = nn.CrossEntropyLoss(weight=w)
        elif loss_name == "focal":
            gamma = cfg.training.focal_gamma
            self.loss_fn = lambda logits, y: _focal_loss(logits, y, gamma)
        else:
            self.loss_fn = F.cross_entropy

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = self.loss_fn(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()

        if self.teacher is not None:
            kd = next((a for a in self.cfg.get("auxiliaries", []) if a.type == "kd"), None)
            self._teacher_on_cpu = _teacher_to_device(
                self.teacher, batch.x.device, self._teacher_on_cpu, self.cfg
            )

            with torch.no_grad():
                t_logits = self.teacher(batch)

            self._teacher_on_cpu = _teacher_offload(self.teacher, self.cfg)

            kd_loss = soft_label_kd_loss(logits, t_logits, kd.temperature)
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

    def test_step(self, batch, _idx):
        logits = self(batch)
        preds = logits.argmax(1)
        scores = F.softmax(logits, dim=1)[:, 1]
        self.test_metrics.update(preds, batch.y)
        self.log_dict(self.test_metrics, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        return build_optimizer_dict(opt, self.cfg)


class CurriculumDataModule(pl.LightningDataModule):
    """Resamples training data each epoch with increasing difficulty."""

    def __init__(self, normals, attacks, scores, val_data, cfg):
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
        bs = max(8, int(self.cfg.training.batch_size * self.cfg.training.safety_factor))
        max_nodes = None
        if self.cfg.training.dynamic_batching:
            max_nodes = compute_node_budget(bs, self.cfg)
        return make_dataloader(sampled, self.cfg, bs, shuffle=True, max_num_nodes=max_nodes)

    def val_dataloader(self):
        bs = max(8, int(self.cfg.training.batch_size * self.cfg.training.safety_factor))
        max_nodes = None
        if self.cfg.training.dynamic_batching:
            max_nodes = compute_node_budget(bs, self.cfg)
        return make_dataloader(self.val_data, self.cfg, bs, shuffle=False, max_num_nodes=max_nodes)


def _curriculum_sample(normals, attacks, scores, epoch, cfg):
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
