"""Lightning modules for VGAE and GAT: train + val + test."""

from __future__ import annotations

import contextlib
import functools
import math
from typing import NamedTuple

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

import structlog

from graphids.config import cache_dir

from .trainer_factory import build_optimizer_dict

_log = structlog.get_logger()


class NodeBudgetInfo(NamedTuple):
    """Result of compute_node_budget: budget for DynamicBatchSampler + mean for num_steps."""
    budget: int
    mean_nodes: float


def compute_node_budget(batch_size: int, cfg) -> NodeBudgetInfo:
    """Derive max_num_nodes from batch_size * p95 graph node count.

    Reads graph statistics from cache_metadata.json written during preprocessing.
    Returns NodeBudgetInfo(budget, mean_nodes) so callers can pass mean_nodes
    to make_dataloader without a redundant file read.
    Raises FileNotFoundError if metadata is missing — rebuild caches first.
    """
    import json

    lake_root = cfg.lake_root if hasattr(cfg, "lake_root") else cfg["lake_root"]
    dataset = cfg.dataset if hasattr(cfg, "dataset") else cfg["dataset"]
    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        msg = (
            f"cache_metadata.json not found at {metadata_path}. "
            "Rebuild caches with: python -m graphids stage=preprocess dataset=..."
        )
        raise FileNotFoundError(msg)
    meta = json.loads(metadata_path.read_text())
    stats = meta["graph_stats"]["node_count"]
    p95 = stats["p95"]
    mean = stats["mean"]
    budget = int(batch_size * p95)
    _log.info("node_budget_computed", batch_size=batch_size, p95_nodes=p95,
             mean_nodes=mean, budget=budget)
    return NodeBudgetInfo(budget=budget, mean_nodes=mean)


class OOMSkipMixin:
    """Skip batch on CUDA OOM. Follows fairseq pattern (single-GPU safe).

    Lightning natively handles training_step returning None — it skips the
    optimizer step and continues training.
    """

    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning("oom_batch_skipped", batch_idx=batch_idx,
                         num_graphs=batch.num_graphs, num_nodes=batch.num_nodes)
            return None


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


def _get_kd_config(cfg):
    """Get KD auxiliary config, or None if not configured."""
    return next((a for a in cfg.get("auxiliaries", []) if a.type == "kd"), None)


@contextlib.contextmanager
def _teacher_on_device(module, device):
    """Move teacher to device for inference, offload back to CPU after."""
    if module.cfg.training.offload_teacher_to_cpu and module._teacher_on_cpu:
        module.teacher.to(device)
        module._teacher_on_cpu = False
    try:
        yield
    finally:
        if module.cfg.training.offload_teacher_to_cpu:
            module.teacher.to("cpu")
            module._teacher_on_cpu = True


class VGAEModule(OOMSkipMixin, pl.LightningModule):
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
            self.model = torch.compile(self.model, dynamic=True)
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
            edge_attr=edge_attr, mask_ratio=mask_ratio, node_id=batch.node_id,
        )

    def _task_loss(self, batch):
        cont_out, canid_logits, nbr_logits, z, kl_loss, mask = self(batch)
        target = batch.x
        if mask is not None:
            # Reconstruction loss on masked positions only (GraphMAE-style)
            recon = F.mse_loss(cont_out[mask], target[mask])
        else:
            recon = F.mse_loss(cont_out, target)
        canid = F.cross_entropy(canid_logits, batch.node_id)
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        nbr_loss = GraphAutoencoderNeighborhood.neighborhood_loss_negsampled(
            nbr_logits, batch.node_id, batch.edge_index,
            self.hparams["num_ids"], k_neg=self.cfg.vgae.k_neg,
        )
        w = self.cfg.vgae
        task_loss = recon + w.canid_weight * canid + w.nbr_weight * nbr_loss + w.kl_weight * kl_loss
        return task_loss, cont_out, z

    def _step(self, batch):
        task_loss, cont_out, z = self._task_loss(batch)

        if self.teacher is not None:
            kd = _get_kd_config(self.cfg)
            with _teacher_on_device(self, batch.x.device):
                with torch.no_grad():
                    batch_idx = (
                        batch.batch
                        if batch.batch is not None
                        else torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)
                    )
                    t_edge_attr = getattr(batch, "edge_attr", None)
                    t_cont, _, _, t_z, _, _ = self.teacher(
                        batch.x, batch.edge_index, batch_idx, edge_attr=t_edge_attr,
                        node_id=batch.node_id,
                    )

            z_s = self.projection(z) if self.projection is not None else z
            min_n = min(z_s.size(0), t_z.size(0))
            latent_kd = F.mse_loss(z_s[:min_n], t_z[:min_n])

            min_r = min(cont_out.size(0), t_cont.size(0))
            recon_kd = F.mse_loss(cont_out[:min_r], t_cont[:min_r])

            kd_loss = kd.vgae_latent_weight * latent_kd + kd.vgae_recon_weight * recon_kd
            return kd.alpha * kd_loss + (1 - kd.alpha) * task_loss

        return task_loss

    def _training_step_inner(self, batch, _idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, _idx):
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        cont, canid_logits, nbr_logits, _, _, _ = self.model(
            batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr,
            node_id=batch.node_id,
        )
        # Per-node continuous reconstruction error
        per_node_se = (cont - batch.x).pow(2).mean(dim=1)
        recon = scatter(per_node_se, batch.batch, dim=0, reduce="max")
        # Per-node CAN ID classification error
        canid_err = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
        canid_per_graph = scatter(canid_err, batch.batch, dim=0, reduce="max")
        # Per-node neighborhood prediction error
        nbr_targets = self.model.create_neighborhood_targets(
            batch.node_id, batch.edge_index, batch.batch,
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
            # Pass continuous error scores — BinaryAUROC needs probabilities,
            # and BinaryAccuracy/F1/Precision/Recall threshold at 0.5 internally.
            self.test_metrics.update(errors, batch.y)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_threshold is not None:
            self.log_dict(self.test_metrics.compute())

    def get_test_errors(self) -> tuple:
        """Return accumulated (errors, labels) as numpy arrays after test."""
        import numpy as np
        if not self._test_errors:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
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


class GATModule(OOMSkipMixin, pl.LightningModule):
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
            self.model = torch.compile(self.model, dynamic=True)
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
            self.loss_fn = functools.partial(_focal_loss, gamma=gamma)
        else:
            self.loss_fn = F.cross_entropy

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        logits = self(batch)
        task_loss = self.loss_fn(logits, batch.y)
        acc = (logits.argmax(1) == batch.y).float().mean()

        if self.teacher is not None:
            kd = _get_kd_config(self.cfg)
            with _teacher_on_device(self, batch.x.device):
                with torch.no_grad():
                    t_logits = self.teacher(batch)

            kd_loss = soft_label_kd_loss(logits, t_logits, kd.temperature)
            loss = kd.alpha * kd_loss + (1 - kd.alpha) * task_loss
        else:
            loss = task_loss

        return loss, acc

    def _training_step_inner(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_acc", acc, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, _idx):
        logits = self(batch)
        preds = logits.argmax(1)
        scores = F.softmax(logits, dim=1)[:, 1]
        self.test_metrics.update(scores, batch.y)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        return build_optimizer_dict(opt, self.cfg)


class CurriculumDynamicBatchSampler:
    """Curriculum selection + node-budget packing in a single batch sampler.

    Runs in the main process. Workers receive index batches via queues,
    so set_epoch() mutations propagate even with persistent_workers=True + spawn.
    """

    def __init__(
        self,
        dataset,
        normal_indices: list[int],
        attack_indices: list[int],
        scores: list[float],
        cfg,
        max_num_nodes: int | None,
    ):
        assert len(scores) == len(normal_indices), (
            f"scores ({len(scores)}) must align 1:1 with normal_indices ({len(normal_indices)})"
        )
        self.dataset = dataset
        self.normal_indices = normal_indices
        self.attack_indices = attack_indices
        self.scores = scores
        self.cfg = cfg
        self.max_num_nodes = max_num_nodes  # None = fixed-count mode
        self._active_indices = normal_indices + attack_indices
        # Cache node counts once — dataset is immutable
        self._node_counts = [dataset[i].num_nodes for i in range(len(dataset))]
        self._cached_len: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Update active indices for curriculum progression.

        Scores[i] corresponds to normal_indices[i]. This invariant is
        enforced by the assert in __init__.
        """
        cfg = self.cfg
        progress = min(epoch / max(cfg.training.max_epochs, 1), 1.0)
        ratio = math.lerp(cfg.training.curriculum_start_ratio, cfg.training.curriculum_end_ratio, progress)
        percentile = math.lerp(cfg.training.difficulty_percentile, 95.0, progress)

        if self.scores:
            scores_t = torch.tensor(self.scores)
            threshold = scores_t.quantile(percentile / 100).item()
            hard = [i for i, s in zip(self.normal_indices, self.scores) if s >= threshold]
            if not hard:
                hard = self.normal_indices
        else:
            hard = self.normal_indices

        n_normals = min(int(len(self.attack_indices) * ratio), len(hard))
        if n_normals and n_normals < len(hard):
            perm = torch.randperm(len(hard))[:n_normals]
            selected = [hard[i] for i in perm.tolist()]
        else:
            selected = hard
        self._active_indices = selected + self.attack_indices
        self._cached_len = None  # invalidate

    def __iter__(self):
        perm = torch.randperm(len(self._active_indices)).tolist()
        if self.max_num_nodes is None:
            # Fixed-count mode: batch_size graphs per batch
            bs = max(8, self.cfg.training.batch_size)
            for start in range(0, len(perm), bs):
                yield [self._active_indices[perm[j]] for j in range(start, min(start + bs, len(perm)))]
            return
        # Node-budget packing mode
        batch, batch_nodes = [], 0
        for i in perm:
            idx = self._active_indices[i]
            n = self._node_counts[idx]
            if n > self.max_num_nodes:
                continue  # skip_too_big
            if batch_nodes + n > self.max_num_nodes:
                yield batch
                batch, batch_nodes = [idx], n
            else:
                batch.append(idx)
                batch_nodes += n
        if batch:
            yield batch

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        if self.max_num_nodes is None:
            bs = max(8, self.cfg.training.batch_size)
            self._cached_len = max(1, (len(self._active_indices) + bs - 1) // bs)
        else:
            total = sum(self._node_counts[i] for i in self._active_indices)
            self._cached_len = max(1, total // self.max_num_nodes)
        return self._cached_len


class CurriculumDataModule(pl.LightningDataModule):
    """Curriculum learning with persistent workers.

    Builds ONE DataLoader at init. set_epoch() on the batch_sampler controls
    which graphs are yielded each epoch — no DataLoader rebuild needed.
    """

    def __init__(self, normals, attacks, scores, val_data, cfg):
        super().__init__()
        self.val_data = val_data
        self.cfg = cfg
        self._current_epoch = 0

        # Full dataset = normals + attacks; track indices
        full_dataset = normals + attacks
        normal_indices = list(range(len(normals)))
        attack_indices = list(range(len(normals), len(full_dataset)))

        from torch_geometric.loader import DataLoader as PyGDataLoader

        bs = max(8, cfg.training.batch_size)
        nw = cfg.num_workers
        common = dict(
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=True,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        if cfg.training.dynamic_batching:
            info = compute_node_budget(bs, cfg)
            self._mean_nodes = info.mean_nodes
            self._batch_sampler = CurriculumDynamicBatchSampler(
                full_dataset, normal_indices, attack_indices, scores, cfg, info.budget,
            )
            self._train_loader = PyGDataLoader(
                full_dataset, batch_sampler=self._batch_sampler, **common,
            )
        else:
            # Fixed-count batching: no node budget, just graph count
            self._batch_sampler = CurriculumDynamicBatchSampler(
                full_dataset, normal_indices, attack_indices, scores, cfg,
                max_num_nodes=None,
            )
            self._train_loader = PyGDataLoader(
                full_dataset, batch_sampler=self._batch_sampler, **common,
            )
            self._mean_nodes = None

    def train_dataloader(self):
        self._batch_sampler.set_epoch(self._current_epoch)
        self._current_epoch += 1
        return self._train_loader

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader as PyGDataLoader, DynamicBatchSampler

        bs = max(8, self.cfg.training.batch_size)
        nw = self.cfg.num_workers
        common = dict(
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        if self.cfg.training.dynamic_batching:
            info = compute_node_budget(bs, self.cfg)
            mean_nodes = self._mean_nodes
            num_steps = max(1, int(len(self.val_data) * mean_nodes / info.budget))
            sampler = DynamicBatchSampler(
                self.val_data, max_num=info.budget, mode="node", shuffle=False,
                num_steps=num_steps, skip_too_big=True,
            )
            return PyGDataLoader(self.val_data, batch_sampler=sampler, **common)

        return PyGDataLoader(self.val_data, batch_size=bs, shuffle=False, **common)


# ---------------------------------------------------------------------------
# DGI Module
# ---------------------------------------------------------------------------


class DGIModule(OOMSkipMixin, pl.LightningModule):
    """DGI contrastive training: maximize node–summary mutual information.

    Anomaly scoring at test time uses discriminator confidence:
    low discriminator agreement → anomalous graph.
    """

    def __init__(self, cfg):
        super().__init__()
        num_ids, in_channels = cfg.num_ids, cfg.in_channels
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg), "num_ids": num_ids, "in_channels": in_channels})
        from graphids.core.models.dgi import GraphInfomaxModel

        self.cfg = cfg
        self.model = GraphInfomaxModel.from_config(cfg, num_ids, in_channels)
        if cfg.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, dynamic=True)
        # Test mode: discriminator-based anomaly scoring
        self.test_threshold: float | None = None
        self.test_metrics = MetricCollection({
            "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
            "precision": BinaryPrecision(), "recall": BinaryRecall(),
            "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
        })
        self._test_scores: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        return self.model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=edge_attr, node_id=batch.node_id,
        )

    def _training_step_inner(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._oom_safe_step(batch, batch_idx, self._training_step_inner)

    def validation_step(self, batch, _idx):
        pos_z, neg_z, summary = self(batch)
        loss = self.model.dgi_loss(pos_z, neg_z, summary, batch.batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, _idx):
        from torch_geometric.utils import scatter

        pos_z = self.model.encode(
            batch.x, batch.edge_index,
            getattr(batch, "edge_attr", None),
            batch.batch, batch.node_id,
        )
        summary = self.model.summarize(pos_z, batch.batch)
        node_scores = self.model.discriminate(pos_z, summary, batch.batch)
        # Low mean discriminator score → anomalous graph
        graph_scores = 1 - scatter(node_scores, batch.batch, dim=0, reduce="mean")
        self._test_scores.append(graph_scores)
        self._test_labels.append(batch.y)
        if self.test_threshold is not None:
            self.test_metrics.update(graph_scores, batch.y)

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_threshold is not None:
            self.log_dict(self.test_metrics.compute())

    def get_test_errors(self) -> tuple:
        """Return accumulated (anomaly_scores, labels) as numpy arrays."""
        import numpy as np
        if not self._test_scores:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        return (torch.cat(self._test_scores).cpu().numpy(),
                torch.cat(self._test_labels).cpu().numpy())

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        return build_optimizer_dict(opt, self.cfg)
