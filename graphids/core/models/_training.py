"""Lightning training helpers shared across VGAE, GAT, and DGI modules."""

import contextlib
from typing import NamedTuple

import structlog
import torch
import torch.nn.functional as F
from torch import Tensor

_log = structlog.get_logger()


class NodeBudgetInfo(NamedTuple):
    """Result of compute_node_budget: budget for DynamicBatchSampler + mean for num_steps."""
    budget: int
    mean_nodes: float


def compute_node_budget(batch_size: int, cfg) -> NodeBudgetInfo:
    """Derive max_num_nodes from batch_size * p95 graph node count."""
    import json
    from graphids.config import cache_dir

    lake_root = cfg.lake_root if hasattr(cfg, "lake_root") else cfg["lake_root"]
    dataset = cfg.dataset if hasattr(cfg, "dataset") else cfg["dataset"]
    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cache_metadata.json not found at {metadata_path}. "
            "Rebuild caches with: python -m graphids stage=preprocess dataset=..."
        )
    meta = json.loads(metadata_path.read_text())
    stats = meta["graph_stats"]["node_count"]
    budget = int(batch_size * stats["p95"])
    _log.info("node_budget_computed", batch_size=batch_size, p95_nodes=stats["p95"],
             mean_nodes=stats["mean"], budget=budget)
    return NodeBudgetInfo(budget=budget, mean_nodes=stats["mean"])


class OOMSkipMixin:
    """Skip batch on CUDA OOM. Lightning natively handles training_step returning None."""

    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning("oom_batch_skipped", batch_idx=batch_idx,
                         num_graphs=batch.num_graphs, num_nodes=batch.num_nodes)
            return None


def soft_label_kd_loss(student_logits, teacher_logits, temperature: float):
    """Hinton soft-label KD loss: KL(student/T || teacher/T) * T^2."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def focal_loss(logits, targets, gamma: float = 2.0):
    """Focal loss (Lin et al. 2017) for class-imbalanced classification."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _get_kd_config(cfg):
    """Get KD auxiliary config, or None if not configured."""
    return next((a for a in cfg.get("auxiliaries", []) if a.type == "kd"), None)


@contextlib.contextmanager
def teacher_on_device(module, device):
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


def build_optimizer_dict(optimizer, cfg):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    if not cfg.training.use_scheduler or cfg.training.scheduler is None:
        return optimizer
    from hydra.utils import instantiate
    sched = instantiate(cfg.training.scheduler, optimizer=optimizer)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "monitor": cfg.training.monitor_metric}}
