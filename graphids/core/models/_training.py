"""Lightning training helpers shared across VGAE, GAT, and DGI modules."""

import contextlib
import math
from typing import NamedTuple

import structlog
import torch
import torch.nn.functional as F
from torch import Tensor

_log = structlog.get_logger()

# Conv types with O(N²) global attention (full attention matrix across all batch nodes).
_QUADRATIC_CONV_TYPES = frozenset({"gps"})

# Fraction of total VRAM reserved for the attention matrix (rest for model + activations + framework).
_ATTN_VRAM_FRACTION = 0.6


class NodeBudgetInfo(NamedTuple):
    """Result of compute_node_budget: budget for DynamicBatchSampler + mean for num_steps."""
    budget: int
    mean_nodes: float


def _available_vram_bytes() -> int:
    """Total GPU VRAM in bytes. Falls back to 12 GB for CPU/testing."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 12 * 1024**3


def compute_node_budget(
    batch_size: int, cfg, *, conv_type: str = "gatv2", heads: int = 4,
) -> NodeBudgetInfo:
    """Derive max_num_nodes for DynamicBatchSampler from graph stats and conv complexity.

    For linear convs (gatv2, gat, transformer): budget = batch_size * p95_nodes.
    For quadratic convs (gps): budget = min(linear_budget, VRAM-safe node ceiling).

    The quadratic cap prevents GPS's O(N²) global attention from allocating an
    attention matrix larger than available VRAM.  The ceiling is derived from
    ``sqrt(vram_bytes * attn_fraction / (heads * 3 * dtype_bytes))``.
    """
    import json
    from graphids.config import cache_dir

    lake_root = cfg.lake_root
    dataset = cfg.dataset
    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cache_metadata.json not found at {metadata_path}. "
            "Rebuild caches with: python -m graphids stage=preprocess dataset=..."
        )
    meta = json.loads(metadata_path.read_text())
    stats = meta["graph_stats"]["node_count"]
    linear_budget = int(batch_size * stats["p95"])

    if conv_type in _QUADRATIC_CONV_TYPES:
        vram = _available_vram_bytes()
        # Attention matrix: N² * num_heads * 3 (Q, K, V) * 2 bytes (fp16)
        cost_per_n2 = heads * 3 * 2
        quadratic_cap = int(math.sqrt(vram * _ATTN_VRAM_FRACTION / cost_per_n2))
        budget = min(linear_budget, quadratic_cap)
        _log.info("node_budget_computed", conv_type=conv_type, batch_size=batch_size,
                  p95_nodes=stats["p95"], linear_budget=linear_budget,
                  quadratic_cap=quadratic_cap, vram_gb=round(vram / 1e9, 1),
                  budget=budget, mean_nodes=stats["mean"])
    else:
        budget = linear_budget
        _log.info("node_budget_computed", conv_type=conv_type, batch_size=batch_size,
                  p95_nodes=stats["p95"], budget=budget, mean_nodes=stats["mean"])

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


def binary_test_metrics():
    """Standard binary classification MetricCollection shared by all Lightning modules."""
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        BinaryAccuracy, BinaryAUROC, BinaryF1Score,
        BinaryPrecision, BinaryRecall, BinarySpecificity,
    )
    return MetricCollection({
        "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
        "precision": BinaryPrecision(), "recall": BinaryRecall(),
        "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
    })


# ---------------------------------------------------------------------------
# Evaluation helpers (shared by module evaluate() classmethods)
# ---------------------------------------------------------------------------


def _eval_trainer():
    """Single reusable Trainer for test-time evaluation."""
    import pytorch_lightning as pl
    return pl.Trainer(
        accelerator="auto", devices="auto",
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )


def find_threshold(
    module, data: list, *, score_key: str = "scores", batch_size: int = 256,
) -> tuple[float, float]:
    """Find optimal anomaly threshold via Youden's J on validation data.

    Works with any LightningModule whose predict_step returns a dict with
    ``score_key`` (continuous anomaly scores) and ``"labels"`` (binary ground truth).

    Args:
        module: LightningModule with a compatible predict_step.
        data: List of PyG Data objects (validation set).
        score_key: Key in predict_step output containing anomaly scores.
        batch_size: Batch size for the prediction DataLoader.

    Returns:
        (threshold, youden_j) tuple.
    """
    from torchmetrics.functional.classification import binary_roc

    from graphids.core.preprocessing.datamodule import make_graph_loader

    trainer = _eval_trainer()
    loader = make_graph_loader(data, batch_size=batch_size)
    preds = trainer.predict(module, dataloaders=loader)

    scores = torch.cat([p[score_key] for p in preds]).cpu()
    labels = torch.cat([p["labels"] for p in preds]).cpu()

    if len(scores) == 0:
        return 0.5, 0.0
    if labels.unique().numel() < 2:
        return float(scores.median()), 0.0

    fpr_v, tpr_v, thresholds_v = binary_roc(scores, labels.long())
    j_scores = tpr_v - fpr_v

    if len(j_scores) == 0 or len(thresholds_v) == 0:
        return float(scores.median()), 0.0

    best_idx = torch.argmax(j_scores).item()
    thresh = float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(scores.median())
    return thresh, float(j_scores[best_idx])


def test_model(module, data, batch_size: int = 256, *, trainer=None) -> dict:
    """Run trainer.test() on a module and return metrics.

    Args:
        data: Either a list of PyG Data objects (creates PyGDataLoader) or
              a pre-built DataLoader (used as-is, e.g. for fusion tensor batches).
        trainer: Reuse an existing Trainer. Created if not provided.
    """
    from graphids.core.preprocessing.datamodule import make_graph_loader

    if trainer is None:
        trainer = _eval_trainer()
    loader = make_graph_loader(data, batch_size=batch_size) if isinstance(data, list) else data
    results = trainer.test(module, dataloaders=loader, verbose=False)
    metrics = dict(results[0]) if results else {}
    metrics["balanced_accuracy"] = (metrics.get("recall", 0) + metrics.get("specificity", 0)) / 2
    return metrics


def eval_with_scenarios(module, val_data, test_scenarios, batch_size: int) -> tuple[dict, dict]:
    """Run test on val + each test scenario. Returns (val_metrics, scenario_metrics)."""
    trainer = _eval_trainer()
    val_metrics = test_model(module, val_data, batch_size=batch_size, trainer=trainer)
    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tdata, batch_size=batch_size, trainer=trainer)
    return val_metrics, scenario_metrics


def gpu_cleanup(*objs):
    """Delete objects and free GPU memory."""
    import gc as _gc
    for o in objs:
        del o
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model loading + KD preparation (used by pipeline and module __init__)
# ---------------------------------------------------------------------------


def load_inner_model(
    model_type: str, ckpt_path, device,
) -> tuple[torch.nn.Module, object]:
    """Load a Lightning checkpoint, return (inner nn.Module on device in eval, hparams cfg).

    Uses Lightning's load_from_checkpoint under the hood. Extracts the raw
    nn.Module (not the LightningModule wrapper) for use as teacher / inference.
    """
    from pathlib import Path

    from graphids.core.models.registry import get_module_cls

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    module = get_module_cls(model_type).load_from_checkpoint(
        str(ckpt_path), map_location="cpu", weights_only=True,
    )
    model = module.model
    model.to(device).eval()
    return model, module.hparams


def prepare_kd(
    cfg, model_type: str, device,
) -> tuple[torch.nn.Module | None, torch.nn.Linear | None]:
    """Resolve teacher checkpoint, load + freeze, create projection if needed.

    Returns (teacher, projection) when KD is active, (None, None) otherwise.
    Called by module __init__ or pipeline build_module.
    """
    from pathlib import Path

    if not any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        return None, None

    kd = next(a for a in cfg.get("auxiliaries", []) if a.type == "kd")
    if kd.get("model_path"):
        teacher_path = Path(kd.model_path)
    else:
        from graphids.config import checkpoint_path
        teacher_scale = kd.get("teacher_scale", "large")
        teacher_path = checkpoint_path(
            cfg.lake_root, cfg.dataset, model_type, teacher_scale, cfg.seed, cfg,
            gat_stage=getattr(cfg, "gat_stage", "curriculum"),
        )
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_path}. "
                f"Train {model_type}/{teacher_scale} first, or set model_path explicitly."
            )

    teacher, tcfg = load_inner_model(model_type, teacher_path, device)
    teacher.requires_grad_(False)

    # Projection layer for VGAE latent space alignment
    projection = None
    if model_type == "vgae":
        s_dim = cfg.vgae.latent_dim
        t_dim = tcfg.vgae.latent_dim
        if s_dim != t_dim:
            _log.info("projection_layer", student_dim=s_dim, teacher_dim=t_dim)
            projection = torch.nn.Linear(s_dim, t_dim).to(device)

    return teacher, projection
