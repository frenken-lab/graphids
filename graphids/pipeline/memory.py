"""Context-aware memory management for GPU training.

Batch size computation through layered estimation:
1. Static (fast): Model parameters, embeddings, optimizer states, CUDA overhead
2. Measured (default): Forward hooks to measure actual activation memory

Includes batch size caching: ``save_budget_cache`` / ``load_budget_cache``
persist the ``MemoryBudget`` to ``memory_cache.json`` in the run directory,
keyed by a config fingerprint for invalidation on config changes.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch_geometric.data import Data

from graphids.config.constants import CUDA_CONTEXT_MB, FRAGMENTATION_BUFFER, get_batch_index

log = logging.getLogger(__name__)


@dataclass
class MemoryBudget:
    """Memory budget breakdown for training."""

    total_gpu_mb: float
    cuda_context_mb: float = CUDA_CONTEXT_MB
    model_params_mb: float = 0.0
    embedding_mb: float = 0.0
    optimizer_mb: float = 0.0
    gradient_mb: float = 0.0
    activation_mb: float = 0.0
    teacher_mb: float = 0.0
    per_graph_mb: float = 0.0
    available_for_data_mb: float = 0.0
    recommended_batch_size: int = 8
    target_utilization: float = 0.7
    estimation_mode: str = "static"
    warnings: list[str] = field(default_factory=list)

    @property
    def static_memory_mb(self) -> float:
        """Total static memory (always occupied during training)."""
        return (
            self.cuda_context_mb
            + self.model_params_mb
            + self.embedding_mb
            + self.optimizer_mb
            + self.gradient_mb
            + self.teacher_mb
        )

    def __str__(self) -> str:
        return (
            f"MemoryBudget(total={self.total_gpu_mb:.0f}MB "
            f"static={self.static_memory_mb:.1f}MB "
            f"[params={self.model_params_mb:.1f} embed={self.embedding_mb:.1f} "
            f"opt={self.optimizer_mb:.1f} grad={self.gradient_mb:.1f} "
            f"teacher={self.teacher_mb:.1f}] "
            f"activation={self.activation_mb:.1f}MB "
            f"per_graph={self.per_graph_mb:.3f}MB "
            f"available={self.available_for_data_mb:.1f}MB "
            f"batch_size={self.recommended_batch_size} "
            f"mode={self.estimation_mode})"
        )


# ---------------------------------------------------------------------------
# Static Estimation
# ---------------------------------------------------------------------------


def _get_gpu_memory_mb(device: torch.device | None = None) -> float:
    """Get total GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024**2)


def _count_embedding_memory_mb(model: nn.Module) -> float:
    """Count memory from nn.Embedding layers."""
    total_bytes = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            total_bytes += module.weight.numel() * module.weight.element_size()
    return total_bytes / (1024**2)


def _estimate_model_memory_mb(model: nn.Module) -> tuple[float, float, float, float]:
    """Estimate model memory: (params_mb, embedding_mb, optimizer_mb, gradient_mb)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024**2)
    embedding_mb = _count_embedding_memory_mb(model)
    params_without_embed_mb = param_mb - embedding_mb
    optimizer_mb = param_mb * 2  # Adam stores 2 states per param
    gradient_mb = param_mb
    return params_without_embed_mb, embedding_mb, optimizer_mb, gradient_mb


def _estimate_graph_memory_mb(sample_graph: Data) -> float:
    """Estimate memory per graph based on a sample."""
    total_bytes = 0
    for attr in ["x", "edge_index", "edge_attr", "y", "batch"]:
        tensor = getattr(sample_graph, attr, None)
        if tensor is not None:
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def _estimate_activation_heuristic(model: nn.Module, sample_graph: Data, precision: str) -> float:
    """Heuristic activation memory for GNNs."""
    num_nodes = sample_graph.x.size(0) if sample_graph.x is not None else 100
    num_edges = sample_graph.edge_index.size(1) if sample_graph.edge_index is not None else 500

    num_layers = 0
    max_hidden = 64
    num_heads = 1

    for module in model.modules():
        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            num_layers += 1
            max_hidden = max(max_hidden, module.out_channels)
            if hasattr(module, "heads"):
                num_heads = max(num_heads, module.heads)

    num_layers = max(1, num_layers)
    bytes_per_elem = 2 if "16" in precision else 4

    forward_bytes = num_nodes * max_hidden * num_layers * bytes_per_elem
    message_bytes = num_edges * max_hidden * num_layers * bytes_per_elem
    attention_bytes = num_edges * num_heads * num_layers * bytes_per_elem

    # 2x for backward pass storage
    return (forward_bytes + message_bytes + attention_bytes) * 2 / (1024**2)


# ---------------------------------------------------------------------------
# Measured Estimation (Forward Hooks)
# ---------------------------------------------------------------------------


def _measure_activation_memory_mb(
    model: nn.Module, sample_graph: Data, device: torch.device
) -> float:
    """Measure activation memory using forward hooks."""
    activation_bytes: list[int] = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_bytes.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    activation_bytes.append(t.numel() * t.element_size())

    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    was_training = model.training
    try:
        model.eval()
        sample = sample_graph.clone().to(device)

        with torch.no_grad():
            batch_idx = get_batch_index(sample, device)

            if hasattr(model, "encode"):
                model(sample.x, sample.edge_index, batch_idx)
            else:
                model(sample)

        del sample
        torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    # 2.5x for backward pass storage
    return sum(activation_bytes) * 2.5 / (1024**2)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def _try_forward_backward(
    model: nn.Module,
    graphs: list[Data],
    device: torch.device,
    precision: str = "16-mixed",
) -> bool:
    """Attempt a forward+backward pass on a batch of graphs. Returns True on success."""
    from torch_geometric.data import Batch

    try:
        batch = Batch.from_data_list([g.clone() for g in graphs]).to(device)
        model.train()

        use_amp = "16" in precision
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=use_amp)
            if device.type == "cuda"
            else torch.amp.autocast("cpu", enabled=False)
        )

        with autocast_ctx:
            if hasattr(model, "encode"):
                # VGAE path
                edge_attr = getattr(batch, "edge_attr", None)
                cont, canid_logits, _, _, kl = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    edge_attr=edge_attr,
                )
                loss = F.mse_loss(cont, batch.x[:, 1:]) + kl * 0.001
            else:
                # GAT path
                logits = model(batch)
                loss = F.cross_entropy(logits, batch.y.long())

        loss.backward()
        model.zero_grad(set_to_none=True)
        del batch, loss
        return True

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        return False

    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def _trial_batch_size(
    model: nn.Module,
    sample_graphs: list[Data],
    device: torch.device,
    min_bs: int = 8,
    max_bs: int = 512,
    precision: str = "16-mixed",
) -> int:
    """Find maximum batch size via binary search with actual forward+backward trials.

    Returns a safe batch size (90% of the largest successful value).
    """
    if len(sample_graphs) < min_bs:
        log.warning(
            "Trial batch: only %d graphs available, using min_bs=%d", len(sample_graphs), min_bs
        )
        return min_bs

    # Clamp max_bs to available graphs
    max_bs = min(max_bs, len(sample_graphs))

    low, high = min_bs, max_bs
    known_good = min_bs

    log.info("Trial batch size search: range [%d, %d]", low, high)

    while high - low > max(1, low // 8):
        mid = (low + high) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        success = _try_forward_backward(model, sample_graphs[:mid], device, precision)

        if success:
            known_good = mid
            low = mid + 1
            log.info("Trial batch: %d OK", mid)
        else:
            high = mid - 1
            log.info("Trial batch: %d OOM", mid)

    safe_bs = max(min_bs, int(known_good * 0.9))
    log.info("Trial batch result: known_good=%d, safe=%d", known_good, safe_bs)
    return safe_bs


def compute_batch_size(
    model: nn.Module,
    sample_graph: Data,
    device: torch.device,
    teacher: nn.Module | None = None,
    precision: str = "32",
    target_utilization: float = 0.7,
    min_batch_size: int = 8,
    max_batch_size: int = 8192,
    mode: str = "static",
) -> MemoryBudget:
    """Compute optimal batch size.

    Args:
        mode: "static" (fast heuristic) or "measured" (forward hooks, more accurate)
    """
    warnings = []
    total_gpu_mb = _get_gpu_memory_mb(device)

    if total_gpu_mb == 0:
        log.warning("No GPU detected, using fallback batch size")
        return MemoryBudget(
            total_gpu_mb=0,
            recommended_batch_size=min_batch_size,
            estimation_mode=mode,
            warnings=["No GPU detected"],
        )

    # Static estimation
    params_mb, embedding_mb, optimizer_mb, gradient_mb = _estimate_model_memory_mb(model)

    teacher_mb = 0.0
    if teacher is not None:
        t_params, t_embed, _, _ = _estimate_model_memory_mb(teacher)
        # Teacher doesn't need gradients/optimizer
        teacher_mb = t_params + t_embed

    per_graph_mb = _estimate_graph_memory_mb(sample_graph)

    # Activation memory
    if mode == "measured":
        try:
            activation_mb = _measure_activation_memory_mb(model, sample_graph, device)
            log.info("Measured activation memory: %.2f MB", activation_mb)
        except Exception as e:
            log.warning("Activation measurement failed: %s, using heuristic", e)
            activation_mb = _estimate_activation_heuristic(model, sample_graph, precision)
            warnings.append(f"Activation measurement failed: {e}")
    else:
        activation_mb = _estimate_activation_heuristic(model, sample_graph, precision)

    # Compute available memory
    static_mb = CUDA_CONTEXT_MB + params_mb + embedding_mb + optimizer_mb + gradient_mb + teacher_mb

    effective_utilization = target_utilization * (1 - FRAGMENTATION_BUFFER)
    available_for_data_mb = (total_gpu_mb * effective_utilization) - static_mb

    if available_for_data_mb <= 0:
        log.warning("Static memory (%.1fMB) exceeds target. Using minimum batch size.", static_mb)
        warnings.append("Static memory exceeds target")
        available_for_data_mb = 0

    effective_available = available_for_data_mb - activation_mb

    if per_graph_mb > 0 and effective_available > 0:
        raw_batch_size = int(effective_available / per_graph_mb)
    else:
        raw_batch_size = min_batch_size

    recommended_batch_size = max(min_batch_size, min(raw_batch_size, max_batch_size))

    budget = MemoryBudget(
        total_gpu_mb=total_gpu_mb,
        cuda_context_mb=CUDA_CONTEXT_MB,
        model_params_mb=params_mb,
        embedding_mb=embedding_mb,
        optimizer_mb=optimizer_mb,
        gradient_mb=gradient_mb,
        activation_mb=activation_mb,
        teacher_mb=teacher_mb,
        per_graph_mb=per_graph_mb,
        available_for_data_mb=available_for_data_mb,
        recommended_batch_size=recommended_batch_size,
        target_utilization=target_utilization,
        estimation_mode=mode,
        warnings=warnings,
    )

    log.info("Memory budget: %s", budget)
    return budget


def _config_hash(cfg) -> str:
    """Compute a short fingerprint of a PipelineConfig for cache invalidation."""
    return str(hash(str(cfg.model_dump())))


def save_budget_cache(budget: MemoryBudget, run_dir: Path, cfg) -> None:
    """Persist a MemoryBudget to ``memory_cache.json`` for future runs."""
    cache_path = run_dir / "memory_cache.json"
    data = {
        "config_hash": _config_hash(cfg),
        "recommended_batch_size": budget.recommended_batch_size,
        "total_gpu_mb": budget.total_gpu_mb,
        "model_params_mb": budget.model_params_mb,
        "embedding_mb": budget.embedding_mb,
        "optimizer_mb": budget.optimizer_mb,
        "gradient_mb": budget.gradient_mb,
        "activation_mb": budget.activation_mb,
        "teacher_mb": budget.teacher_mb,
        "per_graph_mb": budget.per_graph_mb,
        "available_for_data_mb": budget.available_for_data_mb,
        "estimation_mode": budget.estimation_mode,
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(data, indent=2))
        log.info("Saved memory budget cache: %s", cache_path)
    except Exception as e:
        log.warning("Failed to save memory cache: %s", e)


def load_budget_cache(run_dir: Path, cfg) -> MemoryBudget | None:
    """Load a cached MemoryBudget if the config hash matches.

    Returns None on cache miss or hash mismatch.
    """
    cache_path = run_dir / "memory_cache.json"
    if not cache_path.exists():
        return None
    try:
        data = _json.loads(cache_path.read_text())
        if data.get("config_hash") != _config_hash(cfg):
            log.info("Memory cache stale (config changed), re-computing")
            return None
        budget = MemoryBudget(
            total_gpu_mb=data["total_gpu_mb"],
            model_params_mb=data["model_params_mb"],
            embedding_mb=data["embedding_mb"],
            optimizer_mb=data["optimizer_mb"],
            gradient_mb=data["gradient_mb"],
            activation_mb=data["activation_mb"],
            teacher_mb=data["teacher_mb"],
            per_graph_mb=data["per_graph_mb"],
            available_for_data_mb=data["available_for_data_mb"],
            recommended_batch_size=data["recommended_batch_size"],
            estimation_mode=data["estimation_mode"],
        )
        log.info("Loaded cached memory budget: batch_size=%d", budget.recommended_batch_size)
        return budget
    except Exception as e:
        log.warning("Failed to load memory cache: %s", e)
        return None


def log_memory_state(prefix: str = "") -> None:
    """Log current GPU memory state for debugging."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)

    log.info(
        "%sGPU memory: allocated=%.1fMB, reserved=%.1fMB",
        f"[{prefix}] " if prefix else "",
        allocated,
        reserved,
    )
