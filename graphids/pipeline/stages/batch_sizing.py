"""Batch size computation and memory-aware sizing."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from graphids.config import PipelineConfig, cache_dir

from ..memory import (
    MemoryBudget,
    _get_gpu_memory_mb,
    compute_batch_size,
    load_budget_cache,
    save_budget_cache,
)

log = logging.getLogger(__name__)


def effective_batch_size(cfg: PipelineConfig) -> int:
    """Apply safety factor to batch size (legacy fallback)."""
    return max(8, int(cfg.training.batch_size * cfg.training.safety_factor))


def resolve_batch_config(cfg, model, train_data, teacher=None):
    """Compute batch size and optional dynamic batching node budget."""
    from .data_loading import compute_node_budget

    if cfg.training.optimize_batch_size:
        bs = compute_optimal_batch_size(model, train_data, cfg, teacher=teacher)
    else:
        bs = effective_batch_size(cfg)
    max_nodes = None
    if cfg.training.dynamic_batching:
        max_nodes = compute_node_budget(bs, cfg)
        if max_nodes:
            log.info("Dynamic batching: max_num_nodes=%d (batch_size=%d × p95)", max_nodes, bs)
        else:
            log.info(
                "Dynamic batching: no cache metadata, falling back to static batch_size=%d", bs
            )
    return bs, max_nodes


def _get_representative_graph(train_data, cfg: PipelineConfig):
    """Get the p95 graph by node count for conservative batch sizing.

    Falls back to ``train_data[0]`` when cache metadata is unavailable.
    """
    import json as _json

    metadata_path = cache_dir(cfg) / "cache_metadata.json"
    if metadata_path.exists():
        try:
            meta = _json.loads(metadata_path.read_text())
            p95_nodes = meta.get("graph_stats", {}).get("node_count", {}).get("p95")
            if p95_nodes:
                candidates = [train_data[i] for i in range(min(1000, len(train_data)))]
                best = min(candidates, key=lambda g: abs(g.x.size(0) - p95_nodes))
                log.info(
                    "Representative graph: p95=%d nodes, selected=%d nodes",
                    p95_nodes,
                    best.x.size(0),
                )
                return best
        except Exception as e:
            log.warning("Failed to read graph stats: %s", e)

    return train_data[0]


def compute_optimal_batch_size(
    model: nn.Module,
    train_data,
    cfg: PipelineConfig,
    teacher: nn.Module | None = None,
    run_dir: Path | None = None,
) -> int:
    """Compute optimal batch size using memory analysis.

    Uses the p95 graph from cache metadata for conservative sizing.
    Falls back to safety_factor if estimation fails.  Results are cached
    to ``memory_cache.json`` in *run_dir* (if provided) for faster
    subsequent runs with the same config.
    """
    if len(train_data) == 0:
        log.warning("Empty training data, using fallback batch size")
        return effective_batch_size(cfg)

    # Check cache first
    if run_dir is not None:
        cached = load_budget_cache(run_dir, cfg)
        if cached is not None:
            log.info("Using cached batch size: %d", cached.recommended_batch_size)
            return cached.recommended_batch_size

    sample_graph = _get_representative_graph(train_data, cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Trial mode: binary search with actual forward+backward passes
    if cfg.training.memory_estimation == "trial":
        from ..memory import _trial_batch_size

        try:
            trial_bs = _trial_batch_size(
                model,
                train_data,
                device,
                min_bs=8,
                max_bs=cfg.training.batch_size,
                precision=cfg.training.precision,
            )
            log.info("Trial batch size: %d (max=%d)", trial_bs, cfg.training.batch_size)
            # Cache result using existing mechanism
            if run_dir is not None:
                budget = MemoryBudget(
                    total_gpu_mb=_get_gpu_memory_mb(device),
                    recommended_batch_size=trial_bs,
                    estimation_mode="trial",
                )
                save_budget_cache(budget, run_dir, cfg)
            return trial_bs
        except Exception as e:
            log.warning("Trial batch size failed: %s, falling back to measured", e)

    mode = (
        cfg.training.memory_estimation
        if cfg.training.memory_estimation in ("static", "measured")
        else "measured"
    )

    try:
        target_utilization = min(0.85, cfg.training.safety_factor + 0.15)

        budget = compute_batch_size(
            model=model,
            sample_graph=sample_graph,
            device=device,
            teacher=teacher,
            precision=cfg.training.precision,
            target_utilization=target_utilization,
            min_batch_size=8,
            max_batch_size=cfg.training.batch_size,
            mode=mode,
        )

        log.info(
            "Batch size: %d (mode=%s, max=%d, KD=%s)",
            budget.recommended_batch_size,
            mode,
            cfg.training.batch_size,
            teacher is not None,
        )

        # Save to cache for next run
        if run_dir is not None:
            save_budget_cache(budget, run_dir, cfg)

        return budget.recommended_batch_size

    except Exception as e:
        log.warning("Memory estimation failed: %s", e)

    return effective_batch_size(cfg)
