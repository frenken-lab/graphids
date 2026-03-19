"""Batch size resolution for training stages.

Uses the configured batch_size with safety_factor applied, plus optional
DynamicBatchSampler node budget for variable-size graphs. The previous
custom GPU memory estimation (471 lines in memory.py) has been replaced
by this simple config-driven approach — Lightning Tuner or manual tuning
can adjust batch_size in config YAML if needed.
"""

from __future__ import annotations

import structlog

from graphids.config import PipelineConfig

log = structlog.get_logger()


def effective_batch_size(cfg: PipelineConfig) -> int:
    """Apply safety factor to configured batch size."""
    return max(8, int(cfg.training.batch_size * cfg.training.safety_factor))


def resolve_batch_config(cfg: PipelineConfig):
    """Compute batch size and optional dynamic batching node budget."""
    from .data_loading import compute_node_budget

    bs = effective_batch_size(cfg)

    max_nodes = None
    if cfg.training.dynamic_batching:
        max_nodes = compute_node_budget(bs, cfg)
        if max_nodes:
            log.info("dynamic_batching_enabled", max_num_nodes=max_nodes, batch_size=bs)
        else:
            log.info("dynamic_batching_fallback_to_static", batch_size=bs)
    return bs, max_nodes
