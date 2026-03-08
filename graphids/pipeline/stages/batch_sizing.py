"""Batch size resolution for training stages.

Uses the configured batch_size with safety_factor applied, plus optional
DynamicBatchSampler node budget for variable-size graphs. The previous
custom GPU memory estimation (471 lines in memory.py) has been replaced
by this simple config-driven approach — Lightning Tuner or manual tuning
can adjust batch_size in config YAML if needed.
"""

from __future__ import annotations

import logging

from graphids.config import PipelineConfig

log = logging.getLogger(__name__)


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
            log.info("Dynamic batching: max_num_nodes=%d (batch_size=%d × p95)", max_nodes, bs)
        else:
            log.info(
                "Dynamic batching: no cache metadata, falling back to static batch_size=%d", bs
            )
    return bs, max_nodes
