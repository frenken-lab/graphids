"""Shared utilities for training stages.

This module re-exports from focused submodules for backward compatibility.
New code should import from the specific submodule directly.
"""

from __future__ import annotations

import gc
import logging

import torch

from .batch_sizing import (
    compute_optimal_batch_size,
    effective_batch_size,
    resolve_batch_config,
)
from .callbacks import (
    MemoryMonitorCallback,
    ProfilerCallback,
)
from .data_loading import (
    cache_predictions,
    compute_node_budget,
    graph_label,
    load_data,
    make_dataloader,
    training_preamble,
)
from .trainer_factory import (
    _cross_model_path,
    _extract_state_dict,
    build_optimizer_dict,
    load_frozen_cfg,
    load_model,
    load_teacher,
    make_projection,
    make_trainer,
)

log = logging.getLogger(__name__)

__all__ = [
    "MemoryMonitorCallback",
    "ProfilerCallback",
    "_cross_model_path",
    "build_optimizer_dict",
    "cache_predictions",
    "cleanup",
    "compute_node_budget",
    "compute_optimal_batch_size",
    "effective_batch_size",
    "graph_label",
    "load_data",
    "load_frozen_cfg",
    "load_model",
    "load_teacher",
    "make_dataloader",
    "make_projection",
    "make_trainer",
    "resolve_batch_config",
    "training_preamble",
]


def cleanup():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
