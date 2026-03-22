"""Data loading and caching utilities for training stages."""

from __future__ import annotations

import gc

import structlog
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from graphids.config import cache_dir

log = structlog.get_logger()


def cleanup():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def compute_node_budget(batch_size: int, cfg) -> int | None:
    """Derive max_num_nodes from batch_size * p95 graph node count.

    Returns None when cache metadata is unavailable (falls back to static batching).
    """
    import json

    metadata_path = cache_dir(cfg.lake_root, cfg.dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        meta = json.loads(metadata_path.read_text())
        p95 = meta.get("graph_stats", {}).get("node_count", {}).get("p95")
        return int(batch_size * p95) if p95 else None
    except Exception as e:
        log.warning("graph_stats_read_failed", error=str(e))
        return None


def make_dataloader(
    data,
    cfg,
    batch_size: int,
    shuffle: bool = True,
    max_num_nodes: int | None = None,
) -> DataLoader:
    """Create a DataLoader with consistent settings.

    Uses DynamicBatchSampler when max_num_nodes is provided.
    Spawn multiprocessing is hardcoded for CUDA safety.
    """
    nw = cfg.num_workers

    common = dict(
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        multiprocessing_context="spawn" if nw > 0 else None,
    )

    if max_num_nodes is not None:
        # Estimate actual batch count for DynamicBatchSampler
        n_sample = min(500, len(data))
        indices = torch.randperm(len(data))[:n_sample].tolist()
        mean_nodes = sum(data[i].num_nodes for i in indices) / max(n_sample, 1)
        num_steps = max(1, int(len(data) * mean_nodes / max_num_nodes))

        sampler = DynamicBatchSampler(
            data, max_num=max_num_nodes, mode="node", shuffle=shuffle, num_steps=num_steps,
        )
        return DataLoader(data, batch_sampler=sampler, **common)

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, **common)


def cache_predictions(models: dict[str, nn.Module], data, device, max_samples: int = 150_000, batch_size: int = 256):
    """Run registered extractors over data, produce N-D state vectors for DQN.

    Uses a DataLoader for batched clone+transfer, then extracts per-graph
    features within each on-device batch (extractors are not batch-aware).
    """
    from graphids.core.models.registry import extractors as registry_extractors
    from graphids.core.preprocessing import get_batch_index

    active = [(name, ext) for name, ext in registry_extractors() if name in models]
    for model in models.values():
        model.eval()

    capped = data[:max_samples]
    loader = DataLoader(capped, batch_size=batch_size, shuffle=False)

    states, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for g in batch.to_data_list():
                batch_idx = get_batch_index(g, device)
                features = [ext.extract(models[name], g, batch_idx, device) for name, ext in active]
                states.append(torch.cat(features))
                labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}
