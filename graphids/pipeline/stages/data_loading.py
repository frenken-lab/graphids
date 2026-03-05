"""Data loading and caching utilities for training stages."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from graphids.config import PipelineConfig, cache_dir, data_dir
from graphids.config.constants import MMAP_TENSOR_LIMIT, get_batch_index

log = logging.getLogger(__name__)


def graph_label(g) -> int:
    """Extract scalar graph-level label consistently."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


def load_data(cfg: PipelineConfig):
    """Load graph dataset. Returns (train_graphs, val_graphs, num_ids, in_channels)."""
    from graphids.core.training.datamodules import load_dataset

    train_data, val_data, num_ids = load_dataset(
        cfg.dataset,
        dataset_path=data_dir(cfg),
        cache_dir_path=cache_dir(cfg),
        seed=cfg.seed,
    )
    in_channels = train_data[0].x.shape[1] if train_data else 11
    return train_data, val_data, num_ids, in_channels


def _estimate_tensor_count(data) -> int:
    """Estimate number of tensor storages in a graph dataset."""
    if not data:
        return 0
    sample = data[0]
    tensors_per_graph = sum(
        1
        for attr in ["x", "edge_index", "y", "edge_attr", "batch"]
        if hasattr(sample, attr) and getattr(sample, attr) is not None
    )
    return len(data) * tensors_per_graph


def _safe_num_workers(data, cfg: PipelineConfig) -> int:
    """Return num_workers, falling back to 0 if dataset exceeds mmap limits.

    With spawn multiprocessing, every tensor storage needs a separate mmap
    entry.  Calling share_memory_() does NOT help -- it also creates one mmap
    per tensor.  The only safe option for large datasets is num_workers=0.
    """
    nw = cfg.num_workers
    if nw > 0 and cfg.mp_start_method == "spawn":
        tensor_count = _estimate_tensor_count(data)
        if tensor_count > MMAP_TENSOR_LIMIT:
            log.warning(
                "Dataset has %d tensor storages (limit %d for vm.max_map_count). "
                "Falling back to num_workers=0 to avoid mmap OOM.",
                tensor_count,
                MMAP_TENSOR_LIMIT,
            )
            return 0
    return nw


def compute_node_budget(batch_size: int, cfg: PipelineConfig) -> int | None:
    """Derive max_num_nodes from batch_size * p95 graph node count.

    Returns None when cache metadata is unavailable (falls back to static batching).
    """
    import json as _json

    metadata_path = cache_dir(cfg) / "cache_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        meta = _json.loads(metadata_path.read_text())
        p95 = meta.get("graph_stats", {}).get("node_count", {}).get("p95")
        if not p95:
            return None
        return int(batch_size * p95)
    except Exception as e:
        log.warning("Failed to read graph stats for node budget: %s", e)
        return None


def _estimate_dynamic_steps(data, max_num_nodes: int, batch_size: int) -> int:
    """Estimate actual batch count for DynamicBatchSampler.

    Samples a subset of graphs to compute mean node count, then estimates
    how many batches the sampler will yield given the node budget.
    Falls back to len(data) // batch_size if sampling fails.
    """
    try:
        # Sample up to 500 graphs for mean node count
        n_sample = min(500, len(data))
        total_nodes = sum(data[i].num_nodes for i in range(n_sample))
        mean_nodes = total_nodes / n_sample
        estimated_steps = max(1, int(len(data) * mean_nodes / max_num_nodes))
        return estimated_steps
    except Exception:
        return max(1, len(data) // max(1, batch_size))


def make_dataloader(
    data,
    cfg: PipelineConfig,
    batch_size: int,
    shuffle: bool = True,
    max_num_nodes: int | None = None,
) -> DataLoader:
    """Create a DataLoader with consistent settings.

    When *max_num_nodes* is provided, uses ``DynamicBatchSampler`` to pack
    variable-size graphs up to a node budget per batch.  Falls back to
    single-process loading (num_workers=0) when the dataset has too many
    tensor storages for the kernel mmap limit.
    """
    nw = _safe_num_workers(data, cfg)

    if max_num_nodes is not None:
        # num_steps required so Lightning can call len(dataloader).
        # Must reflect actual iteration count, not len(data)//batch_size.
        # DynamicBatchSampler packs graphs by node budget, so each batch
        # holds many more graphs than batch_size when graphs are small.
        num_steps = _estimate_dynamic_steps(data, max_num_nodes, batch_size)
        sampler = DynamicBatchSampler(
            data,
            max_num=max_num_nodes,
            mode="node",
            shuffle=shuffle,
            num_steps=num_steps,
        )
        return DataLoader(
            data,
            batch_sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            multiprocessing_context=cfg.mp_start_method if nw > 0 else None,
        )

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        multiprocessing_context=cfg.mp_start_method if nw > 0 else None,
    )


def cache_predictions(models: dict[str, nn.Module], data, device, max_samples: int = 150_000):
    """Run registered extractors over data, produce N-D state vectors for DQN.

    ``models`` maps model_type name to loaded model (e.g. ``{"vgae": vgae, "gat": gat}``).
    Feature concatenation order follows registry registration order (VGAE then GAT)
    to preserve the existing 15-D layout.
    """
    from graphids.core.models.registry import extractors as registry_extractors

    registered = registry_extractors()
    active = [(name, ext) for name, ext in registered if name in models]

    states, labels = [], []
    for model in models.values():
        model.eval()
    n_samples = min(len(data), max_samples)

    with torch.no_grad():
        for i in range(n_samples):
            g = data[i].clone().to(device)
            batch_idx = get_batch_index(g, device)

            features = [ext.extract(models[name], g, batch_idx, device) for name, ext in active]
            states.append(torch.cat(features))
            labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}
