"""Cache metadata and graph statistics for preprocessing cache validation."""

from __future__ import annotations

import json
import structlog
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from graphids.config import (
    EDGE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_VERSION,
    compute_preprocessing_hash,
)

from ._dataset import CollatedGraphDataset
from ._schema import EDGE_MANIFEST, NODE_MANIFEST

log = structlog.get_logger()


def compute_graph_stats(graphs) -> dict:
    """Compute per-graph statistics for batch size estimation."""
    if isinstance(graphs, CollatedGraphDataset):
        x_slices = graphs._slices.get("x")
        ei_slices = graphs._slices.get("edge_index")
        if x_slices is not None:
            nc = (x_slices[1:] - x_slices[:-1]).numpy()
        else:
            nc = np.zeros(len(graphs))
        if ei_slices is not None:
            ec = (ei_slices[1:] - ei_slices[:-1]).numpy()
        else:
            ec = np.zeros(len(graphs))

        nf = graphs._data.x.size(1) if graphs._data.x is not None else 0
        ef = graphs._data.edge_attr.size(1) if graphs._data.edge_attr is not None else 0
        per_graph_bytes = (nc * nf * 4 + ec * 2 * 8 + ec * ef * 4 + 4).astype(float)

        def _stats(values):
            return {
                "mean": round(float(np.mean(values)), 1),
                "median": int(np.median(values)),
                "p95": int(np.percentile(values, 95)),
                "max": int(np.max(values)),
            }

        return {
            "node_count": _stats(nc),
            "edge_count": _stats(ec),
            "per_graph_bytes": _stats(per_graph_bytes),
        }

    raise TypeError(f"Expected CollatedGraphDataset, got {type(graphs).__name__}")


def write_cache_metadata(
    cache_dir,
    dataset_name,
    graphs,
    id_mapping,
    csv_files,
    window_size: int,
    stride: int,
):
    """Write cache_metadata.json and feature_manifest.json alongside processed cache files."""
    import torch_geometric

    metadata = {
        "dataset": dataset_name,
        "created_at": datetime.now(UTC).isoformat(),
        "window_size": window_size,
        "stride": stride,
        "num_graphs": len(graphs),
        "num_unique_ids": len(id_mapping) if id_mapping else 0,
        "node_feature_dim": NODE_FEATURE_COUNT,
        "edge_feature_dim": EDGE_FEATURE_COUNT,
        "source_csv_count": len(csv_files),
        "preprocessing_version": PREPROCESSING_VERSION,
        "config_hash": compute_preprocessing_hash(),
        "torch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__,
        "storage_format": "collated",
    }

    try:
        metadata["graph_stats"] = compute_graph_stats(graphs)
        log.info("graph_stats_computed", node_count=metadata["graph_stats"]["node_count"])
    except Exception as e:
        log.warning("graph_stats_computation_failed", error=str(e))

    from graphids.storage import StorageGateway

    gw = StorageGateway(
        lake_root=".", dataset="cache", model_type="cache", scale="cache",
    )

    cache_path = Path(cache_dir)
    metadata_file = cache_path / "cache_metadata.json"
    try:
        gw.write_json(metadata_file, metadata)
        log.info("cache_metadata_written", path=str(metadata_file))
    except Exception as e:
        log.warning("cache_metadata_write_failed", error=str(e))

    # Write feature manifest alongside cache
    manifest_file = cache_path / "feature_manifest.json"
    try:
        manifest_data = {
            "node_features": NODE_MANIFEST.to_json(),
            "edge_features": EDGE_MANIFEST.to_json(),
        }
        gw.write_json(manifest_file, manifest_data)
        log.info("feature_manifest_written", path=str(manifest_file))
    except Exception as e:
        log.warning("feature_manifest_write_failed", error=str(e))
