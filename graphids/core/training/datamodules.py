"""
Dataset loading with intelligent caching for CAN-Graph training.

Core function:
    load_dataset(): Load graph data with automatic cache management
"""

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

import graphids.config.constants as constants
from graphids.config.constants import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    EDGE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_VERSION,
)
from graphids.core.preprocessing.dataset import GraphDataset

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Loading and Caching
# ============================================================================


def load_dataset(
    dataset_name: str,
    dataset_path: Path,
    cache_dir_path: Path,
    force_rebuild_cache: bool = False,
    seed: int = 42,
):
    """
    Load and prepare dataset with intelligent caching.

    Args:
        dataset_name: Dataset name (hcrl_sa, set_01, etc.)
        dataset_path: Path to the raw dataset directory
        cache_dir_path: Path to the cache directory for processed graphs
        force_rebuild_cache: Force rebuild cached data

    Returns:
        Tuple of (train_dataset, val_dataset, num_unique_ids)
    """
    cache_file = cache_dir_path / "processed_graphs.pt"
    id_mapping_file = cache_dir_path / "id_mapping.pkl"

    graphs, id_mapping = None, None

    if not force_rebuild_cache:
        graphs, id_mapping = _load_cached_data(
            cache_file,
            id_mapping_file,
            dataset_name,
        )

    # Process from scratch if needed
    if graphs is None or id_mapping is None:
        graphs, id_mapping = _process_dataset_from_scratch(
            dataset_path,
            dataset_name,
            cache_file,
            id_mapping_file,
            force_rebuild_cache,
        )

    # Unwrap GraphDataset to plain list if needed (avoid double-wrapping)
    if hasattr(graphs, "data_list"):
        graphs = graphs.data_list

    # Create dataset and split
    dataset = GraphDataset(graphs)
    logger.info(f"Created dataset with {len(dataset)} total graphs")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation")

    num_ids = len(id_mapping) if id_mapping else 1000
    return train_dataset, val_dataset, num_ids


# ============================================================================
# Internal Helper Functions
# ============================================================================


def _load_cached_data(cache_file, id_mapping_file, dataset_name):
    """Load cached graphs and ID mapping with robust error handling.

    Note: Uses weights_only=False to support PyTorch Geometric Data objects.
    This is safe for our own cached data but should not be used with untrusted files.

    Memory optimization (PyTorch 2.1+): Uses mmap=True to memory-map the cache file,
    reducing RAM usage by loading graph data on-demand rather than all at once.
    This is especially important for large datasets (set_01-04, hcrl_ch).
    """
    if not (cache_file.exists() and id_mapping_file.exists()):
        return None, None

    try:
        import pickle

        # Memory-mapped loading: graphs are loaded on-demand from disk
        # instead of fully deserializing into RAM. Requires PyTorch 2.1+.
        # This significantly reduces memory for large datasets.
        try:
            graphs = torch.load(
                cache_file,
                map_location="cpu",
                weights_only=False,
                mmap=True,  # Memory-map for reduced RAM usage
            )
        except TypeError:
            # Fallback for older PyTorch versions without mmap support
            logger.warning("mmap not supported (PyTorch < 2.1), using standard load")
            graphs = torch.load(cache_file, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"Cache load failed (possibly corrupted): {e}")
            return None, None

        with open(id_mapping_file, "rb") as f:
            id_mapping = pickle.load(f)

        # Validate loaded data
        if not isinstance(id_mapping, dict):
            logger.warning("Invalid cache format: id_mapping is not a dict.")
            return None, None
        # Handle GraphDataset objects saved as cache
        if hasattr(graphs, "data_list"):
            graphs = graphs.data_list
        if not isinstance(graphs, (list, tuple)):
            try:
                graphs = list(graphs)
            except (TypeError, ValueError):
                logger.warning("Invalid cache format: graphs is not list-like.")
                return None, None

        logger.info(f"Loaded {len(graphs)} cached graphs with {len(id_mapping)} unique IDs")

        # Validate cache using metadata if available
        metadata_file = cache_file.parent / "cache_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                expected = metadata.get("num_graphs", 0)
                actual = len(graphs)
                version = metadata.get("preprocessing_version", "unknown")

                # Check preprocessing params match current config
                stale_reasons = []
                if version != PREPROCESSING_VERSION:
                    stale_reasons.append(f"version {version} != {PREPROCESSING_VERSION}")
                for key in ("window_size", "stride", "node_feature_dim", "edge_feature_dim"):
                    cached_val = metadata.get(key)
                    const_name = key.upper()
                    # Map metadata keys to constant names
                    const_map = {
                        "WINDOW_SIZE": "DEFAULT_WINDOW_SIZE",
                        "STRIDE": "DEFAULT_STRIDE",
                        "NODE_FEATURE_DIM": "NODE_FEATURE_COUNT",
                        "EDGE_FEATURE_DIM": "EDGE_FEATURE_COUNT",
                    }
                    actual_const = const_map.get(const_name, const_name)
                    current_val = getattr(constants, actual_const, None)
                    if (
                        cached_val is not None
                        and current_val is not None
                        and cached_val != current_val
                    ):
                        stale_reasons.append(f"{key}: {cached_val} != {current_val}")
                if stale_reasons:
                    logger.warning("Cache stale (%s). Rebuilding.", "; ".join(stale_reasons))
                    return None, None

                if expected > 0 and actual < expected * 0.1:
                    logger.warning(
                        f"CACHE ISSUE: Only {actual} graphs found, expected {expected} "
                        f"(preprocessing v{version}). Rebuilding cache."
                    )
                    return None, None
                elif expected > 0 and actual < expected * 0.5:
                    logger.warning(
                        f"Cache has fewer graphs than expected: {actual} vs {expected}. "
                        "Use --force-rebuild to recreate."
                    )
                else:
                    logger.info(
                        f"Cache validated: {actual} graphs (expected {expected}, v{version})"
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Cache metadata unreadable: {e}. Proceeding with loaded data.")
        else:
            logger.info(f"No cache metadata found. Loaded {len(graphs)} graphs.")

        return graphs, id_mapping

    except (pickle.UnpicklingError, AttributeError, EOFError) as e:
        logger.warning(f"Cache file corrupted ({type(e).__name__}). Deleting and rebuilding.")
        try:
            cache_file.unlink(missing_ok=True)
            id_mapping_file.unlink(missing_ok=True)
        except OSError:
            pass
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load cached data: {e}. Processing from scratch.")
        return None, None


def _process_dataset_from_scratch(
    dataset_path,
    dataset_name,
    cache_file,
    id_mapping_file,
    force_rebuild,
):
    """Process dataset from CSV files using the new adapter/engine pipeline."""
    logger.info(
        f"Processing dataset: {'forced rebuild' if force_rebuild else 'processing from scratch'}..."
    )
    logger.info(f"Dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    from graphids.core.preprocessing.adapters.can_bus import CANBusAdapter
    from graphids.core.preprocessing.parallel import process_dataset

    adapter = CANBusAdapter(include_attack_type=True)
    csv_files = adapter.discover_files(str(dataset_path), "train_")
    logger.info("Found %d CSV files in %s", len(csv_files), dataset_path)

    if not csv_files:
        raise FileNotFoundError(f"No train CSV files found in {dataset_path}")

    logger.info("Starting graph creation from CSV files...")
    graphs, id_mapping = process_dataset(
        dataset_path,
        split="train_",
        return_vocab=True,
        verbose=True,
        include_attack_type=True,
    )

    # Unwrap GraphDataset if returned
    if hasattr(graphs, "data_list"):
        graphs = graphs.data_list
    if not isinstance(graphs, list):
        graphs = list(graphs)

    # Save cache atomically
    import pickle

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed data to cache: {cache_file}")

    temp_cache = cache_file.with_suffix(".tmp")
    temp_mapping = id_mapping_file.with_suffix(".tmp")

    try:
        torch.save(graphs, temp_cache, pickle_protocol=4)
        with open(temp_mapping, "wb") as f:
            pickle.dump(id_mapping, f, protocol=4)

        # Flush to disk to ensure NFS visibility before rename
        for tmp in (temp_cache, temp_mapping):
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())

        # Retry rename with backoff (NFS may delay visibility)
        for tmp, final in ((temp_cache, cache_file), (temp_mapping, id_mapping_file)):
            for attempt in range(3):
                try:
                    tmp.rename(final)
                    break
                except OSError as e:
                    if attempt < 2:
                        logger.warning(
                            "Cache rename attempt %d failed: %s. Retrying...", attempt + 1, e
                        )
                        time.sleep(1)
                    else:
                        raise
        logger.info(f"Cache saved: {len(graphs)} graphs")

        # Write cache metadata for validation on future loads
        _write_cache_metadata(cache_file.parent, dataset_name, graphs, id_mapping, csv_files)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        temp_cache.unlink(missing_ok=True)
        temp_mapping.unlink(missing_ok=True)

    return graphs, id_mapping


def load_test_scenarios(
    dataset_name: str,
    dataset_path: Path,
    cache_dir_path: Path,
    force_rebuild_cache: bool = False,
) -> dict[str, list]:
    """Load held-out test scenarios with per-scenario caching.

    Each test scenario (test_01_..., test_02_..., etc.) is cached as a
    separate .pt file in the cache directory, using the training id_mapping
    for consistent CAN ID encoding.

    Returns:
        Dict mapping scenario name → list of PyG Data objects.
    """
    import pickle

    from graphids.core.preprocessing.parallel import process_dataset
    from graphids.core.preprocessing.vocabulary import EntityVocabulary

    id_mapping_file = cache_dir_path / "id_mapping.pkl"
    if not id_mapping_file.exists():
        logger.warning("No id_mapping at %s -- skipping test data", id_mapping_file)
        return {}

    with open(id_mapping_file, "rb") as f:
        id_mapping = pickle.load(f)

    vocab = EntityVocabulary.from_legacy_mapping(id_mapping)

    if not dataset_path.exists():
        logger.warning("Dataset path %s not found -- skipping test data", dataset_path)
        return {}

    scenarios: dict[str, list] = {}
    for folder in sorted(dataset_path.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("test_")):
            continue

        name = folder.name
        cache_file = cache_dir_path / f"{name}.pt"

        # Try loading from cache
        if not force_rebuild_cache and cache_file.exists():
            try:
                graphs = torch.load(cache_file, map_location="cpu", weights_only=False)
                if hasattr(graphs, "data_list"):
                    graphs = graphs.data_list
                if not isinstance(graphs, list):
                    graphs = list(graphs)
                logger.info("Loaded %d cached test graphs for %s", len(graphs), name)
                scenarios[name] = graphs
                continue
            except Exception as e:
                logger.warning("Test cache load failed for %s: %s. Rebuilding.", name, e)

        # Build from CSV using new pipeline
        logger.info("Building test graphs for %s", name)
        graphs = process_dataset(
            str(dataset_path),
            split=name,
            vocab=vocab,
            return_vocab=False,
        )
        if hasattr(graphs, "data_list"):
            graphs = graphs.data_list
        if not isinstance(graphs, list):
            graphs = list(graphs)

        if graphs:
            # Save cache atomically
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            try:
                torch.save(graphs, tmp, pickle_protocol=4)
                with open(tmp, "rb") as fh:
                    os.fsync(fh.fileno())
                tmp.rename(cache_file)
                logger.info("Cached %d test graphs for %s → %s", len(graphs), name, cache_file)
            except Exception as e:
                logger.warning("Failed to cache test graphs for %s: %s", name, e)
                tmp.unlink(missing_ok=True)

            scenarios[name] = graphs
        else:
            logger.warning("No graphs created for test scenario %s", name)

    return scenarios


def _compute_graph_stats(graphs) -> dict:
    """Compute per-graph statistics for batch size estimation."""
    import numpy as np

    # Unwrap GraphDataset to plain list if needed
    if hasattr(graphs, "data_list"):
        graphs = graphs.data_list

    node_counts = []
    edge_counts = []
    per_graph_bytes = []

    for g in graphs:
        node_counts.append(g.x.size(0) if g.x is not None else 0)
        edge_counts.append(g.edge_index.size(1) if g.edge_index is not None else 0)
        byte_count = 0
        for attr in ("x", "edge_index", "edge_attr", "y"):
            t = getattr(g, attr, None)
            if t is not None:
                byte_count += t.numel() * t.element_size()
        per_graph_bytes.append(byte_count)

    def _stats(values):
        arr = np.array(values)
        return {
            "mean": round(float(arr.mean()), 1),
            "median": int(np.median(arr)),
            "p95": int(np.percentile(arr, 95)),
            "max": int(arr.max()),
        }

    return {
        "node_count": _stats(node_counts),
        "edge_count": _stats(edge_counts),
        "per_graph_bytes": _stats(per_graph_bytes),
    }


def _write_cache_metadata(cache_dir, dataset_name, graphs, id_mapping, csv_files):
    """Write cache_metadata.json alongside processed cache files."""
    import torch_geometric

    metadata = {
        "dataset": dataset_name,
        "created_at": datetime.now(UTC).isoformat(),
        "window_size": DEFAULT_WINDOW_SIZE,
        "stride": DEFAULT_STRIDE,
        "num_graphs": len(graphs),
        "num_unique_ids": len(id_mapping) if id_mapping else 0,
        "node_feature_dim": NODE_FEATURE_COUNT,
        "edge_feature_dim": EDGE_FEATURE_COUNT,
        "source_csv_count": len(csv_files),
        "preprocessing_version": PREPROCESSING_VERSION,
        "torch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__,
    }

    try:
        metadata["graph_stats"] = _compute_graph_stats(graphs)
        logger.info("Graph stats: %s", metadata["graph_stats"]["node_count"])
    except Exception as e:
        logger.warning("Failed to compute graph stats: %s", e)

    metadata_file = Path(cache_dir) / "cache_metadata.json"
    try:
        metadata_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Cache metadata written to {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to write cache metadata: {e}")
