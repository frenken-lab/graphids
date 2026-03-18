"""
Dataset loading with intelligent caching for CAN-Graph training.

Core function:
    load_dataset(): Load graph data with automatic cache management
"""

import json
import logging
import os
from pathlib import Path

import torch

from graphids.config import (
    EDGE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_VERSION,
)

from ._atomic_io import atomic_rename, atomic_save_collated
from ._cache_metadata import write_cache_metadata
from ._dataset import (
    CollatedGraphDataset,
    load_collated,
)

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
    train_val_split: float = 0.8,
    adapter=None,
    window_size: int | None = None,
    stride: int | None = None,
):
    """
    Load and prepare dataset with intelligent caching.

    Args:
        dataset_name: Dataset name (hcrl_sa, set_01, etc.)
        dataset_path: Path to the raw dataset directory
        cache_dir_path: Path to the cache directory for processed graphs
        force_rebuild_cache: Force rebuild cached data
        seed: Random seed for train/val split
        train_val_split: Fraction of data for training
        adapter: Domain adapter (defaults to CANBusAdapter)
        window_size: Sliding window size (for cache validation; defaults from config)
        stride: Sliding window stride (for cache validation; defaults from config)

    Returns:
        Tuple of (train_dataset, val_dataset, num_unique_ids)
    """
    if window_size is None or stride is None:
        from graphids.config import PreprocessingConfig

        _defaults = PreprocessingConfig()
        if window_size is None:
            window_size = _defaults.window_size
        if stride is None:
            stride = _defaults.stride

    cache_file = cache_dir_path / "processed_graphs.pt"
    id_mapping_file = cache_dir_path / "id_mapping.pkl"

    dataset, id_mapping = None, None

    if not force_rebuild_cache:
        dataset, id_mapping = _load_cached_data(
            cache_file,
            id_mapping_file,
            dataset_name,
            window_size=window_size,
            stride=stride,
        )

    # Process from scratch if needed
    if dataset is None or id_mapping is None:
        dataset, id_mapping = _process_dataset_from_scratch(
            dataset_path,
            dataset_name,
            cache_file,
            id_mapping_file,
            force_rebuild_cache,
            adapter=adapter,
            window_size=window_size,
            stride=stride,
        )

    logger.info(f"Created dataset with {len(dataset)} total graphs")

    train_size = int(train_val_split * len(dataset))
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


def _load_cached_data(
    cache_file,
    id_mapping_file,
    dataset_name,
    *,
    window_size: int,
    stride: int,
):
    """Load cached collated graphs and ID mapping with robust error handling.

    Supports both the new collated format (data_dict + slices) and legacy
    list[Data] format (auto-converted on load, rebuilt on next preprocessing).
    """
    if not (cache_file.exists() and id_mapping_file.exists()):
        return None, None

    try:
        import pickle

        dataset = load_collated(cache_file)

        with open(id_mapping_file, "rb") as f:
            id_mapping = pickle.load(f)

        # Validate loaded data
        if not isinstance(id_mapping, dict):
            logger.warning("Invalid cache format: id_mapping is not a dict.")
            return None, None

        logger.info(f"Loaded {len(dataset)} cached graphs with {len(id_mapping)} unique IDs")

        # Validate cache using metadata if available
        metadata_file = cache_file.parent / "cache_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                expected = metadata.get("num_graphs", 0)
                actual = len(dataset)
                version = metadata.get("preprocessing_version", "unknown")

                # Check preprocessing params match current config
                stale_reasons = []
                if version != PREPROCESSING_VERSION:
                    stale_reasons.append(f"version {version} != {PREPROCESSING_VERSION}")
                # Map metadata keys to current expected values
                expected_vals = {
                    "window_size": window_size,
                    "stride": stride,
                    "node_feature_dim": NODE_FEATURE_COUNT,
                    "edge_feature_dim": EDGE_FEATURE_COUNT,
                }
                for key, current_val in expected_vals.items():
                    cached_val = metadata.get(key)
                    if cached_val is not None and cached_val != current_val:
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
            logger.info(f"No cache metadata found. Loaded {len(dataset)} graphs.")

        return dataset, id_mapping

    except (EOFError, AttributeError) as e:
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
    adapter=None,
    window_size: int = 100,
    stride: int = 100,
):
    """Process dataset from CSV files and save in collated format."""
    logger.info(
        f"Processing dataset: {'forced rebuild' if force_rebuild else 'processing from scratch'}..."
    )
    logger.info(f"Dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    from ._parallel import process_dataset

    if adapter is None:
        from .adapters._can_bus import CANBusAdapter

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
        adapter=adapter,
    )

    if hasattr(graphs, "data_list"):
        graphs = graphs.data_list
    if not isinstance(graphs, list):
        graphs = list(graphs)

    # Save cache atomically in collated format, with advisory lock
    import pickle

    from graphids.lake.locking import cache_lock

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed data to cache (collated format): {cache_file}")

    temp_cache = cache_file.with_suffix(".tmp")
    temp_mapping = id_mapping_file.with_suffix(".tmp")

    try:
        with cache_lock(cache_file.parent):
            atomic_save_collated(graphs, temp_cache, cache_file)
            with open(temp_mapping, "wb") as f:
                pickle.dump(id_mapping, f, protocol=4)
            with open(temp_mapping, "rb") as f:
                os.fsync(f.fileno())
            atomic_rename(temp_mapping, id_mapping_file)

        logger.info(f"Cache saved (collated): {len(graphs)} graphs")

        # Write cache metadata for validation on future loads
        write_cache_metadata(
            cache_file.parent,
            dataset_name,
            graphs,
            id_mapping,
            csv_files,
            window_size=window_size,
            stride=stride,
        )
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        temp_cache.unlink(missing_ok=True)
        temp_mapping.unlink(missing_ok=True)

    # Return a CollatedGraphDataset (reload from saved file for mmap)
    return load_collated(cache_file), id_mapping


def load_test_scenarios(
    dataset_name: str,
    dataset_path: Path,
    cache_dir_path: Path,
    force_rebuild_cache: bool = False,
    adapter=None,
) -> dict[str, CollatedGraphDataset]:
    """Load held-out test scenarios with per-scenario caching (collated format).

    Each test scenario (test_01_..., test_02_..., etc.) is cached as a
    separate collated .pt file in the cache directory, using the training
    id_mapping for consistent CAN ID encoding.

    Returns:
        Dict mapping scenario name -> CollatedGraphDataset.
    """
    import pickle

    from ._parallel import process_dataset
    from ._vocabulary import EntityVocabulary

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

    scenarios: dict[str, CollatedGraphDataset] = {}
    for folder in sorted(dataset_path.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("test_")):
            continue

        name = folder.name
        cache_file = cache_dir_path / f"{name}.pt"

        # Try loading from cache
        if not force_rebuild_cache and cache_file.exists():
            try:
                dataset = load_collated(cache_file)
                logger.info("Loaded %d cached test graphs for %s", len(dataset), name)
                scenarios[name] = dataset
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
            adapter=adapter,
        )
        if hasattr(graphs, "data_list"):
            graphs = graphs.data_list
        if not isinstance(graphs, list):
            graphs = list(graphs)

        if graphs:
            # Save cache atomically in collated format
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            try:
                atomic_save_collated(graphs, tmp, cache_file)
                logger.info(
                    "Cached %d test graphs (collated) for %s -> %s", len(graphs), name, cache_file
                )
            except Exception as e:
                logger.warning("Failed to cache test graphs for %s: %s", name, e)
                tmp.unlink(missing_ok=True)

            scenarios[name] = load_collated(cache_file)
        else:
            logger.warning("No graphs created for test scenario %s", name)

    return scenarios
