"""
Dataset loading with intelligent caching for CAN-Graph training.

Core function:
    load_dataset(): Load graph data with automatic cache management
"""

import json
import structlog
import os
from pathlib import Path

import torch

from graphids.config import (
    EDGE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
    PREPROCESSING_VERSION,
)

from ._cache_metadata import write_cache_metadata
from ._dataset import (
    CollatedGraphDataset,
    load_collated,
)

log = structlog.get_logger()


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
        from graphids.config import PREPROCESSING_DEFAULTS

        
        if window_size is None:
            window_size = PREPROCESSING_DEFAULTS["window_size"]
        if stride is None:
            stride = PREPROCESSING_DEFAULTS["stride"]

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

    log.info("dataset_created", total_graphs=len(dataset))

    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    log.info("dataset_split", training=len(train_dataset), validation=len(val_dataset))

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
            log.warning("Invalid cache format: id_mapping is not a dict.")
            return None, None

        log.info("cache_loaded", graphs=len(dataset), unique_ids=len(id_mapping))

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
                    log.warning("cache_stale_rebuilding", reasons="; ".join(stale_reasons))
                    return None, None

                if expected > 0 and actual < expected * 0.1:
                    log.warning(
                        "cache_graph_count_too_low",
                        actual=actual,
                        expected=expected,
                        version=version,
                    )
                    return None, None
                elif expected > 0 and actual < expected * 0.5:
                    log.warning(
                        "cache_fewer_graphs_than_expected",
                        actual=actual,
                        expected=expected,
                    )
                else:
                    log.info(
                        "cache_validated",
                        actual=actual,
                        expected=expected,
                        version=version,
                    )
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("cache_metadata_unreadable", error=str(e))
        else:
            log.info("cache_loaded_no_metadata", graphs=len(dataset))

        return dataset, id_mapping

    except (EOFError, AttributeError) as e:
        log.warning("cache_corrupted_rebuilding", error_type=type(e).__name__)
        try:
            cache_file.unlink(missing_ok=True)
            id_mapping_file.unlink(missing_ok=True)
        except OSError:
            pass
        return None, None
    except Exception as e:
        log.warning("cache_load_failed", error=str(e))
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
    log.info(
        "processing_dataset",
        mode="forced_rebuild" if force_rebuild else "from_scratch",
        path=str(dataset_path),
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    from ._parallel import process_dataset

    if adapter is None:
        from .adapters._can_bus import CANBusAdapter

        adapter = CANBusAdapter(include_attack_type=True)
    csv_files = adapter.discover_files(str(dataset_path), "train_")
    log.info("csv_files_found", count=len(csv_files), path=str(dataset_path))

    if not csv_files:
        raise FileNotFoundError(f"No train CSV files found in {dataset_path}")

    log.info("graph_creation_started")
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

    # Save cache atomically with advisory lock
    import fcntl
    import os
    import pickle
    from graphids.core.preprocessing._dataset import save_collated as _save_collated

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("saving_cache", format="collated", path=str(cache_file))

    try:
        lock_path = cache_file.parent / ".lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Atomic collated save
            tmp = cache_file.with_suffix(".tmp")
            _save_collated(graphs, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            tmp.rename(cache_file)

            # Atomic pickle save
            tmp_pkl = id_mapping_file.with_suffix(".tmp")
            with open(tmp_pkl, "wb") as f:
                pickle.dump(id_mapping, f, protocol=4)
            with open(tmp_pkl, "rb") as f:
                os.fsync(f.fileno())
            tmp_pkl.rename(id_mapping_file)

        log.info("cache_saved", format="collated", graphs=len(graphs))

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
        log.error("cache_save_failed", error=str(e))

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
        log.warning("no_id_mapping_skipping_test_data", path=str(id_mapping_file))
        return {}

    with open(id_mapping_file, "rb") as f:
        id_mapping = pickle.load(f)

    vocab = EntityVocabulary.from_dict(id_mapping)

    if not dataset_path.exists():
        log.warning("dataset_path_not_found_skipping_test_data", path=str(dataset_path))
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
                log.info("test_cache_loaded", scenario=name, graphs=len(dataset))
                scenarios[name] = dataset
                continue
            except Exception as e:
                log.warning("test_cache_load_failed", scenario=name, error=str(e))

        # Build from CSV using new pipeline
        log.info("building_test_graphs", scenario=name)
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
            import os
            from graphids.core.preprocessing._dataset import save_collated as _save_collated

            cache_dir_path.mkdir(parents=True, exist_ok=True)
            try:
                tmp = cache_file.with_suffix(".tmp")
                _save_collated(graphs, tmp)
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
                tmp.rename(cache_file)
                log.info(
                    "test_cache_saved",
                    scenario=name,
                    graphs=len(graphs),
                    path=str(cache_file),
                )
            except Exception as e:
                log.warning("test_cache_save_failed", scenario=name, error=str(e))

            scenarios[name] = load_collated(cache_file)
        else:
            log.warning("no_graphs_for_test_scenario", scenario=name)

    return scenarios
