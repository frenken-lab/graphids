"""Parallel preprocessing driver using Ray for per-file fan-out.

Replaces the monolithic ``graph_creation()`` function with a pipeline that
uses the new adapter/engine architecture:

    CANBusAdapter.discover_files()  →  per-file Ray tasks  →  GraphEngine  →  graphs

Each file is processed independently via ``@ray.remote``, making this
embarrassingly parallel. Falls back to sequential processing when Ray
is not available or when running in local mode with few files.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from graphids.config.constants import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
)

from .adapters.can_bus import CANBusAdapter
from .engine import GraphEngine
from .schema import IRSchema

if TYPE_CHECKING:
    from torch_geometric.data import Data
from .vocabulary import EntityVocabulary

log = logging.getLogger(__name__)

# Threshold: use Ray only when there are enough files to justify overhead
_RAY_FILE_THRESHOLD = 4


def _process_single_file(
    file_path: str,
    vocab_dict: dict,
    schema: IRSchema,
    window_size: int,
    stride: int,
    include_attack_type: bool = True,
) -> list[Data]:
    """Process one CSV file → list of PyG Data objects.

    This function is the unit of work for both sequential and Ray execution.
    It receives a plain dict (not EntityVocabulary) to avoid pickling issues.
    """
    vocab = EntityVocabulary(vocab_dict)
    adapter = CANBusAdapter(include_attack_type=include_attack_type)
    engine = GraphEngine(schema, window_size=window_size, stride=stride)

    ir_df = adapter.read_and_convert(file_path, vocab)
    if ir_df.empty:
        return []

    # Fill any residual NaN (defensive)
    if ir_df.isnull().values.any():
        ir_df.fillna(0, inplace=True)

    return engine.create_graphs(ir_df)


def process_dataset(
    root: str | Path,
    split: str = "train_",
    vocab: EntityVocabulary | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    return_vocab: bool = False,
    verbose: bool = False,
    include_attack_type: bool = True,
) -> list[Data] | tuple[list[Data], dict]:
    """Process a dataset directory into PyG graphs using the new pipeline.

    Parameters
    ----------
    root : path
        Root directory containing the dataset.
    split : str
        Split identifier (``"train_"``, ``"test_01_DoS"``, etc.).
    vocab : EntityVocabulary, optional
        Pre-built vocabulary. If None, builds from discovered files.
    window_size, stride : int
        Sliding window parameters.
    return_vocab : bool
        If True, return ``(graphs, vocab_dict)`` tuple.
    verbose : bool
        Log progress details.
    include_attack_type : bool
        If True, include attack_type metadata on each graph (default True).

    Returns
    -------
    list[Data] or (list[Data], dict)
        Graphs, and optionally the vocabulary dict for caching.
    """
    adapter = CANBusAdapter(include_attack_type=include_attack_type)
    files = adapter.discover_files(root, split)

    if not files:
        log.warning("No CSV files found in %s for split '%s'", root, split)
        empty: list[Data] = []
        return (empty, {"OOV": 0}) if return_vocab else empty

    log.info("Found %d CSV files to process", len(files))

    # Build vocabulary if not provided
    if vocab is None:
        log.info("Building vocabulary from %d files...", len(files))
        vocab = adapter.build_vocabulary(files)
    vocab_dict = vocab.to_dict()

    # Decide: Ray parallel vs sequential
    all_graphs = _dispatch(
        files=files,
        vocab_dict=vocab_dict,
        schema=adapter.schema,
        window_size=window_size,
        stride=stride,
        verbose=verbose,
        include_attack_type=include_attack_type,
    )

    log.info("Total graphs created: %d", len(all_graphs))

    if return_vocab:
        return all_graphs, vocab_dict
    return all_graphs


def _dispatch(
    files: Sequence[Path],
    vocab_dict: dict,
    schema: IRSchema,
    window_size: int,
    stride: int,
    verbose: bool,
    include_attack_type: bool = True,
) -> list[Data]:
    """Choose between Ray parallel and sequential processing."""
    use_ray = len(files) >= _RAY_FILE_THRESHOLD and _ray_available()

    if use_ray:
        return _process_ray(
            files, vocab_dict, schema, window_size, stride, verbose, include_attack_type
        )
    return _process_sequential(
        files, vocab_dict, schema, window_size, stride, verbose, include_attack_type
    )


def _ray_available() -> bool:
    """Check if Ray is initialized or can be initialized."""
    try:
        import ray

        if ray.is_initialized():
            return True
        # Don't auto-initialize Ray; let the caller decide
        return False
    except ImportError:
        return False


def _process_sequential(
    files: Sequence[Path],
    vocab_dict: dict,
    schema: IRSchema,
    window_size: int,
    stride: int,
    verbose: bool,
    include_attack_type: bool = True,
) -> list[Data]:
    """Sequential fallback: process files one at a time."""
    all_graphs: list[Data] = []

    for i, f in enumerate(files):
        if verbose or i % 10 == 0:
            log.info("Processing file %d/%d: %s", i + 1, len(files), f.name)

        try:
            graphs = _process_single_file(
                str(f),
                vocab_dict,
                schema,
                window_size,
                stride,
                include_attack_type,
            )
            all_graphs.extend(graphs)
            log.info("  Created %d graphs (%d total)", len(graphs), len(all_graphs))
        except Exception as e:
            log.warning("Error processing %s: %s", f, e)

    return all_graphs


def _process_ray(
    files: Sequence[Path],
    vocab_dict: dict,
    schema: IRSchema,
    window_size: int,
    stride: int,
    verbose: bool,
    include_attack_type: bool = True,
) -> list[Data]:
    """Ray parallel processing: one task per file."""
    import ray

    # Put shared data in object store once
    vocab_ref = ray.put(vocab_dict)

    @ray.remote
    def _remote_process(file_path: str, v_ref, ws: int, st: int, nf: int, iat: bool):
        """Ray remote task wrapping _process_single_file."""
        schema_local = IRSchema(num_features=nf, include_attack_type=iat)
        v = ray.get(v_ref) if not isinstance(v_ref, dict) else v_ref
        return _process_single_file(file_path, v, schema_local, ws, st, iat)

    log.info("Submitting %d files to Ray...", len(files))
    futures = [
        _remote_process.remote(
            str(f),
            vocab_ref,
            window_size,
            stride,
            schema.num_features,
            include_attack_type,
        )
        for f in files
    ]

    all_graphs: list[Data] = []
    for i, future in enumerate(futures):
        try:
            graphs = ray.get(future)
            all_graphs.extend(graphs)
            if verbose or (i + 1) % 10 == 0:
                log.info(
                    "Completed file %d/%d: %d graphs (%d total)",
                    i + 1,
                    len(files),
                    len(graphs),
                    len(all_graphs),
                )
        except Exception as e:
            log.warning("Ray task %d failed: %s", i, e)

    return all_graphs
