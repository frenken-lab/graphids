"""Parallel preprocessing driver using Ray for per-file fan-out.

Each file is processed independently via ``@ray.remote``, making this
embarrassingly parallel. Falls back to sequential processing when Ray
is not available or when running in local mode with few files.

The adapter is now a parameter (not hardcoded), resolved by the caller
(typically PreprocessingPipeline).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from graphids.config.schema import PreprocessingConfig

from ._engine import GraphEngine
from ._schema import IRSchema
from ._vocabulary import EntityVocabulary
from .adapters.base import DomainAdapter

if TYPE_CHECKING:
    from torch_geometric.data import Data

log = logging.getLogger(__name__)

_PREP_DEFAULTS = PreprocessingConfig()


def _process_single_file(
    file_path: str,
    vocab_dict: dict,
    schema: IRSchema,
    window_size: int,
    stride: int,
    adapter_cls: type | None = None,
    adapter_kwargs: dict | None = None,
) -> list[Data]:
    """Process one CSV file -> list of PyG Data objects.

    This function is the unit of work for both sequential and Ray execution.
    It receives a plain dict (not EntityVocabulary) to avoid pickling issues.
    For Ray, adapter_cls + adapter_kwargs are used to reconstruct the adapter.
    """
    vocab = EntityVocabulary(vocab_dict)

    if adapter_cls is not None:
        adapter = adapter_cls(**(adapter_kwargs or {}))
    else:
        from .adapters._can_bus import CANBusAdapter

        adapter = CANBusAdapter(include_attack_type=True)

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
    window_size: int = _PREP_DEFAULTS.window_size,
    stride: int = _PREP_DEFAULTS.stride,
    return_vocab: bool = False,
    verbose: bool = False,
    adapter: DomainAdapter | None = None,
) -> list[Data] | tuple[list[Data], dict]:
    """Process a dataset directory into PyG graphs.

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
    adapter : DomainAdapter, optional
        Domain adapter to use. Defaults to CANBusAdapter.

    Returns
    -------
    list[Data] or (list[Data], dict)
        Graphs, and optionally the vocabulary dict for caching.
    """
    if adapter is None:
        from .adapters._can_bus import CANBusAdapter

        adapter = CANBusAdapter(include_attack_type=True)

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
    ray_threshold = _PREP_DEFAULTS.ray_file_threshold
    adapter_cls = type(adapter)
    adapter_kwargs = adapter.to_init_kwargs()

    all_graphs = _dispatch(
        files=files,
        vocab_dict=vocab_dict,
        schema=adapter.schema,
        window_size=window_size,
        stride=stride,
        verbose=verbose,
        adapter_cls=adapter_cls,
        adapter_kwargs=adapter_kwargs,
        ray_threshold=ray_threshold,
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
    adapter_cls: type,
    adapter_kwargs: dict,
    ray_threshold: int = 4,
) -> list[Data]:
    """Choose between Ray parallel and sequential processing."""
    use_ray = len(files) >= ray_threshold and _ray_available()

    if use_ray:
        return _process_ray(
            files,
            vocab_dict,
            schema,
            window_size,
            stride,
            verbose,
            adapter_cls,
            adapter_kwargs,
        )
    return _process_sequential(
        files,
        vocab_dict,
        schema,
        window_size,
        stride,
        verbose,
        adapter_cls,
        adapter_kwargs,
    )


def _ray_available() -> bool:
    """Check if Ray is initialized or can be initialized."""
    try:
        import ray

        if ray.is_initialized():
            return True
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
    adapter_cls: type,
    adapter_kwargs: dict,
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
                adapter_cls,
                adapter_kwargs,
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
    adapter_cls: type,
    adapter_kwargs: dict,
) -> list[Data]:
    """Ray parallel processing: one task per file."""
    import ray

    # Put shared data in object store once
    vocab_ref = ray.put(vocab_dict)

    @ray.remote
    def _remote_process(file_path: str, v_ref, ws: int, st: int, nf: int, iat: bool, a_cls, a_kw):
        """Ray remote task wrapping _process_single_file."""
        schema_local = IRSchema(num_features=nf, include_attack_type=iat)
        v = ray.get(v_ref) if not isinstance(v_ref, dict) else v_ref
        return _process_single_file(file_path, v, schema_local, ws, st, a_cls, a_kw)

    log.info("Submitting %d files to Ray...", len(files))
    futures = [
        _remote_process.remote(
            str(f),
            vocab_ref,
            window_size,
            stride,
            schema.num_features,
            schema.include_attack_type,
            adapter_cls,
            adapter_kwargs,
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
