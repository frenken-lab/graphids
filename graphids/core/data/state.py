"""Process-level dataset cache.

``get_or_build`` memoizes the expensive ``Dataset.build()`` call keyed by
``dataset.cache_key`` so subsequent stages sharing a Python process hit
the in-memory state instead of remmapping torch tensors.

The cache is intentionally dumb: duck-types the ``Dataset`` protocol
(``cache_key: str`` + ``build() -> DatasetState``), lives in a module
dict, and dies with the process. No preprocessing knowledge, no disk
I/O, no PyG imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DatasetState:
    """Result of ``Dataset.build()`` — ready-to-serve train/val/test splits."""

    train: Any
    val: Any
    test: dict[str, Any]


class _CacheableDataset(Protocol):
    cache_key: str

    def build(self) -> DatasetState: ...


_REGISTRY: dict[str, DatasetState] = {}


def get_or_build(dataset: _CacheableDataset) -> DatasetState:
    """Return cached ``DatasetState`` for ``dataset``, building on first call."""
    key = dataset.cache_key
    state = _REGISTRY.get(key)
    if state is None:
        state = dataset.build()
        _REGISTRY[key] = state
    return state


def clear_cache() -> None:
    """Drop all cached states. Intended for test teardown."""
    _REGISTRY.clear()
