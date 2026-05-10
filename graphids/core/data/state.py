"""Process-level dataset cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DatasetState:
    """Ready-to-serve train/val/test splits."""

    train: Any
    val: Any
    test: dict[str, Any]


class _CacheableDataset(Protocol):
    cache_key: str

    def build(self) -> DatasetState: ...


_REGISTRY: dict[str, DatasetState] = {}


def get_or_build(dataset: _CacheableDataset) -> DatasetState:
    """Return cached ``DatasetState`` for ``dataset``."""
    key = dataset.cache_key
    state = _REGISTRY.get(key)
    if state is None:
        state = dataset.build()
        _REGISTRY[key] = state
    return state


def clear_cache() -> None:
    """Drop all cached states. Intended for test teardown."""
    _REGISTRY.clear()
