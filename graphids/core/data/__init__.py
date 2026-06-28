"""Data-layer public API.

The runtime datamodules are imported lazily so preprocessing and discovery
can be used without importing optional training dependencies.
"""

from __future__ import annotations

from .datasets import (
    ATTACK_TYPE_CODES,
    ATTACK_TYPE_NAMES,
    CANBusTemporalSource,
)
from .state import DatasetState, clear_cache, get_or_build

__all__ = [
    "CANBusTemporalSource",
    "DatasetState",
    "get_or_build",
    "clear_cache",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
]


def __getattr__(name: str):
    if name == "TemporalDataModule":
        from . import datamodule

        return getattr(datamodule, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"TemporalDataModule"})
