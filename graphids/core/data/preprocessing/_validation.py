"""Small validation helpers for preprocessing config primitives."""

from __future__ import annotations


def require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
