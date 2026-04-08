"""Reusable Pydantic validator factories for set-membership checks."""

from __future__ import annotations

from collections.abc import Callable


def check_in(valid: frozenset[str], label: str) -> Callable[[str], str]:
    """``AfterValidator`` factory: reject values not in *valid*."""

    def _validator(v: str) -> str:
        if v not in valid:
            raise ValueError(f"{label}={v!r} not in {sorted(valid)}")
        return v

    return _validator


def check_all_in(valid: dict | frozenset, label: str) -> Callable[[list[str]], list[str]]:
    """``AfterValidator`` factory: reject any element not in *valid*."""

    def _validator(v: list[str]) -> list[str]:
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"Unknown {label}(s): {bad}. Valid: {sorted(valid)}")
        return v

    return _validator
