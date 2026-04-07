"""Shared orchestration data types.

``StageConfig`` is the planner's per-asset value — the input to
``ConfigResolver.resolve`` and the shape every asset materialization
passes to SLURM submission. Pure data, no dagster dependency, no
torch/Lightning imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageConfig:
    """Training config for one asset. Pure data, no dagster dependency."""

    asset_name: str
    stage: str
    model_type: str
    scale: str
    jsonnet_path: str = ""
    model_init_overrides: dict[str, Any] = field(default_factory=dict)
    identity: str = ""
    kd_tag: str = ""
    resource_model: str = ""  # model key for resource lookup (fusion method for fusion stages)
    kd_overrides: dict[str, Any] = field(default_factory=dict)  # raw KDEntry payload
    trainer_overrides: dict[str, Any] = field(default_factory=dict)
    stage_overrides: dict[str, Any] = field(default_factory=dict)
    resource_overrides: dict[str, str | int] = field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (Monarch endpoint args must be serializable)."""
        import dataclasses as _dc

        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageConfig:
        """Reconstruct from a dict (inverse of ``to_dict``).

        ``dataclasses.asdict`` converts tuples to lists — coerce back.
        """
        d = dict(d)
        if "upstream_asset_names" in d:
            d["upstream_asset_names"] = tuple(d["upstream_asset_names"])
        return cls(**d)
