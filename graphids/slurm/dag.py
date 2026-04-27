"""``Node`` model + plan-jsonnet parser + topological sort.

A *plan* is a jsonnet file declaring ``{ nodes: [...] }``. Each entry is
one ``graphids submit`` call. :func:`parse_plan` validates via pydantic
and returns ``tuple[Node, ...]``; :func:`toposort` orders by deps.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Node(BaseModel):
    """One ``graphids submit`` line. Preset XOR command.

    For preset nodes, ``group`` / ``variant`` are derived from the preset
    path's ``<group>/<variant>.jsonnet`` convention. Plans need only declare
    ``preset:`` per node; explicit ``group`` / ``variant`` override the
    inference (rarely needed — for off-convention paths).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str
    deps: tuple[str, ...] = ()
    # Preset mode (`preset` is the on-disk plan-jsonnet field name):
    preset_path: str | None = Field(default=None, alias="preset")
    action: Literal["fit", "test"] = "fit"
    group: str | None = None
    variant: str | None = None
    # Command mode:
    command: str | None = None
    # Resource overrides (None → profile default):
    mode: Literal["gpu", "cpu"] = "gpu"
    length: Literal["short", "long"] = "long"
    timeout_min: int | None = None
    mem_gb: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_group_variant(cls, data: Any) -> Any:
        """Infer ``group`` / ``variant`` from ``<group>/<variant>.jsonnet`` preset path.

        Plan jsonnet only needs to declare ``preset``; the post-validator
        below catches off-convention paths that fail inference.
        """
        if not isinstance(data, dict):
            return data
        preset = data.get("preset_path") or data.get("preset")
        if not preset or "group" in data or "variant" in data:
            return data
        parts = PurePosixPath(preset).parts
        if len(parts) >= 2 and parts[-1].endswith(".jsonnet"):
            data["group"] = parts[-2]
            data["variant"] = parts[-1].removesuffix(".jsonnet")
        return data

    @model_validator(mode="after")
    def _check_preset_xor_command(self) -> Node:
        has_preset = self.preset_path is not None
        has_cmd = self.command is not None
        if has_preset == has_cmd:
            raise ValueError(f"Node {self.name!r}: must have exactly one of preset or command")
        if has_preset and (self.group is None or self.variant is None):
            raise ValueError(
                f"Node {self.name!r}: preset {self.preset_path!r} is off-convention; "
                "expected '<group>/<variant>.jsonnet' or explicit group + variant fields"
            )
        return self

    @property
    def is_command(self) -> bool:
        return self.command is not None


class _Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[Node]


def parse_plan(rendered: dict[str, Any]) -> tuple[Node, ...]:
    """Validate a rendered-jsonnet plan via pydantic. Unknown fields raise."""
    from pydantic import ValidationError

    try:
        return tuple(_Plan.model_validate(rendered).nodes)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def toposort(nodes: tuple[Node, ...]) -> list[Node]:
    """Stable topological sort by dep-names. Raises on cycle or missing dep."""
    from graphlib import TopologicalSorter

    by_name = {n.name: n for n in nodes}
    for n in by_name.values():
        for dep in n.deps:
            if dep not in by_name:
                raise RuntimeError(f"node {n.name!r} deps on unknown {dep!r}")
    ts = TopologicalSorter({n.name: set(n.deps) for n in by_name.values()})
    return [by_name[name] for name in ts.static_order()]


def filter_with_upstream(nodes: tuple[Node, ...], include: tuple[str, ...]) -> tuple[Node, ...]:
    """Return ``include`` plus every transitive upstream dep, topo-sorted.

    Downstream nodes are NOT auto-included. Unknown names raise ``ValueError``.
    """
    by_name = {n.name: n for n in nodes}
    unknown = sorted(set(include) - by_name.keys())
    if unknown:
        raise ValueError(f"unknown nodes {unknown}. Known: {sorted(by_name)}")
    selected: set[str] = set()
    stack = list(include)
    while stack:
        name = stack.pop()
        if name in selected:
            continue
        selected.add(name)
        stack.extend(by_name[name].deps)
    return tuple(toposort(tuple(n for n in nodes if n.name in selected)))
