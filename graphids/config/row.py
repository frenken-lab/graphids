"""Row builders.

A composer returns a :class:`RowSpec` carrying the rendered training
config plus out-of-band identity bits (``meta``, ``resources``,
``upstreams``). Calling ``.fit(name)`` / ``.test(name)`` emits a dict
with the shape consumed by :class:`graphids.configs.blueprint.TrainRow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphids.config.catalog import run_dir as _run_dir
from graphids.graphids.config.blueprint import RenderedConfig


@dataclass(frozen=True)
class RowSpec:
    """Composer output. Not a row yet — call ``.fit()`` / ``.test()``.

    ``rendered`` is a frozen :class:`graphids.configs.blueprint.RenderedConfig` —
    typo'd field access (``spec.rendered.trianer``) raises
    :class:`AttributeError`, and constructing one with an unknown key
    raises :class:`pydantic.ValidationError` (``extra="forbid"``).
    """

    rendered: RenderedConfig
    meta: dict[str, Any]
    resources: dict[str, Any]
    upstreams: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        required = {"group", "variant", "dataset", "seed", "model_type", "scale"}
        missing = required - set(self.meta)
        if missing:
            raise ValueError(f"RowSpec.meta missing keys: {sorted(missing)}")
        mode = self.resources.get("mode")
        if mode not in {"gpu", "cpu"}:
            raise ValueError(f"RowSpec.resources.mode must be 'gpu'|'cpu', got {mode!r}")

    def fit(self, name: str, *, length: str = "long") -> dict[str, Any]:
        return _emit("fit", name, self, length)

    def test(self, name: str, *, length: str = "long") -> dict[str, Any]:
        return _emit("test", name + "-test", self, length)


def extract(
    *,
    name: str,
    dataset: str,
    extractor_ckpts: dict[str, str],
    output_dir: str,
    mode: str = "gpu",
    length: str = "short",
    max_samples: int = 150_000,
    max_val_samples: int = 30_000,
    batch_size: int = 256,
    seed: int = 42,
    window_size: int = 100,
    stride: int = 100,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """One-shot fusion-feature extraction row — port of ``row.libsonnet:extract``."""
    return {
        "name": name,
        "action": "extract",
        "dataset": dataset,
        "extractor_ckpts": dict(extractor_ckpts),
        "output_dir": output_dir,
        "resources": {"mode": mode, "length": length},
        "max_samples": max_samples,
        "max_val_samples": max_val_samples,
        "batch_size": batch_size,
        "seed": seed,
        "window_size": window_size,
        "stride": stride,
        "val_fraction": val_fraction,
    }


def _accelerator_for(mode: str) -> str:
    return "cpu" if mode == "cpu" else "auto"


def _emit(action: str, name: str, spec: RowSpec, length: str) -> dict[str, Any]:
    m = spec.meta
    rendered = spec.rendered.model_dump()
    rendered["trainer"]["accelerator"] = _accelerator_for(spec.resources["mode"])
    return {
        "name": name,
        "action": action,
        "identity": {
            "run_name": f"{m['group']}_{m['variant']}_{m['dataset']}_seed{m['seed']}",
            "run_dir": _run_dir(m["dataset"], m["group"], m["variant"], m["seed"]),
            "jobname": f"{m['model_type']}-{m['scale']}-{m['variant']}",
        },
        "meta": dict(m),
        "rendered_config": rendered,
        "upstreams": list(spec.upstreams),
        "resources": {"mode": spec.resources["mode"], "length": length},
    }
