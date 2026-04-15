"""Orchestrate data types.

Boundary types shared by the CLI and the stage primitives.
``ResolvedConfig`` is the handoff from ``render → validate`` into
``build / train / evaluate``; ``InstantiatedRun`` is the wired
``(trainer, model, datamodule)`` triple produced by ``instantiate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

    from graphids.config.schemas import ValidatedConfig
    from graphids.core.trainer import Trainer


@dataclass(frozen=True)
class ResolvedConfig:
    """Rendered, validated config ready for instantiation.

    Stage primitives (``orchestrate/stage.py``) consume this directly:
    they read ``rendered`` + ``validated`` for building trainer/model,
    ``run_dir`` / ``ckpt_file`` for marker + OTel export wiring, and
    ``stage_name`` for log fields. ``run_dir`` is ``None`` only for
    smoke invocations of the Typer CLI with no ``default_root_dir``
    set — markers and file exporters are skipped in that case.
    """

    rendered: dict[str, Any]
    validated: ValidatedConfig
    stage_name: str
    run_dir: Path | None
    ckpt_file: Path | None

    @classmethod
    def from_rendered(cls, rendered: dict[str, Any], *, stage_name: str) -> ResolvedConfig:
        """Validate a pre-rendered dict and pull ``run_dir`` from jsonnet."""
        from graphids.config.constants import CKPT_SUBPATH
        from graphids.config.schemas import validate_config

        validated = validate_config(rendered)
        default_root = (rendered.get("trainer") or {}).get("default_root_dir") or ""
        run_dir = Path(default_root) if default_root else None
        ckpt_file = run_dir / CKPT_SUBPATH if run_dir else None
        return cls(
            rendered=rendered,
            validated=validated,
            stage_name=stage_name,
            run_dir=run_dir,
            ckpt_file=ckpt_file,
        )


@dataclass
class InstantiatedRun:
    """A wired (trainer, model, datamodule) triple built from a rendered config."""

    trainer: Trainer
    model: nn.Module
    datamodule: Any
