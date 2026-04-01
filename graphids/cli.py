"""Config resolution and Lightning dispatch. No torch at import time.

Dev path:      run_lightning() → lazy-import _lightning.py → GraphIDSCLI
Pipeline path: resolve_configs() → direct instantiation (train_entrypoint.py)
"""

from __future__ import annotations

from typing import Any

from graphids.config.yaml_utils import merge_yaml_chain


def resolve_configs(
    config_files: tuple[str, ...] | list[str],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge YAML config chain + dotted overrides. No torch dependency."""
    return merge_yaml_chain(config_files, overrides)


def run_lightning(args: list[str]) -> None:
    """Lazy-import LightningCLI and execute. Torch loaded here, not at import."""
    from graphids._lightning import CLI_KWARGS, GraphIDSCLI

    GraphIDSCLI(**CLI_KWARGS, args=args)
