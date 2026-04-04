"""Lightning dispatch + shared wiring constants. No torch at import time.

Dev path:      run_lightning() → lazy-import _lightning.py → GraphIDSCLI
Pipeline path: merge_yaml_chain() → direct instantiation (train_entrypoint.py)

Shared wiring constants (LINK_TARGETS, CHECKPOINT_DEFAULTS, EARLY_STOPPING_DEFAULTS)
live here so both paths consume a single source of truth.
"""

from __future__ import annotations

from typing import Any

# --- Shared wiring: single source of truth for dev + pipeline paths ---

LINK_TARGETS: list[tuple[str, str]] = [
    ("data.init_args.dataset", "model.init_args.dataset"),
    ("data.init_args.lake_root", "model.init_args.lake_root"),
    ("seed_everything", "model.init_args.seed"),
    ("seed_everything", "data.init_args.seed"),
    ("model.init_args.conv_type", "data.init_args.conv_type"),
    ("model.init_args.heads", "data.init_args.heads"),
]

CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "monitor": "val_loss",
    "mode": "min",
    "save_top_k": 1,
    "save_last": True,
    "filename": "best_model",
}

EARLY_STOPPING_DEFAULTS: dict[str, Any] = {
    "monitor": "val_loss",
    "patience": 100,
    "mode": "min",
}


def run_lightning(args: list[str]) -> None:
    """Lazy-import LightningCLI and execute. Torch loaded here, not at import."""
    from graphids._lightning import CLI_KWARGS, GraphIDSCLI

    GraphIDSCLI(**CLI_KWARGS, args=args)
