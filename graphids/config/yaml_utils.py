"""YAML loading, merging, and writing helpers for config modules.

No torch/Lightning dependencies — safe for login nodes and dagster workers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML content in {path}, got {type(loaded).__name__}")
    return loaded


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge. Overlay values win. Returns new dict."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_dotted_overrides(merged: dict, overrides: dict[str, Any]) -> dict:
    """Apply dotted-key overrides (e.g. ``"trainer.max_epochs": "2"``) into nested dict."""
    merged = dict(merged)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = merged
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return merged


def merge_yaml_chain(
    config_files: tuple[str, ...] | list[str],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge a YAML config chain and apply dotted overrides. No torch dependency."""
    merged: dict[str, Any] = {}
    for path_str in config_files:
        p = Path(path_str)
        merged = deep_merge(merged, read_yaml(p))  # raises FileNotFoundError on typos
    if overrides:
        merged = apply_dotted_overrides(merged, overrides)
    return merged


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write dict as YAML atomically (NFS-safe: temp file → fsync → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
