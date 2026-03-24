"""Programmatic experiment manifest builder — generates YAML with only overrides."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import yaml


class ManifestBuilder:
    """Build experiment manifests with sweep dimensions, defaults, and named configs.

    Args:
        sweep: Iteration dimensions expanded as Cartesian product (e.g. dataset, seed).
        defaults: Default config values (Hydra dotlist keys). Include ``stages`` for the
            default stage list.
        expand: Optional shorthand expansions. Maps a short key to multiple dotlist keys
            that all receive the same value (e.g. ``{"conv_type": ["vgae.conv_type", "gat.conv_type"]}``).
    """

    def __init__(
        self,
        sweep: dict[str, list[Any]],
        defaults: dict[str, Any],
        *,
        expand: dict[str, list[str]] | None = None,
    ):
        self.sweep = sweep
        self.defaults = dict(defaults)
        self._expand = expand or {}
        self._configs: dict[str, dict[str, Any]] = {}

    def _apply_expand(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Expand shorthand keys into their dotlist targets."""
        result: dict[str, Any] = {}
        for k, v in overrides.items():
            if k in self._expand:
                for target in self._expand[k]:
                    result[target] = v
            else:
                result[k] = v
        return result

    def add(self, name: str, **overrides: Any) -> None:
        """Add a named config with only-override keys (diff from defaults)."""
        expanded = self._apply_expand(overrides)
        expanded_defaults = self._apply_expand(self.defaults)
        self._configs[name] = {
            k: v for k, v in expanded.items() if v != expanded_defaults.get(k)
        }

    def factorial(self, name_prefix: str, **axes: Any) -> None:
        """Add Cartesian product of parameter axes."""
        keys = list(axes.keys())
        values = [v if isinstance(v, (list, tuple)) else [v] for v in axes.values()]
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            varying = [
                str(v)
                for k, v in zip(keys, combo)
                if isinstance(axes[k], (list, tuple)) and len(axes[k]) > 1
            ] or [str(v) for v in combo]
            self.add(f"{name_prefix}_{'_'.join(varying)}", **overrides)

    def sweep_axis(self, name_prefix: str, **overrides: Any) -> None:
        """Sweep over a single list-valued axis, keeping others fixed."""
        sweep_key = next(
            (k for k, v in overrides.items() if isinstance(v, (list, tuple))), None
        )
        if sweep_key is None:
            self.add(name_prefix, **overrides)
            return
        sweep_vals = overrides.pop(sweep_key)
        for val in sweep_vals:
            self.add(f"{name_prefix}_{val}", **{sweep_key: val, **overrides})

    def write(self, path: str | Path) -> None:
        """Write manifest YAML with sweep + defaults + configs sections."""
        doc = {
            "sweep": self.sweep,
            "defaults": self._apply_expand(self.defaults),
            "configs": self._configs,
        }
        Path(path).write_text(
            yaml.dump(doc, default_flow_style=False, sort_keys=False)
        )
