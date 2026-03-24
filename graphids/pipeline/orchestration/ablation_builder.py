"""Programmatic ablation manifest builder — generates YAML with only overrides."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import yaml


class AblationBuilder:
    def __init__(self, datasets: list[str], seeds: list[int], defaults: dict[str, Any]):
        self.datasets = datasets
        self.seeds = seeds
        self.defaults = dict(defaults)
        self._configs: dict[str, dict[str, Any]] = {}  # name -> overrides
        self._resolved: dict[str, dict[str, Any]] = {}  # name -> full merged

    def add(self, name: str, **overrides: Any) -> None:
        self._configs[name] = {k: v for k, v in overrides.items() if v != self.defaults.get(k)}
        self._resolved[name] = {**self.defaults, **overrides}

    def factorial(self, name_prefix: str, **axes: Any) -> None:
        keys = list(axes.keys())
        values = [v if isinstance(v, (list, tuple)) else [v] for v in axes.values()]
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            varying = [str(v) for k, v in zip(keys, combo) if len(axes[k]) > 1] if any(
                isinstance(v, (list, tuple)) and len(v) > 1 for v in axes.values()
            ) else [str(v) for v in combo]
            name = f"{name_prefix}_{'_'.join(varying)}"
            self.add(name, **overrides)

    def sweep(self, name_prefix: str, **overrides: Any) -> None:
        sweep_key = next((k for k, v in overrides.items() if isinstance(v, (list, tuple))), None)
        if sweep_key is None:
            self.add(name_prefix, **overrides)
            return
        sweep_vals = overrides.pop(sweep_key)
        for val in sweep_vals:
            self.add(f"{name_prefix}_{val}", **{sweep_key: val, **overrides})

    def write(self, path: str | Path) -> None:
        doc = {
            "defaults": {"datasets": self.datasets, "seeds": self.seeds, **self.defaults},
            "configs": self._configs,
        }
        Path(path).write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))

    def summary(self) -> None:
        ae_keys, gat_keys, fusion_count = set(), set(), 0
        for r in self._resolved.values():
            ae_key = (r.get("scale"), r.get("conv_type"), r.get("unsupervised", True))
            gat_key = (ae_key, r.get("conv_type"), r.get("loss_fn"), r.get("gat_stage"))
            ae_keys.add(ae_key)
            gat_keys.add(gat_key)
            skip = r.get("skip_stages", [])
            if "fusion" not in (skip if isinstance(skip, list) else [skip]):
                fusion_count += 1
        n = len(self._configs)
        d, s = len(self.datasets), len(self.seeds)
        print(f"Configs: {n}")
        print(f"Unique AE runs: {len(ae_keys)} x {d} datasets x {s} seeds = {len(ae_keys)*d*s}")
        print(f"Unique GAT runs: {len(gat_keys)} x {d} datasets x {s} seeds = {len(gat_keys)*d*s}")
        print(f"Fusion runs: {fusion_count} x {d} datasets x {s} seeds = {fusion_count*d*s}")
        print(f"Estimated total jobs: {(len(ae_keys) + len(gat_keys) + fusion_count)*d*s}")


def _build_paper_plan() -> AblationBuilder:
    """The 17-config ablation plan for the paper.

    Paper narrative: start from vanilla small model, add components, show each helps.
    Baseline (vanilla): small / gatv2 / VGAE / ce / normal / weighted_avg
    Proposed method: small_kd / gatv2 / VGAE / focal / curriculum / bandit
    """
    plan = AblationBuilder(
        datasets=["hcrl_ch", "set_01"],
        seeds=[42],
        defaults=dict(
            scale="small", conv_type="gatv2", unsupervised="vgae",
            loss_fn="focal", gat_stage="curriculum", fusion_method="bandit",
        ),
    )
    # Claim 4: Loss × Curriculum factorial (2×3 = 6 configs)
    plan.factorial("loss_x_curriculum",
        loss_fn=["ce", "focal", "weighted_ce"],
        gat_stage=["curriculum", "normal"],
        fusion_method="weighted_avg",
    )
    # Claim 2: Fusion method sweep (4 configs, share upstream GAT)
    plan.sweep("fusion", fusion_method=["bandit", "dqn", "mlp", "weighted_avg"])
    # Claim 3: KD & scale (2 configs + baseline small is already in factorial)
    plan.add("kd_student", scale="small_kd")
    plan.add("large_reference", scale="large")
    # Claim 5: Conv type (2 configs)
    plan.add("conv_gatv1", conv_type="gat")
    plan.add("conv_gps", conv_type="gps")
    # Claim 6: Unsupervised method (2 configs)
    plan.add("unsup_gae", unsupervised="gae")
    plan.add("unsup_dgi", unsupervised="dgi")
    # Claim 1: Single-model baselines (2 configs)
    plan.add("vgae_only", skip_stages=["curriculum", "fusion"])
    plan.add("gat_only", gat_stage="normal", skip_stages=["autoencoder", "fusion"])
    return plan


if __name__ == "__main__":
    plan = _build_paper_plan()
    out = Path("ablation.yaml")
    plan.write(out)
    print(f"Wrote {out.resolve()}")
    plan.summary()
