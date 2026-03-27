"""Experiment presets — recipes that return configured Manifest instances."""
from __future__ import annotations

from pathlib import Path

import yaml

from .manifest import Manifest

_CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "defaults" / "datasets.yaml"
ALL_DATASETS: list[str] = [
    k for k in yaml.safe_load(_CATALOG_PATH.read_text()) if not k.startswith("_")
]
ALL_SEEDS = [42, 123, 456]
ALL_SCALES = ["large", "small"]


def ablation() -> Manifest:
    """Paper ablation: 18 configs × 2 datasets × 1 seed."""
    m = Manifest(
        sweep={"dataset": ["set_01", "set_02"], "seed": [42]},
        defaults={
            "stages": ["autoencoder", "curriculum", "fusion", "evaluation"],
            "scale": "small",
            "training.loss_fn": "focal",
            "fusion.method": "bandit",
        },
        expand={"conv_type": ["vgae.conv_type", "gat.conv_type"]},
    )

    # Claim 4: Loss × Curriculum (3 losses × 2 training modes = 6)
    for loss in ["ce", "focal", "weighted_ce"]:
        m.add(
            f"loss_x_curriculum_{loss}_curriculum",
            **{"training.loss_fn": loss, "fusion.method": "weighted_avg"},
        )
        m.add(
            f"loss_x_curriculum_{loss}_normal",
            **{
                "training.loss_fn": loss,
                "fusion.method": "weighted_avg",
                "stages": ["autoencoder", "normal", "fusion", "evaluation"],
                "gat_stage": "normal",
            },
        )

    # Claim 2: Fusion method (4)
    m.sweep_axis("fusion", **{"fusion.method": ["bandit", "dqn", "mlp", "weighted_avg"]})

    # Claim 5: Conv type (3)
    m.add("conv_gatv2")
    m.add("conv_gatv1", conv_type="gat")
    m.add("conv_gps", conv_type="gps")

    # Claim 6: Unsupervised method (3)
    m.add("unsup_vgae")
    m.add("unsup_gae", **{"vgae.variational": False})
    m.add("unsup_dgi", model_type="dgi", stages=["autoencoder", "normal", "evaluation"], gat_stage="normal")

    # Claim 1: Single-model baselines
    m.add("vgae_only", stages=["autoencoder", "evaluation"])
    m.add("gat_only", stages=["normal", "evaluation"], gat_stage="normal")

    return m


def pipeline(
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    scales: list[str] | None = None,
) -> Manifest:
    """Full pipeline sweep."""
    m = Manifest(
        sweep={"dataset": datasets or ALL_DATASETS, "seed": seeds or ALL_SEEDS},
        defaults={
            "stages": ["autoencoder", "curriculum", "fusion", "evaluation"],
            "scale": "small",
            "training.loss_fn": "focal",
            "fusion.method": "bandit",
        },
    )
    for scale in (scales or ALL_SCALES):
        m.add(scale, scale=scale)
    return m


PRESETS: dict[str, dict] = {
    "ablation": {"build_fn": ablation, "default_out": "ablation.yaml"},
    "pipeline": {"build_fn": pipeline, "default_out": "pipeline.yaml"},
}
