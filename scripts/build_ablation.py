#!/usr/bin/env python3
"""Build ablation.yaml manifest for the KD-GAT paper.

Usage:
    python scripts/build_ablation.py                # writes ablation.yaml
    python scripts/build_ablation.py --out my.yaml  # custom output path
    python scripts/build_ablation.py --dry-run      # build + immediately dry-run the DAG
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from graphids.config.manifest_builder import ManifestBuilder


def build() -> ManifestBuilder:
    b = ManifestBuilder(
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
        b.add(
            f"loss_x_curriculum_{loss}_curriculum",
            **{"training.loss_fn": loss, "fusion.method": "weighted_avg"},
        )
        b.add(
            f"loss_x_curriculum_{loss}_normal",
            **{
                "training.loss_fn": loss,
                "fusion.method": "weighted_avg",
                "stages": ["autoencoder", "normal", "fusion", "evaluation"],
                "gat_stage": "normal",
            },
        )

    # Claim 2: Fusion method (4)
    b.sweep_axis("fusion", **{"fusion.method": ["bandit", "dqn", "mlp", "weighted_avg"]})

    # Claim 3: KD & scale (deferred — requires small_kd preset + teacher wiring)
    # b.add("kd_student", scale="small_kd")
    # b.add("large_reference", scale="large")

    # Claim 5: Conv type (3)
    b.add("conv_gatv2")
    b.add("conv_gatv1", conv_type="gat")
    b.add("conv_gps", conv_type="gps", **{"training.batch_size": 256})

    # Claim 6: Unsupervised method (3)
    b.add("unsup_vgae")
    b.add("unsup_gae", **{"vgae.variational": False})
    b.add(
        "unsup_dgi",
        model_type="dgi",
        stages=["autoencoder", "normal", "evaluation"],
        gat_stage="normal",
    )

    # Claim 1: Single-model baselines
    b.add("vgae_only", stages=["autoencoder", "evaluation"])
    b.add("gat_only", stages=["normal", "evaluation"], gat_stage="normal")

    return b


def main():
    parser = argparse.ArgumentParser(description="Build ablation manifest")
    parser.add_argument("--out", default="ablation.yaml", help="Output path (default: ablation.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Build then dry-run the DAG")
    args = parser.parse_args()

    b = build()
    out = Path(args.out)
    b.write(out)
    print(f"Wrote {len(b._configs)} configs to {out}")

    if args.dry_run:
        subprocess.run(
            [sys.executable, "-m", "graphids.pipeline.orchestration.manifest", str(out), "--dry-run"],
            check=True,
        )


if __name__ == "__main__":
    main()
