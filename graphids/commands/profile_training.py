"""Profile training — run a short profiled fit with PyTorchProfiler overlay.

Usage:
    python -m graphids profile-training                          # defaults: autoencoder small_vgae hcrl_ch
    python -m graphids profile-training normal small_gat set_01
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from graphids.config import CONFIG_DIR


def run_profile_training(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile a training stage")
    parser.add_argument("stage", nargs="?", default="autoencoder")
    parser.add_argument("scale", nargs="?", default="small_vgae")
    parser.add_argument("dataset", nargs="?", default="hcrl_ch")
    args = parser.parse_args(argv)

    stages_dir = CONFIG_DIR / "stages"
    overlays_dir = CONFIG_DIR / "overlays"

    cmd = [
        sys.executable, "-m", "graphids", "fit",
        f"--config={stages_dir / f'{args.stage}.yaml'}",
        f"--config={overlays_dir / f'{args.scale}.yaml'}",
        f"--config={overlays_dir / 'profile.yaml'}",
        f"--data.init_args.dataset={args.dataset}",
    ]
    sys.exit(subprocess.run(cmd).returncode)


main = run_profile_training
