"""CLI entry point: single stage runs via Hydra, DAG submission via manifest.

Usage:
    python -m graphids stage=autoencoder model_type=vgae scale=large dataset=hcrl_sa
    python -m graphids --multirun stage=autoencoder model_type=vgae scale=large training.lr=0.001,0.01
    python -m graphids manifest ablation.yaml --dry-run
    python -m graphids manifest ablation.yaml --filter baseline_bandit
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.multiprocessing as mp
## TODO: This may be obsolete as spawn does work well with IPC inter-process communication
mp.set_start_method("spawn", force=True)
# Use file-based IPC instead of /dev/shm mmap. OSC SLURM nodes restrict
# /dev/shm and vm.max_map_count (65530), causing OOM with large datasets
# (e.g. 700K graphs × 6 tensors × N workers exceeds mmap limits).
mp.set_sharing_strategy("file_system")


from graphids.logging import configure_logging as _configure_logging


def main(argv: list[str] | None = None) -> None:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf

    from graphids.config import CONFIG_DIR, STAGES, Config
    from graphids.pipeline.stages import run_stage

    # Register structured config as schema — Hydra validates types on compose
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)

    args = argv if argv is not None else sys.argv[1:]

    json_logs = "--json-logs" in args
    if json_logs:
        args = [a for a in args if a != "--json-logs"]
    _configure_logging(json=json_logs or None)

    # Extract Hydra overrides (key=value args, not --flags) for preset merge
    cli_overrides = [a for a in args if "=" in a and not a.startswith("-")]

    sys.argv = [sys.argv[0]] + args

    @hydra.main(config_path="config", config_name="config", version_base="1.3")
    def run(cfg: DictConfig) -> float | None:
        # Merge model preset; re-apply CLI overrides so they win
        models = OmegaConf.load(CONFIG_DIR / "models.yaml")
        preset = models.get(f"{cfg.model_type}_{cfg.scale}")
        if preset:
            cfg = OmegaConf.merge(cfg, preset, OmegaConf.from_dotlist(cli_overrides))

        stage = cfg.get("stage")
        if not stage or stage not in STAGES:
            raise SystemExit(f"stage= required. Valid: {list(STAGES.keys())}")

        result = run_stage(cfg, stage)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        return metrics.get("val_loss", float("inf"))

    run()


def manifest(argv: list[str] | None = None) -> None:
    """Submit experiment manifest to SLURM."""
    import argparse

    from graphids.pipeline.manifest import submit_manifest

    parser = argparse.ArgumentParser(description="Submit experiment manifest to SLURM")
    parser.add_argument("manifest", type=Path, help="Path to manifest YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    parser.add_argument("--filter", nargs="*", help="Only run these config names")
    args = parser.parse_args(argv)

    futures = submit_manifest(args.manifest, dry_run=args.dry_run, filter_configs=args.filter)
    if futures:
        print(f"Submitted {len(futures)} jobs")


if __name__ == "__main__":
    if sys.argv[1:2] == ["manifest"]:
        manifest(sys.argv[2:])
    else:
        main()
