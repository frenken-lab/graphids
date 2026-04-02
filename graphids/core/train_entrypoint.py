"""Training entrypoint for pipeline runs.

Builds CLI args from a TrainingSpec, writes a config snapshot for
reproducibility, then delegates to LightningCLI via run_lightning().
No shadow instantiation — LightningCLI handles link_arguments, forced
callbacks, path patching, and class instantiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphids.cli import resolve_configs, run_lightning
from graphids.config.yaml_utils import write_yaml
from graphids.core.contracts import TrainingContract, TrainingSpec


def _build_cli_args(spec: TrainingSpec) -> list[str]:
    """Convert TrainingSpec to LightningCLI args list."""
    overrides = TrainingContract.to_override_dict(spec)
    args = ["fit"]
    for cf in spec.config_files:
        args.extend(["--config", cf])
    for key, val in overrides.items():
        args.append(f"--{key}={val}")
    return args


def run_training_from_spec(spec: TrainingSpec) -> None:
    """Resolve config chain, write snapshot, run training via LightningCLI."""
    overrides = TrainingContract.to_override_dict(spec)
    resolved = resolve_configs(spec.config_files, overrides)

    # Snapshot for reproducibility (written before training starts)
    rd = Path(spec.run_dir)
    if str(rd):
        rd.mkdir(parents=True, exist_ok=True)
        write_yaml(resolved, rd / "config_snapshot.yaml")

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    run_lightning(_build_cli_args(spec))


def run_training_from_payload(payload: dict[str, Any]) -> None:
    run_training_from_spec(TrainingContract.from_envelope(payload))


def run_test_from_spec(spec: TrainingSpec) -> None:
    """Run LightningCLI test using best checkpoint from a completed training run."""
    import structlog

    log = structlog.get_logger()

    ckpt = Path(spec.run_dir) / "checkpoints" / "best_model.ckpt"
    if not ckpt.exists():
        log.warning("no_checkpoint_for_test", run_dir=spec.run_dir)
        return

    overrides = TrainingContract.to_override_dict(spec)
    args = ["test"]
    for cf in spec.config_files:
        args.extend(["--config", cf])
    for key, val in overrides.items():
        args.append(f"--{key}={val}")
    args.append(f"--ckpt_path={ckpt}")

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    run_lightning(args)


def run_test_from_payload(payload: dict[str, Any]) -> None:
    run_test_from_spec(TrainingContract.from_envelope(payload))
