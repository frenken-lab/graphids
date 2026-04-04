"""Training entrypoint for pipeline runs.

Resolves the YAML chain, parses it through ``graphids._lightning.build_cli``
(which runs ``parse_object``, ``before_instantiate_classes`` path patching,
and trainer/model/data construction with forced callbacks), then invokes
``trainer.fit`` / ``trainer.test``. No CLI string round-trip.
"""

from __future__ import annotations

from pathlib import Path

from graphids.log import get_logger

from graphids.config import LAST_CKPT_SUBPATH
from graphids.config.yaml_utils import merge_yaml_chain, write_yaml
from graphids.core.contracts import TrainingContract, TrainingSpec

log = get_logger(__name__)


def _instantiate_from_spec(spec: TrainingSpec):
    """Merge YAML chain, snapshot, return a fully instantiated ``GraphIDSCLI``."""
    from graphids._lightning import build_cli  # lazy torch import

    merged = merge_yaml_chain(spec.config_files, TrainingContract.to_override_dict(spec))
    rd = Path(spec.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    write_yaml(merged, rd / "config_snapshot.yaml")
    return build_cli(merged)


def run_training_from_spec(spec: TrainingSpec) -> None:
    """Resolve config chain, snapshot, instantiate, run ``trainer.fit``."""
    # Auto-resume: check for last.ckpt on THIS node (not the orchestrator).
    if "ckpt_path" not in spec.runtime_overrides:
        last_ckpt = Path(spec.run_dir) / LAST_CKPT_SUBPATH
        if last_ckpt.exists():
            spec = spec.model_copy(update={
                "runtime_overrides": {**spec.runtime_overrides, "ckpt_path": str(last_ckpt)},
            })
            log.info("auto_resume", ckpt=str(last_ckpt))

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    cli = _instantiate_from_spec(spec)
    cli.trainer.fit(
        cli.model, datamodule=cli.datamodule,
        ckpt_path=spec.runtime_overrides.get("ckpt_path"),
    )


def run_test_from_spec(spec: TrainingSpec) -> None:
    """Run LightningCLI test using best available checkpoint from a completed training run."""
    from graphids.config import CKPT_SUBPATH

    run_dir = Path(spec.run_dir)
    ckpt = run_dir / CKPT_SUBPATH
    if not ckpt.exists():
        ckpt = run_dir / LAST_CKPT_SUBPATH
        if not ckpt.exists():
            log.warning("no_checkpoint_for_test", run_dir=spec.run_dir)
            return
        log.info("using_last_checkpoint", ckpt=str(ckpt))

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    cli = _instantiate_from_spec(spec)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=str(ckpt))
