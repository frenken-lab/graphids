"""Pipeline actor -- holds TMPDIR staging across stages, sequences training.

Each ``@endpoint`` method wraps the existing instantiate chain unchanged.
Data staging happens once via ``bootstrap_staging`` (passed to
``spawn_procs(bootstrap=...)``). Checkpoint paths thread between stages
via return values.

When monarch is not installed, this module still imports cleanly -- the
``endpoint`` decorator is a no-op passthrough so tests can instantiate the
actor as a plain Python object.
"""

from __future__ import annotations

import importlib
import io
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from graphids.log import get_logger

log = get_logger(__name__)

# Conditional monarch import -- fall back to identity decorator so the
# class is usable as a plain Python object for testing / dry-run.
try:
    from monarch.actor import Actor, endpoint  # type: ignore[import-not-found]
except ImportError:

    class Actor:  # type: ignore[no-redef]
        pass

    def endpoint(fn):  # type: ignore[no-redef]
        return fn


_SPAWN_SET = False


def _ensure_spawn() -> None:
    """Set start method to spawn (critical constraint: CUDA + fork = segfault).

    Needed for PyTorch DataLoader workers, not Monarch process management.
    Uses importlib to satisfy project convention hooks.
    """
    global _SPAWN_SET  # noqa: PLW0603
    if not _SPAWN_SET:
        mp = importlib.import_module("multiprocessing")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        importlib.import_module("torch.multiprocessing").set_sharing_strategy("file_system")
        _SPAWN_SET = True


def _touch(path: Path) -> None:
    """Create a phase marker file with durable fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def bootstrap_staging(dataset: str) -> None:
    """Stage data to TMPDIR and set env vars. Intended for
    ``spawn_procs(bootstrap=lambda: bootstrap_staging("hcrl_ch"))``.

    ``stage_data()`` prints ``export K=V`` lines to stdout for bash eval.
    We capture and apply them to ``os.environ`` directly.
    """
    from graphids.slurm.ops.staging import stage_data

    buf = io.StringIO()
    with redirect_stdout(buf):
        stage_data(dataset=dataset)

    for line in buf.getvalue().splitlines():
        m = re.match(r"^export (\w+)=(.*)$", line)
        if m:
            os.environ[m.group(1)] = m.group(2)


# Stage name -> (model_type, upstream ckpt TLA keys to forward)
_STAGE_META: dict[str, tuple[str, tuple[str, ...]]] = {
    "autoencoder": ("vgae", ()),
    "supervised": ("gat", ("vgae_ckpt_path",)),
    "fusion": ("__fusion__", ("vgae_ckpt_path", "gat_ckpt_path")),
}


class PipelineActor(Actor):
    """Single actor running all pipeline stages in one SLURM allocation.

    Holds cached datasets across stages. Each endpoint renders the
    stage's jsonnet config, instantiates the Lightning stack, and runs
    trainer methods.
    """

    def __init__(
        self,
        dataset: str,
        seed: int,
        scale: str,
        lake_root: str,
        conv_type: str = "gatv2",
        variational: bool = True,
    ) -> None:
        self.dataset = dataset
        self.seed = seed
        self.scale = scale
        self.lake_root = lake_root
        self.conv_type = conv_type
        self.variational = variational
        self._cached_datasets: dict[str, Any] | None = None

    # -- stage preparation (shared by train + eval) ---------------------------

    def _prepare_stage(
        self,
        stage: str,
        fusion_method: str,
        tla_overrides: dict[str, Any] | None,
        vgae_ckpt_path: str | None,
        gat_ckpt_path: str | None,
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Build TLA, resolve model_type, render config.

        Returns (model_type, ckpt_path, run_dir, rendered).
        """
        _ensure_spawn()

        from graphids.config.constants import CKPT_SUBPATH
        from graphids.config.jsonnet import render_config
        from graphids.config.paths import compute_identity_hash
        from graphids.config.schemas import PathContext
        from graphids.orchestrate.contracts import resolve_jsonnet_path

        meta_model, upstream_keys = _STAGE_META[stage]
        model_type = fusion_method if meta_model == "__fusion__" else meta_model

        tla: dict[str, Any] = {
            "dataset": self.dataset,
            "seed": self.seed,
            "scale": self.scale,
            "conv_type": self.conv_type,
            "variational": self.variational,
            **(tla_overrides or {}),
        }
        if stage == "fusion":
            tla["fusion_method"] = fusion_method

        ckpt_map = {"vgae_ckpt_path": vgae_ckpt_path, "gat_ckpt_path": gat_ckpt_path}
        for key in upstream_keys:
            if ckpt_map.get(key):
                tla[key] = ckpt_map[key]

        identity = compute_identity_hash(stage, tla)
        path_ctx = PathContext(
            lake_root=self.lake_root,
            user=os.environ.get("USER", "unknown"),
            dataset=self.dataset,
            model_type=model_type,
            scale=self.scale,
            stage=stage,
            identity=identity,
            kd_tag="",
            seed=self.seed,
        )
        run_dir = str(path_ctx.run_dir)
        tla["run_dir"] = run_dir

        rendered = render_config(resolve_jsonnet_path(stage), tla=tla)
        ckpt_path = str(Path(run_dir) / CKPT_SUBPATH)
        return model_type, ckpt_path, run_dir, rendered

    # -- dataset cache --------------------------------------------------------

    def _instantiate_and_inject(self, rendered: dict[str, Any]) -> Any:
        """Instantiate Lightning stack, injecting cached datasets.

        On first call, datasets load normally and get cached on ``self``.
        Subsequent calls inject cached datasets so ``setup()`` skips reload.
        """
        from graphids.config.schemas import validate_config
        from graphids.instantiate import instantiate

        run = instantiate(rendered, validated=validate_config(rendered))

        if self._cached_datasets is not None:
            run.datamodule._train_ds = self._cached_datasets["train"]
            run.datamodule._val_ds = self._cached_datasets["val"]
            run.datamodule._test_datasets = self._cached_datasets["test"]
        return run

    def _cache_datasets_from(self, datamodule: Any) -> None:
        """Cache datasets from a datamodule after first load."""
        if self._cached_datasets is None and datamodule._train_ds is not None:
            self._cached_datasets = {
                "train": datamodule._train_ds,
                "val": datamodule._val_ds,
                "test": datamodule._test_datasets,
            }
            log.info("datasets_cached", num_train=len(datamodule._train_ds))

    # -- stage endpoints ------------------------------------------------------

    @endpoint
    def train_stage(
        self,
        stage: str,
        tla_overrides: dict[str, Any] | None = None,
        vgae_ckpt_path: str | None = None,
        gat_ckpt_path: str | None = None,
        fusion_method: str = "bandit",
    ) -> str:
        """Train a single stage. Returns checkpoint path.

        Resolves ``model_type`` and upstream checkpoint keys from
        ``_STAGE_META``. Fusion dispatches on ``fusion_method``.
        """
        from graphids.config.constants import PHASE_MARKERS

        _, ckpt_path, run_dir, rendered = self._prepare_stage(
            stage,
            fusion_method,
            tla_overrides,
            vgae_ckpt_path,
            gat_ckpt_path,
        )
        run = self._instantiate_and_inject(rendered)

        log.info("stage_train", stage=stage, run_dir=run_dir)
        run.trainer.fit(run.model, datamodule=run.datamodule)
        self._cache_datasets_from(run.datamodule)
        _touch(Path(run_dir) / PHASE_MARKERS["train"])
        log.info("stage_train_complete", stage=stage, ckpt=ckpt_path)

        return ckpt_path

    @endpoint
    def eval_stage(
        self,
        stage: str,
        tla_overrides: dict[str, Any] | None = None,
        vgae_ckpt_path: str | None = None,
        gat_ckpt_path: str | None = None,
        fusion_method: str = "bandit",
    ) -> None:
        """Run test + analyze + finalize for a completed stage. All lenient."""
        from graphids.config.constants import PHASE_MARKERS

        model_type, ckpt_path, run_dir, rendered = self._prepare_stage(
            stage,
            fusion_method,
            tla_overrides,
            vgae_ckpt_path,
            gat_ckpt_path,
        )
        run_dir_path = Path(run_dir)
        run = self._instantiate_and_inject(rendered)

        # Test (lenient)
        try:
            log.info("stage_test", stage=stage)
            run.trainer.test(run.model, datamodule=run.datamodule, ckpt_path=ckpt_path)
            _touch(run_dir_path / PHASE_MARKERS["test"])
        except Exception as exc:
            log.warning("stage_test_failed", stage=stage, error=str(exc))

        # Analyze (lenient, model-dependent)
        try:
            self._run_analysis(model_type, ckpt_path, run_dir_path)
        except Exception as exc:
            log.warning("stage_analyze_failed", stage=stage, error=str(exc))

        # Finalize run record
        try:
            from graphids.orchestrate.ops.finalize import finalize_run_record

            finalize_run_record(run_dir_path)
        except Exception as exc:
            log.warning("finalize_failed", stage=stage, error=str(exc))

        log.info("stage_eval_complete", stage=stage)

    # -- analysis helper ------------------------------------------------------

    def _run_analysis(self, model_type: str, ckpt_path: str, run_dir: Path) -> None:
        """Run analyzer artifacts if the model type supports them."""
        from graphids.orchestrate.analysis import (
            analysis_flags_for,
            run_analysis,
            supports_analysis,
        )

        if not supports_analysis(model_type):
            return

        from graphids.config.constants import PHASE_MARKERS
        from graphids.core.analysis.schemas import AnalysisSpec

        output_dir = str(Path(ckpt_path).resolve().parent.parent / "artifacts")
        spec = AnalysisSpec(
            ckpt_path=ckpt_path,
            dataset=self.dataset,
            model_type=model_type,
            output_dir=output_dir,
            seed=self.seed,
            **analysis_flags_for(model_type),
        )
        log.info("stage_analyze", model_type=model_type, output_dir=output_dir)
        run_analysis(spec)
        _touch(run_dir / PHASE_MARKERS["analyze"])

    # -- fault tolerance ------------------------------------------------------

    def __supervise__(self, failure: Any) -> bool:
        """Monarch supervision hook -- absorb child mesh failures.

        True = handled (don't propagate to parent / kill controller).
        Monarch does NOT auto-restart actors. Retry logic lives in
        ``pipeline.py._run_with_retry``.
        """
        report = failure.report() if hasattr(failure, "report") else str(failure)
        log.error("actor_supervision", report=report)
        return True
