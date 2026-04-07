"""Pipeline actor -- holds TMPDIR staging across stages, sequences training.

Each ``@endpoint`` method wraps the existing instantiate chain unchanged.
Data staging happens once via ``bootstrap_staging`` (passed to
``spawn_procs(bootstrap=...)``). Checkpoint paths thread between stages
via return values.

The actor accepts ``StageConfig`` dicts directly from the controller --
no in-actor config construction. Both ``monarch-run`` (single pipeline)
and ``monarch-sweep`` (recipe-driven) build StageConfigs on the login
node via the planner, then pass them to endpoints. This eliminates the
former ``_build_stage_config`` duplication with ``planner.enumerate_assets``.

When monarch is not installed, this module still imports cleanly -- the
``endpoint`` decorator is a no-op passthrough so tests can instantiate the
actor as a plain Python object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from graphids.log import get_logger
from graphids.monarch._setup import ensure_spawn, touch_marker

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


class PipelineActor(Actor):
    """Single actor running all pipeline stages in one SLURM allocation.

    Holds cached datasets across stages. Each endpoint receives a
    serialized ``StageConfig`` dict, resolves it via ``ConfigResolver``,
    instantiates the Lightning stack, and runs trainer methods.

    Config resolution delegates to ``ConfigResolver.resolve()`` — the
    same path used by dagster assets — so paths, identity hashes, TLA
    construction, and validation are always in sync.
    """

    def __init__(self, lake_root: str, user: str = "") -> None:
        self.lake_root = lake_root
        self.user = user or os.environ.get("USER", "unknown")
        self._cached_datasets: dict[str, Any] | None = None

    # -- stage preparation (shared by train + eval) ---------------------------

    def _prepare_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str],
    ) -> tuple[str, str, str, Any]:
        """Resolve config via ``ConfigResolver`` and return stage metadata.

        Returns ``(model_type, ckpt_path, run_dir, resolved)`` where
        ``resolved`` is a ``ResolvedConfig`` with ``.rendered`` and
        ``.validated`` fields ready for ``instantiate()``.
        """
        ensure_spawn()

        from graphids.orchestrate.planning import StageConfig
        from graphids.orchestrate.resolve import ConfigResolver

        cfg = StageConfig.from_dict(stage_config)

        resolver = ConfigResolver(lake_root=self.lake_root, user=self.user)
        resolved = resolver.resolve(
            cfg,
            dataset=dataset,
            seed=seed,
            upstream_ckpts=upstream_ckpts,
        )

        return (
            cfg.model_type,
            str(resolved.paths.ckpt_file),
            str(resolved.paths.run_dir),
            resolved,
        )

    # -- dataset cache --------------------------------------------------------

    def _instantiate_and_inject(self, resolved: Any) -> Any:
        """Instantiate Lightning stack from resolved config, injecting cached datasets.

        ``resolved`` is a ``ResolvedConfig`` from ``ConfigResolver.resolve()``.
        Validation already happened inside the resolver — no redundant
        ``validate_config()`` call.
        """
        from graphids.instantiate import instantiate

        run = instantiate(resolved.rendered, validated=resolved.validated)

        if self._cached_datasets is not None:
            run.datamodule._train_ds = self._cached_datasets["train"]
            run.datamodule._val_ds = self._cached_datasets["val"]
            run.datamodule._test_datasets = self._cached_datasets["test"]
        return run

    @staticmethod
    def _clone_to_cpu(dataset: Any) -> Any:
        """Deep-clone a PyG dataset/list to CPU tensors.

        PyG Data.to() is in-place — after training, cached references
        would point to CUDA tensors. Cloning ensures the cache always
        holds CPU data (Monarch pattern: data on CPU, move to GPU at
        computation time).
        """
        if isinstance(dataset, dict):
            return {k: [d.clone().cpu() for d in v] for k, v in dataset.items()}
        if isinstance(dataset, (list, tuple)):
            return [d.clone().cpu() for d in dataset]
        return dataset

    def _cache_datasets_from(self, datamodule: Any) -> None:
        """Cache CPU copies of datasets from a datamodule after setup."""
        if self._cached_datasets is None and datamodule._train_ds is not None:
            self._cached_datasets = {
                "train": self._clone_to_cpu(datamodule._train_ds),
                "val": self._clone_to_cpu(datamodule._val_ds),
                "test": self._clone_to_cpu(datamodule._test_datasets),
            }
            log.info("datasets_cached", num_train=len(datamodule._train_ds))

    # -- stage endpoints ------------------------------------------------------

    @endpoint
    def train_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> str:
        """Train a single stage. Returns checkpoint path.

        Idempotent: if the run directory has a ``.complete`` marker and
        the checkpoint exists, returns immediately without training.
        """
        from graphids.config.constants import PHASE_MARKERS

        upstream_ckpts = upstream_ckpts or {}
        _, ckpt_path, run_dir, resolved = self._prepare_stage(
            stage_config,
            dataset,
            seed,
            upstream_ckpts,
        )

        # Idempotency: skip if already complete
        if resolved.paths.ckpt_file.exists() and resolved.paths.complete_marker.exists():
            log.info("stage_skip_complete", stage=stage_config.get("stage"), run_dir=run_dir)
            return ckpt_path

        run = self._instantiate_and_inject(resolved)

        log.info("stage_train", stage=stage_config.get("stage"), run_dir=run_dir)
        run.trainer.fit(run.model, datamodule=run.datamodule)

        # Cache CPU clones after first load. Must happen after fit()
        # (setup() populates datasets during fit), but we clone to CPU
        # so CUDA in-place mutations from PrefetchLoader don't poison
        # the cache. Subsequent stages/retries get clean CPU data.
        self._cache_datasets_from(run.datamodule)
        touch_marker(Path(run_dir) / PHASE_MARKERS["train"])
        log.info("stage_train_complete", stage=stage_config.get("stage"), ckpt=ckpt_path)

        return ckpt_path

    @endpoint
    def eval_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> None:
        """Run test + analyze + finalize for a completed stage. All lenient.

        Writes the ``.complete`` marker after all phases succeed or are
        handled, so that subsequent runs skip this stage via idempotency.
        """
        from graphids.config.constants import PHASE_MARKERS

        upstream_ckpts = upstream_ckpts or {}
        model_type, ckpt_path, run_dir, resolved = self._prepare_stage(
            stage_config,
            dataset,
            seed,
            upstream_ckpts,
        )
        run_dir_path = Path(run_dir)
        run = self._instantiate_and_inject(resolved)

        # Test (lenient)
        try:
            log.info("stage_test", stage=stage_config.get("stage"))
            run.trainer.test(run.model, datamodule=run.datamodule, ckpt_path=ckpt_path)
            touch_marker(run_dir_path / PHASE_MARKERS["test"])
        except Exception as exc:
            log.warning("stage_test_failed", stage=stage_config.get("stage"), error=str(exc))

        # Analyze (lenient, model-dependent)
        try:
            self._run_analysis(model_type, ckpt_path, run_dir_path, dataset, seed)
        except Exception as exc:
            log.warning("stage_analyze_failed", stage=stage_config.get("stage"), error=str(exc))

        # Finalize run record
        try:
            from graphids.orchestrate.ops.finalize import finalize_run_record

            finalize_run_record(run_dir_path)
        except Exception as exc:
            log.warning("finalize_failed", stage=stage_config.get("stage"), error=str(exc))

        # Write .complete marker so future runs skip this stage
        touch_marker(resolved.paths.complete_marker)

        log.info("stage_eval_complete", stage=stage_config.get("stage"))

    # -- analysis helper ------------------------------------------------------

    def _run_analysis(
        self,
        model_type: str,
        ckpt_path: str,
        run_dir: Path,
        dataset: str,
        seed: int,
    ) -> None:
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
            dataset=dataset,
            model_type=model_type,
            output_dir=output_dir,
            seed=seed,
            **analysis_flags_for(model_type),
        )
        log.info("stage_analyze", model_type=model_type, output_dir=output_dir)
        run_analysis(spec)
        touch_marker(run_dir / PHASE_MARKERS["analyze"])

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
