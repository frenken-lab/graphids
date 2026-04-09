"""Pipeline actor — sequences training stages in one SLURM allocation."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from graphids.log import get_logger
from graphids.orchestrate._setup import ensure_spawn, touch_marker

log = get_logger(__name__)

try:
    from monarch.actor import Actor, endpoint  # type: ignore[import-not-found]
except ImportError:

    class Actor:  # type: ignore[no-redef]
        pass

    def endpoint(fn):  # type: ignore[no-redef]
        return fn


class PipelineActor(Actor):
    """Runs all pipeline stages, caching datasets across stages."""

    def __init__(self, lake_root: str, user: str = "") -> None:
        ensure_spawn()
        self.lake_root = lake_root
        self.user = user or os.environ.get("USER", "unknown")
        self._cached_datasets: dict[str, Any] | None = None
        self._setup_otel()

    def _setup_otel(self) -> None:
        """Phase A: OTel providers for Monarch worker (bypasses __main__.py)."""
        import atexit
        import logging
        import sys

        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import (
            BatchLogRecordProcessor,
            ConsoleLogRecordExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": "graphids.monarch",
            "slurm.job_id": os.environ.get("SLURM_JOB_ID", ""),
        })
        self._tracer_provider = TracerProvider(resource=resource)
        if os.environ.get("WANDB_API_KEY"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            self._tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint="https://trace.wandb.ai/otel/v1/traces",
                headers={"wandb-api-key": os.environ["WANDB_API_KEY"]},
            )))
        trace.set_tracer_provider(self._tracer_provider)
        metrics.set_meter_provider(MeterProvider(resource=resource))

        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr)))
        set_logger_provider(lp)
        logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=lp))
        self._logger_provider = lp
        atexit.register(lambda: (self._tracer_provider.shutdown(), lp.shutdown()))

    def _wire_file_exporters(self, run_dir: Path) -> None:
        """Phase B: add file exporters once run_dir is known for a stage."""
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        self._tracer_provider.add_span_processor(SimpleSpanProcessor(
            ConsoleSpanExporter(out=open(run_dir / "traces.jsonl", "a"))  # noqa: SIM115
        ))
        mp = MeterProvider(
            resource=self._tracer_provider.resource,
            metric_readers=[PeriodicExportingMetricReader(
                ConsoleMetricExporter(out=open(run_dir / "metrics.jsonl", "a")),  # noqa: SIM115
                export_interval_millis=10_000,
            )],
        )
        metrics.set_meter_provider(mp)

    # -- resolve + instantiate --------------------------------------------------

    def _resolve(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str],
    ) -> Any:
        """Resolve a StageConfig dict into a ResolvedConfig."""
        from graphids.orchestrate.planning import StageConfig
        from graphids.orchestrate.resolve import ResolvedConfig

        cfg = StageConfig.model_validate(stage_config)
        return ResolvedConfig.resolve(
            cfg,
            lake_root=self.lake_root,
            user=self.user,
            dataset=dataset,
            seed=seed,
            upstream_ckpts=upstream_ckpts,
        )

    def _instantiate(self, resolved: Any) -> Any:
        """Instantiate Lightning stack, injecting cached datasets. Frees GPU first."""
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.compiler.reset()

        from graphids.instantiate import instantiate

        run = instantiate(resolved.rendered, validated=resolved.validated)

        if self._cached_datasets is not None:
            run.datamodule._train_ds = self._cached_datasets["train"]
            run.datamodule._val_ds = self._cached_datasets["val"]
            run.datamodule._test_datasets = self._cached_datasets["test"]
        return run

    # -- dataset cache ----------------------------------------------------------

    @staticmethod
    def _clone_to_cpu(dataset: Any) -> Any:
        """Deep-clone a PyG dataset to CPU (Data.to() is in-place)."""
        from torch_geometric.data import InMemoryDataset

        if isinstance(dataset, dict):
            return {k: PipelineActor._clone_to_cpu(v) for k, v in dataset.items()}
        if isinstance(dataset, InMemoryDataset):
            return dataset.copy().cpu()
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

    # -- stage endpoints --------------------------------------------------------

    @endpoint
    def train_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> str:
        """Train a single stage. Returns checkpoint path. Idempotent."""
        from graphids.config.constants import PHASE_MARKERS

        resolved = self._resolve(stage_config, dataset, seed, upstream_ckpts or {})
        ckpt_path = str(resolved.paths.ckpt_file)
        run_dir = Path(str(resolved.paths.run_dir))

        if resolved.paths.ckpt_file.exists() and resolved.paths.complete_marker.exists():
            log.info("stage_skip_complete", stage=stage_config.get("stage"), run_dir=str(run_dir))
            return ckpt_path

        # Phase B: wire file exporters for this stage's run_dir
        self._wire_file_exporters(run_dir)

        run = self._instantiate(resolved)

        log.info("stage_train", stage=stage_config.get("stage"), run_dir=str(run_dir))
        try:
            run.trainer.fit(run.model, datamodule=run.datamodule)
        except Exception:
            self._cached_datasets = None
            raise

        self._cache_datasets_from(run.datamodule)
        touch_marker(run_dir / PHASE_MARKERS["train"])
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
        """Run test + analyze + finalize for a completed stage. All lenient."""
        from graphids.config.constants import PHASE_MARKERS

        resolved = self._resolve(stage_config, dataset, seed, upstream_ckpts or {})
        ckpt_path = str(resolved.paths.ckpt_file)
        run_dir = Path(str(resolved.paths.run_dir))
        model_type = stage_config.get("model_type", "")

        run = self._instantiate(resolved)

        # Test (lenient)
        try:
            log.info("stage_test", stage=stage_config.get("stage"))
            run.trainer.test(run.model, datamodule=run.datamodule, ckpt_path=ckpt_path)
            touch_marker(run_dir / PHASE_MARKERS["test"])
        except Exception as exc:
            log.warning("stage_test_failed", stage=stage_config.get("stage"), error=str(exc))

        # Analyze (lenient, model-dependent)
        if model_type in ("vgae", "gat", "dgi"):
            try:
                from graphids.core.analysis.schemas import AnalysisSpec
                from graphids.orchestrate.analysis import run_analysis

                spec = AnalysisSpec(
                    ckpt_path=ckpt_path,
                    dataset=dataset,
                    model_type=model_type,
                    output_dir=str(Path(ckpt_path).resolve().parent.parent / "artifacts"),
                    seed=seed,
                )
                log.info("stage_analyze", model_type=model_type)
                run_analysis(spec)
                touch_marker(run_dir / PHASE_MARKERS["analyze"])
            except Exception as exc:
                log.warning("stage_analyze_failed", stage=stage_config.get("stage"), error=str(exc))

        touch_marker(resolved.paths.complete_marker)
        log.info("stage_eval_complete", stage=stage_config.get("stage"))

    # -- fault tolerance --------------------------------------------------------

    def __supervise__(self, failure: Any) -> bool:
        """Monarch supervision hook — absorb child mesh failures."""
        report = failure.report() if hasattr(failure, "report") else str(failure)
        log.error("actor_supervision", report=report)
        return True
