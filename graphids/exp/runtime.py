"""Execution helpers for the new experiment seam.

Ray/Hydra can attach here later. For now this module gives us a single place to
write manifests and events around any callable run body.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from inspect import signature
from pathlib import Path
from typing import Any

from graphids._mlflow import MLflowSystemMetricsCallback, make_logger
from graphids.exp.config import RunConfig, RunSummary
from graphids.exp.journal import (
    EventRecord,
    append_event,
    load_events,
    load_manifest,
    write_manifest,
)


def _mlflow_artifact_location(run: RunConfig) -> str:
    from graphids.paths import lake_root

    dataset = run.dataset or "unknown"
    return str(Path(lake_root()) / "mlartifacts" / dataset / run.stage)


def _payload(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": repr(obj)}


def _resolve_spec(spec: Any) -> Any:
    if hasattr(spec, "model_dump") and not isinstance(spec, Mapping):
        spec = spec.model_dump(mode="json")
    if isinstance(spec, list):
        return [_resolve_spec(item) for item in spec]
    if not isinstance(spec, Mapping):
        return spec

    if "class_path" in spec:
        class_path = str(spec["class_path"])
        module_path, _, class_name = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        init_args = {k: _resolve_spec(v) for k, v in dict(spec.get("init_args") or {}).items()}
        return cls(**init_args)

    if "type" in spec:
        from graphids import primitives as primitive_mod

        factory = getattr(primitive_mod, str(spec["type"]), None)
        if callable(factory):
            kwargs = {k: _resolve_spec(v) for k, v in spec.items() if k != "type"}
            return factory(**kwargs)

    return {k: _resolve_spec(v) for k, v in spec.items()}


def _build_component(spec: Any, **build_kwargs: Any) -> Any:
    resolved = _resolve_spec(spec)
    builder = getattr(resolved, "build", None)
    if callable(builder):
        if build_kwargs:
            params = signature(builder).parameters
            filtered = {k: v for k, v in build_kwargs.items() if k in params}
            try:
                return builder(**filtered)
            except TypeError:
                pass
        return builder()
    return resolved


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert Lightning callback metrics to logger-friendly scalars."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        item = value
        detach = getattr(item, "detach", None)
        if callable(detach):
            item = detach()
        cpu = getattr(item, "cpu", None)
        if callable(cpu):
            item = cpu()
        scalar = getattr(item, "item", None)
        if callable(scalar):
            try:
                out[str(key)] = float(scalar())
                continue
            except (TypeError, ValueError, RuntimeError):
                continue
        if isinstance(item, int | float):
            out[str(key)] = float(item)
    return out


def _run_fit_or_test(
    action: str,
    cfg: Mapping[str, Any],
    *,
    ckpt_path: str | None = None,
    logger: Any | None = None,
) -> Any:
    import lightning.pytorch as pl

    if "data" not in cfg or "model" not in cfg:
        raise ValueError(f"{action} requires data and model config blocks")

    data = _build_component(cfg["data"])
    if hasattr(data, "setup"):
        data.setup(None)

    loss_fn = _build_component(cfg["loss_fn"]) if cfg.get("loss_fn") is not None else None
    model_spec = _resolve_spec(cfg["model"])
    if hasattr(model_spec, "build"):
        model = model_spec.build(loss_fn=loss_fn) if loss_fn is not None else model_spec.build()
    else:
        model = model_spec

    if hasattr(model, "prepare_from_datamodule"):
        model.prepare_from_datamodule(data)

    trainer_cfg = _resolve_spec(cfg.get("trainer", {}))
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("trainer config must resolve to a mapping of Trainer kwargs")
    trainer_kwargs = dict(trainer_cfg)
    callbacks = cfg.get("callbacks") or {}
    if isinstance(callbacks, Mapping):
        callback_specs = callbacks.values()
    else:
        callback_specs = callbacks
    if callback_specs:
        trainer_kwargs["callbacks"] = [_build_component(cb) for cb in callback_specs]
    if logger is not None:
        trainer_kwargs.setdefault("logger", logger)
        callbacks = list(trainer_kwargs.get("callbacks") or [])
        if not any(isinstance(cb, MLflowSystemMetricsCallback) for cb in callbacks):
            callbacks.append(MLflowSystemMetricsCallback())
        trainer_kwargs["callbacks"] = callbacks

    seed = cfg.get("seed_everything", cfg.get("seed"))
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    trainer = pl.Trainer(**trainer_kwargs)
    if action == "fit":
        trainer.fit(model, datamodule=data)
        metrics = _scalar_metrics(trainer.callback_metrics)
        if logger is not None and metrics:
            logger.log_metrics(metrics, step=trainer.global_step)
        return {"stage": "fit", "trainer": trainer.__class__.__name__, "metrics": metrics}

    test_ckpt = cfg.get("ckpt_path", ckpt_path)
    if test_ckpt in ("", None):
        test_ckpt = None
    trainer.test(model, datamodule=data, ckpt_path=test_ckpt)
    metrics = _scalar_metrics(trainer.callback_metrics)
    if logger is not None and metrics:
        logger.log_metrics(metrics, step=trainer.global_step)
    return {
        "stage": "test",
        "trainer": trainer.__class__.__name__,
        "ckpt_path": test_ckpt,
        "metrics": metrics,
    }


def _run_cache(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if "data" not in cfg:
        raise ValueError("cache requires a data config block")
    data = _build_component(cfg["data"])
    if hasattr(data, "setup"):
        data.setup(None)
    source = getattr(data, "source", data)
    cache_root = getattr(source, "cache_root_path", None)
    cache_ready = getattr(source, "cache_ready", None)
    return {
        "stage": "cache",
        "cache_key": str(getattr(source, "cache_key", "")),
        "cache_root": str(cache_root()) if callable(cache_root) else "",
        "cache_ready": bool(cache_ready()) if callable(cache_ready) else True,
    }


def run_stage(run: RunConfig, logger: Any | None = None) -> dict[str, Any] | None:
    """Default stage dispatcher for experiment launches.

    Fit/test, extract, and analyze all run directly from the typed
    experiment config objects.
    """
    if run.stage in {"fit", "test"}:
        payload = run.payload.model_dump(mode="json")
        return _run_fit_or_test(
            run.stage,
            payload,
            ckpt_path=payload.get("ckpt_path"),
            logger=logger,
        )
    if run.stage == "cache":
        payload = run.payload.model_dump(mode="json")
        return _run_cache(payload)
    if run.stage == "extract":
        from graphids.core.data.extract import extract_states

        run_cfg = run.payload.model_dump(mode="json")
        checkpoints = run_cfg.get("checkpoints") or run_cfg.get("extractor_ckpts")
        if checkpoints is None:
            raise ValueError("extract requires checkpoints or extractor_ckpts")
        dataset = run_cfg.get("dataset")
        output_dir = run_cfg.get("output_dir")
        if not dataset or not output_dir:
            raise ValueError("extract requires dataset and output_dir")
        extract_states(
            checkpoints=checkpoints,
            dataset=dataset,
            output_dir=output_dir,
            max_samples=int(run_cfg.get("max_samples", 150_000)),
            max_val_samples=int(run_cfg.get("max_val_samples", 30_000)),
            batch_size=int(run_cfg.get("batch_size", 256)),
            seed=int(run_cfg.get("seed", run.seed)),
            val_fraction=float(run_cfg.get("val_fraction", 0.2)),
            representation_cfg=run.representation_cfg,
        )
        return {"stage": "extract", "output_dir": output_dir}
    if run.stage == "analyze":
        from graphids.core.artifacts.analyzer import AnalysisConfig, Analyzer

        run_cfg = run.payload.model_dump(mode="json")
        spec = AnalysisConfig(
            name=run_cfg.get("name", run.name),
            plan_id=run_cfg.get("plan_id", run.plan_id or run.name),
            ckpt_path=str(run_cfg.get("ckpt_path", "")),
            dataset=str(run_cfg.get("dataset", run.dataset or "")),
            model_type=str(run_cfg.get("model_type", "gat")),
            output_dir=str(run_cfg.get("output_dir", "")),
            lake_root=str(run_cfg.get("lake_root", "")),
            embeddings=bool(run_cfg.get("embeddings", True)),
            attention=bool(run_cfg.get("attention", False)),
            cka=bool(run_cfg.get("cka", False)),
            landscape=bool(run_cfg.get("landscape", False)),
            fusion_policy=bool(run_cfg.get("fusion_policy", False)),
            cka_teacher_ckpt=str(run_cfg.get("cka_teacher_ckpt", "")),
            cka_max_samples=int(run_cfg.get("cka_max_samples", 500)),
            landscape_resolution=int(run_cfg.get("landscape_resolution", 51)),
            landscape_scale=float(run_cfg.get("landscape_scale", 1.0)),
            landscape_max_graphs=int(run_cfg.get("landscape_max_graphs", 500)),
            embedding_max_samples=int(run_cfg.get("embedding_max_samples", 2000)),
            attention_max_samples=int(run_cfg.get("attention_max_samples", 50)),
            batch_size=int(run_cfg.get("batch_size", 256)),
            seed=int(run_cfg.get("seed", run.seed)),
            vocab_scope=str(run_cfg.get("vocab_scope", "train")),
            representation_cfg=run.representation_cfg,
            vgae_ckpt_path=str(run_cfg.get("vgae_ckpt_path", "")),
            gat_ckpt_path=str(run_cfg.get("gat_ckpt_path", "")),
        )
        Analyzer(spec).run()
        return {"stage": "analyze", "output_dir": spec.output_dir}
    if run.stage == "hf_push":
        raise NotImplementedError(f"stage {run.stage!r} is not wired yet")
    raise ValueError(f"unknown stage: {run.stage!r}")


def _make_run_logger(
    run: RunConfig,
    *,
    run_id: str | None = None,
) -> Any:
    return make_logger(
        experiment_name=f"graphids/{run.dataset or 'unknown'}/{run.stage}",
        run_name=run.name,
        tags=run.mlflow_tags(),
        artifact_location=_mlflow_artifact_location(run),
        run_id=run_id,
    )


def _run_stage_with_existing_mlflow_run(run: RunConfig, run_id: str) -> dict[str, Any] | None:
    """Ray-safe stage entrypoint.

    MLflow logger objects are process-local and should not be serialized into
    Ray workers. Pass the existing run id instead, then bind a fresh
    ``MLFlowLogger`` in the worker to that run.
    """
    logger = _make_run_logger(run, run_id=run_id)
    return run_stage(run, logger=logger)


def launch_run(
    run: RunConfig,
) -> RunSummary:
    """Run one launchable config with manifest/event tracking."""
    backend = run.resources.backend
    if backend == "ray":
        try:
            import ray  # noqa: F401
        except ImportError:
            backend = "local"
    logger = _make_run_logger(run)
    logger.log_hyperparams(run.mlflow_hparams(backend=backend))
    manifest = run.journal_manifest(status="running")
    write_manifest(run.outputs.run_dir, manifest, name=run.outputs.manifest_name)
    append_event(
        run.outputs.run_dir,
        EventRecord(
            status="running",
            stage=run.stage,
            message="launch_started",
            details={"backend": backend},
        ),
        name=run.outputs.events_name,
    )

    try:
        if backend == "ray":
            import ray

            ray.init(ignore_reinit_error=True, include_dashboard=False)
            if logger.run_id is None:
                raise RuntimeError("MLflow logger did not create a run id before Ray launch")
            result = ray.get(ray.remote(_run_stage_with_existing_mlflow_run).remote(run, logger.run_id))
        else:
            result = run_stage(run, logger=logger)
        if logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FINISHED")
        append_event(
            run.outputs.run_dir,
            EventRecord(status="finished", stage=run.stage, message="run_finished", details=_payload(result)),
            name=run.outputs.events_name,
        )
        write_manifest(
            run.outputs.run_dir,
            manifest.model_copy(update={"status": "finished"}),
            name=run.outputs.manifest_name,
        )
        return RunSummary(
            run_dir=str(run.outputs.run_dir),
            status="finished",
            stage=run.stage,
            name=run.name,
            last_event="run_finished",
        )
    except BaseException as exc:  # noqa: BLE001 - record all failures, then re-raise
        failure = f"{type(exc).__name__}: {exc}"
        if logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FAILED")
        append_event(
            run.outputs.run_dir,
            EventRecord(
                status="failed",
                stage=run.stage,
                message="run_failed",
                details={"failure": failure},
            ),
            name=run.outputs.events_name,
        )
        write_manifest(
            run.outputs.run_dir,
            manifest.model_copy(update={"status": "failed", "failure": failure}),
            name=run.outputs.manifest_name,
        )
        raise


def summarize_run(run_dir: str | Path) -> RunSummary | None:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return None
    events = load_events(run_dir)
    last = events[-1].message if events else None
    err = manifest.failure
    return RunSummary(
        run_dir=manifest.run_dir,
        status=manifest.status,
        stage=manifest.stage,
        name=manifest.name,
        last_event=last,
        error=err,
        extra={"git_sha": manifest.git_sha, "run_id": manifest.run_id},
    )
