"""Ray Train launcher for GraphIDS experiment YAML."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from inspect import signature
from pathlib import Path
from typing import Any

from graphids.exp.config import RunConfig
from graphids.exp.journal import (
    EventRecord,
    append_event,
    load_events,
    load_manifest,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Minimal status summary for UI/readout code."""

    run_dir: str
    status: str
    stage: str
    name: str
    last_event: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_spec(spec: Any) -> Any:
    """Resolve a nested primitive/class-path spec without calling ``build()``."""
    if hasattr(spec, "model_dump") and not isinstance(spec, Mapping):
        spec = spec.model_dump(mode="json")
    if isinstance(spec, list):
        return [resolve_spec(item) for item in spec]
    if not isinstance(spec, Mapping):
        return spec

    if "class_path" in spec:
        class_path = str(spec["class_path"])
        module_path, _, class_name = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        init_args = {k: resolve_spec(v) for k, v in dict(spec.get("init_args") or {}).items()}
        return cls(**init_args)

    if "type" in spec:
        from graphids import primitives as primitive_mod

        factory = getattr(primitive_mod, str(spec["type"]), None)
        if callable(factory):
            kwargs = {k: resolve_spec(v) for k, v in spec.items() if k != "type"}
            return factory(**kwargs)

    return {k: resolve_spec(v) for k, v in spec.items()}


def build_component(spec: Any, **build_kwargs: Any) -> Any:
    """Resolve a spec and call ``build()`` when the resolved object supports it."""
    resolved = resolve_spec(spec)
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


def _jsonish(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    return {"value": repr(obj)}


def _tracking_mode() -> str:
    mode = os.environ.get("GRAPHIDS_MLFLOW_MODE", "offline").strip().lower()
    if mode in {"online", "live"}:
        return "online"
    if mode in {"off", "none", "disabled"}:
        return "disabled"
    return "offline"


def _make_logger(run: RunConfig) -> Any:
    from graphids._mlflow import make_logger
    from graphids.exp.ingest import mlflow_artifact_location

    return make_logger(
        experiment_name=f"graphids/{run.dataset or 'unknown'}/{run.stage}",
        run_name=run.name,
        tags=run.mlflow_tags(),
        artifact_location=mlflow_artifact_location(run),
    )


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
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


def _build_lightning_run(
    stage: str,
    cfg: Mapping[str, Any],
    *,
    logger: Any | None,
    ray: bool,
) -> dict[str, Any]:
    import lightning.pytorch as pl

    if "data" not in cfg or "model" not in cfg:
        raise ValueError(f"{stage} requires data and model config blocks")

    data = build_component(cfg["data"])
    if hasattr(data, "setup"):
        data.setup(None)

    loss_fn = build_component(cfg["loss_fn"]) if cfg.get("loss_fn") is not None else None
    model_spec = resolve_spec(cfg["model"])
    if hasattr(model_spec, "build"):
        model = model_spec.build(loss_fn=loss_fn) if loss_fn is not None else model_spec.build()
    else:
        model = model_spec

    if hasattr(model, "prepare_from_datamodule"):
        model.prepare_from_datamodule(data)

    trainer_cfg = resolve_spec(cfg.get("trainer", {}))
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("trainer config must resolve to a mapping of Trainer kwargs")
    trainer_kwargs = dict(trainer_cfg)

    callback_specs = cfg.get("callbacks") or {}
    callbacks = [
        build_component(callback)
        for callback in (callback_specs.values() if isinstance(callback_specs, Mapping) else callback_specs)
    ]
    if logger is not None:
        from graphids._mlflow import MLflowSystemMetricsCallback

        trainer_kwargs["logger"] = logger
        callbacks.append(MLflowSystemMetricsCallback())
    else:
        trainer_kwargs.setdefault("logger", False)

    if ray:
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

        trainer_kwargs["accelerator"] = "auto"
        trainer_kwargs["devices"] = "auto"
        trainer_kwargs["strategy"] = RayDDPStrategy()
        plugins = trainer_kwargs.get("plugins")
        if plugins is None:
            plugins = []
        elif not isinstance(plugins, list):
            plugins = [plugins]
        plugins.append(RayLightningEnvironment())
        trainer_kwargs["plugins"] = plugins
        callbacks.append(RayTrainReportCallback())
        trainer_kwargs.setdefault("enable_checkpointing", False)
    trainer_kwargs["callbacks"] = callbacks

    seed = cfg.get("seed_everything", cfg.get("seed"))
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    trainer = pl.Trainer(**trainer_kwargs)
    if ray:
        trainer = prepare_trainer(trainer)

    if stage == "fit":
        trainer.fit(model, datamodule=data)
        metrics = _scalar_metrics(trainer.callback_metrics)
        if logger is not None and metrics:
            logger.log_metrics(metrics, step=trainer.global_step)
        return {"stage": "fit", "trainer": trainer.__class__.__name__, "metrics": metrics}

    if stage == "test":
        ckpt_path = cfg.get("ckpt_path") or None
        trainer.test(model, datamodule=data, ckpt_path=ckpt_path)
        metrics = _scalar_metrics(trainer.callback_metrics)
        if logger is not None and metrics:
            logger.log_metrics(metrics, step=trainer.global_step)
        return {"stage": "test", "trainer": trainer.__class__.__name__, "ckpt_path": ckpt_path, "metrics": metrics}

    raise ValueError(f"unknown stage: {stage!r}")


def _worker_loop(config: dict[str, Any]) -> None:
    from ray import train

    from graphids.exp.ingest import load_ingest_payload, write_ingest_payload

    mode = _tracking_mode()
    if mode not in {"disabled", "online"}:
        os.environ["GRAPHIDS_MLFLOW_MODE"] = "offline"
    run = RunConfig.model_validate(config["run"])

    if run.stage in {"fit", "test"} and run.resources.accelerator == "gpu":
        from graphids.runtime_checks import assert_pyg_cuda_extensions_match

        assert_pyg_cuda_extensions_match()

    logger = _make_logger(run) if mode == "online" else None
    if logger is not None:
        logger.log_hyperparams(run.mlflow_hparams())
    elif mode == "offline":
        write_ingest_payload(run, status="RUNNING")

    manifest = run.journal_manifest(status="running")
    write_manifest(run.outputs.run_dir, manifest, name=run.outputs.manifest_name)
    append_event(run.outputs.run_dir, EventRecord(status="running", stage=run.stage, message="worker_started"))

    try:
        result = _build_lightning_run(run.stage, run.payload.model_dump(mode="json"), logger=logger, ray=True)
        metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
        if logger is not None and logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FINISHED")
        elif mode == "offline":
            write_ingest_payload(run, status="FINISHED", metrics=metrics, result=_jsonish(result))

        append_event(
            run.outputs.run_dir,
            EventRecord(status="finished", stage=run.stage, message="run_finished", details=_jsonish(result)),
            name=run.outputs.events_name,
        )
        write_manifest(run.outputs.run_dir, manifest.model_copy(update={"status": "finished"}), name=run.outputs.manifest_name)
        try:
            ingest_payload = load_ingest_payload(run.outputs.run_dir)
            metrics = {
                str(key): float(value)
                for key, value in dict(ingest_payload.get("metrics") or {}).items()
                if isinstance(value, int | float)
            }
        except FileNotFoundError:
            pass
        train.report({**metrics, "graphids_finished": 1.0})
    except BaseException as exc:  # noqa: BLE001 - journal failures before Ray re-raises
        failure = f"{type(exc).__name__}: {exc}"
        if logger is not None and logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FAILED")
        elif mode == "offline":
            write_ingest_payload(run, status="FAILED", failure=failure)
        append_event(
            run.outputs.run_dir,
            EventRecord(status="failed", stage=run.stage, message="run_failed", details={"failure": failure}),
            name=run.outputs.events_name,
        )
        write_manifest(
            run.outputs.run_dir,
            manifest.model_copy(update={"status": "failed", "failure": failure}),
            name=run.outputs.manifest_name,
        )
        raise


def probe_ray_train_imports() -> dict[str, str]:
    """Import the Ray APIs GraphIDS relies on and return their module paths."""
    try:
        from ray.train import CheckpointConfig, ScalingConfig
        from ray.train import RunConfig as RayRunConfig
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )
        from ray.train.torch import TorchTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ray is not installed. Install the project dependencies before launching experiments.") from exc

    return {
        "RunConfig": RayRunConfig.__module__,
        "ScalingConfig": ScalingConfig.__module__,
        "CheckpointConfig": CheckpointConfig.__module__,
        "TorchTrainer": TorchTrainer.__module__,
        "prepare_trainer": prepare_trainer.__module__,
        "RayDDPStrategy": RayDDPStrategy.__module__,
        "RayLightningEnvironment": RayLightningEnvironment.__module__,
        "RayTrainReportCallback": RayTrainReportCallback.__module__,
    }


def launch_run(run: RunConfig, *, address: str | None = None) -> RunSummary:
    """Run a GraphIDS experiment through Ray Train."""
    try:
        import ray
        from ray.train import CheckpointConfig, ScalingConfig
        from ray.train import RunConfig as RayRunConfig
        from ray.train.torch import TorchTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ray is not installed. Install the project dependencies before launching experiments.") from exc

    if not ray.is_initialized():
        ray.init(address=address, ignore_reinit_error=True)

    resources = run.resources
    devices = run.payload.trainer.get("devices")
    num_workers = devices if isinstance(devices, int) and devices > 1 else 1
    use_gpu = resources.accelerator == "gpu" or float(resources.gpus_per_worker) > 0
    resources_per_worker: dict[str, float | int] = {"CPU": max(1, int(resources.cpus_per_worker))}
    if float(resources.gpus_per_worker) > 0:
        resources_per_worker["GPU"] = float(resources.gpus_per_worker)

    run_dir = Path(run.outputs.run_dir)
    ray_run_kwargs: dict[str, Any] = {"storage_path": str(run_dir.parent), "name": run_dir.name}
    callbacks = run.payload.callbacks or {}
    callback_values = callbacks.values() if isinstance(callbacks, Mapping) else callbacks
    for callback in callback_values:
        if not isinstance(callback, Mapping):
            continue
        callback_type = str(callback.get("type") or callback.get("class_path") or "")
        if "checkpoint" not in callback_type.lower():
            continue
        checkpoint_kwargs: dict[str, Any] = {}
        if isinstance(callback.get("save_top_k"), int) and callback["save_top_k"] > 0:
            checkpoint_kwargs["num_to_keep"] = callback["save_top_k"]
        if isinstance(callback.get("monitor"), str):
            checkpoint_kwargs["checkpoint_score_attribute"] = callback["monitor"]
        if callback.get("mode") in {"min", "max"}:
            checkpoint_kwargs["checkpoint_score_order"] = callback["mode"]
        if checkpoint_kwargs:
            ray_run_kwargs["checkpoint_config"] = CheckpointConfig(**checkpoint_kwargs)
        break

    result = TorchTrainer(
        _worker_loop,
        train_loop_config={"run": run.model_dump(mode="json")},
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker=resources_per_worker,
        ),
        run_config=RayRunConfig(**ray_run_kwargs),
    ).fit()

    append_event(
        run.outputs.run_dir,
        EventRecord(
            status="finished",
            stage=run.stage,
            message="ray_result",
            details={"path": str(getattr(result, "path", "") or ""), "metrics": _jsonish(getattr(result, "metrics", {}) or {})},
        ),
        name=run.outputs.events_name,
    )
    summary = summarize_run(run.outputs.run_dir)
    if summary is None:
        raise RuntimeError(f"Ray run finished without a GraphIDS manifest: {run.outputs.run_dir}")
    return summary


def summarize_run(run_dir: str | Path) -> RunSummary | None:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return None
    events = load_events(run_dir)
    return RunSummary(
        run_dir=manifest.run_dir,
        status=manifest.status,
        stage=manifest.stage,
        name=manifest.name,
        last_event=events[-1].message if events else None,
        error=manifest.failure,
        extra={"git_sha": manifest.git_sha, "run_id": manifest.run_id},
    )
