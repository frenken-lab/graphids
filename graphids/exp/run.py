"""Construct and execute one validated run locally or through Ray Train."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from numbers import Real
from typing import Any

from pydantic import BaseModel

from graphids.exp.config import RunConfig, RunPayload
from graphids.exp.records import (
    RunSummary,
    load_manifest,
    record_status,
    summarize_run,
    write_config,
)
from graphids.exp.tracking import make_logger, tracking_mode


def resolve_spec(spec: Any) -> Any:
    """Resolve a nested component/class-path spec without calling ``build()``."""
    if isinstance(spec, BaseModel):
        spec = spec.model_dump(mode="json")
    if isinstance(spec, list):
        return [resolve_spec(item) for item in spec]
    if not isinstance(spec, Mapping):
        return spec
    if "class_path" in spec:
        class_path = str(spec["class_path"])
        module_path, _, class_name = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        init_args = {
            k: resolve_spec(v) for k, v in dict(spec.get("init_args") or {}).items()
        }
        return cls(**init_args)

    if "type" in spec:
        match spec["type"]:
            case "can_event_dm":
                from graphids.data.config import CANEventDataConfig

                config_class = CANEventDataConfig
            case "event_graph_mamba":
                from graphids.models.config import EventGraphMambaConfig

                config_class = EventGraphMambaConfig
            case _:
                raise ValueError(f"unknown component type: {spec['type']!r}")
        return config_class.model_validate(
            {k: resolve_spec(v) for k, v in spec.items()}
        )

    return {k: resolve_spec(v) for k, v in spec.items()}


def build_component(spec: Any) -> Any:
    """Build a component config, or return an instantiated class-path object.

    ``build`` is the optional construction hook at this config boundary.
    Errors from the builder propagate unchanged.
    """
    resolved = resolve_spec(spec)
    builder = getattr(resolved, "build", None)
    return builder() if builder is not None else resolved


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Extract numeric scalars from Lightning metrics, ignoring vector metrics."""
    from torch import Tensor

    out = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, Real):
            out[str(key)] = float(value)
    return out


def _compile_model_if_requested(model: Any, compile_cfg: Any) -> None:
    if compile_cfg in (None, False):
        return
    if compile_cfg is True:
        compile_kwargs: dict[str, Any] = {}
    else:
        resolved = resolve_spec(compile_cfg)
        if not isinstance(resolved, Mapping):
            raise TypeError("compile config must resolve to a mapping")
        compile_kwargs = dict(resolved)
        if compile_kwargs.pop("enabled", True) is False:
            return
    model.compile(**compile_kwargs)


def seed_run(seed: int | None) -> None:
    import lightning.pytorch as pl

    if seed is not None:
        pl.seed_everything(int(seed), workers=True)


def build_model(cfg: RunPayload, data: Any) -> Any:
    model = build_component(cfg.model)
    # Custom class-path models may omit this event-data preparation hook.
    prepare = getattr(model, "prepare_from_datamodule", None)
    if prepare is not None:
        prepare(data)
    _compile_model_if_requested(model, cfg.compile)
    return model


def trainer_components(cfg: RunPayload) -> dict[str, Any]:
    trainer_cfg = resolve_spec(cfg.trainer)
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("trainer config must resolve to a mapping of Trainer kwargs")
    kwargs = dict(trainer_cfg)
    specs = cfg.callbacks
    kwargs["callbacks"] = [
        build_component(spec)
        for spec in (specs.values() if isinstance(specs, Mapping) else specs)
    ]
    return kwargs


def run_lightning(
    run: RunConfig,
    *,
    logger: Any | None = None,
    ray: bool = False,
) -> dict[str, Any]:
    import lightning.pytorch as pl

    cfg = run.payload
    stage = run.stage
    seed_run(cfg.seed_everything if cfg.seed_everything is not None else run.seed)
    data = build_component(cfg.data)
    data.setup(None)
    model = build_model(cfg, data)
    trainer_kwargs = trainer_components(cfg)
    trainer_kwargs.setdefault("default_root_dir", str(run.outputs.run_dir))
    callbacks = trainer_kwargs["callbacks"]
    writer = True
    if ray and (logger is not None or hasattr(data, "manifest")):
        from ray import train

        writer = train.get_context().get_world_rank() == 0
    if writer and (logger is not None or hasattr(data, "manifest")):
        from graphids._mlflow import MLflowSystemMetricsCallback

        if logger is not None:
            trainer_kwargs["logger"] = logger
        callbacks.append(MLflowSystemMetricsCallback())
    if logger is None:
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
        if hasattr(data, "train_sampler"):
            trainer_kwargs["use_distributed_sampler"] = False
        plugins = trainer_kwargs.get("plugins")
        if plugins is None:
            plugins = []
        elif not isinstance(plugins, list):
            plugins = [plugins]
        plugins.append(RayLightningEnvironment())
        trainer_kwargs["plugins"] = plugins
        callbacks.append(RayTrainReportCallback())
        trainer_kwargs.setdefault(
            "enable_checkpointing",
            any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in callbacks),
        )
    trainer_kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(**trainer_kwargs)
    if ray:
        trainer = prepare_trainer(trainer)

    ckpt_path = cfg.ckpt_path or None
    if stage == "fit":
        trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)
    elif stage == "test":
        trainer.test(model, datamodule=data, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"unknown stage: {stage!r}")
    metrics = _scalar_metrics(trainer.callback_metrics)
    if logger is not None and metrics:
        logger.log_metrics(metrics, step=trainer.global_step)
    result = {"stage": stage, "trainer": trainer.__class__.__name__, "metrics": metrics}
    if stage == "test":
        result["ckpt_path"] = ckpt_path
    return result


def execute_run(run: RunConfig, *, ray: bool = False) -> dict[str, Any]:
    """Own the run lifecycle; tracking and Ray consume the saved result."""
    writer = True
    if ray:
        from ray import train

        writer = train.get_context().get_world_rank() == 0
    logger = None
    mode = tracking_mode()
    if writer:
        write_config(run)
        record_status(run, "running", message="worker_started", tracking_mode=mode)
    try:
        if (
            run.resources.accelerator == "gpu"
            and run.payload.model.get("type") == "event_graph_mamba"
            and run.payload.model.get("backend", "official") == "official"
        ):
            from graphids.runtime_checks import assert_mamba_backend

            assert_mamba_backend()
        if writer and mode == "online":
            logger = make_logger(run)
        result = run_lightning(run, logger=logger, ray=ray)
        if writer:
            record_status(run, "finished", result=result)
        if logger is not None and logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FINISHED")
        return result
    except BaseException as exc:
        if writer:
            record_status(run, "failed", failure=f"{type(exc).__name__}: {exc}")
        if logger is not None and logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FAILED")
        raise


def _worker_loop(config: dict[str, Any]) -> None:
    from ray import train

    # Serialization and validation happen at the process boundary.
    run = RunConfig.model_validate(config["run"])
    result = execute_run(run, ray=True)
    train.report({**result["metrics"], "graphids_finished": 1.0})


def probe_ray_train_imports() -> dict[str, str]:
    """Import the Ray APIs GraphIDS relies on and return their module paths."""
    try:
        from ray.train import RunConfig as RayRunConfig
        from ray.train import ScalingConfig
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )
        from ray.train.torch import TorchTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ray is not installed. Install the project dependencies before launching experiments."
        ) from exc

    return {
        "RunConfig": RayRunConfig.__module__,
        "ScalingConfig": ScalingConfig.__module__,
        "TorchTrainer": TorchTrainer.__module__,
        "prepare_trainer": prepare_trainer.__module__,
        "RayDDPStrategy": RayDDPStrategy.__module__,
        "RayLightningEnvironment": RayLightningEnvironment.__module__,
        "RayTrainReportCallback": RayTrainReportCallback.__module__,
    }


def launch_run(run: RunConfig, *, address: str | None = None) -> RunSummary:
    """Launch the supported single-worker run in a local or allocated Ray cluster."""
    try:
        import ray
        from ray.train import RunConfig as RayRunConfig
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ray is not installed. Install the project dependencies before launching experiments."
        ) from exc

    try:
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True)
        resources = run.resources
        use_gpu = resources.accelerator == "gpu" or resources.gpus_per_worker > 0
        worker_resources = {"CPU": max(1, resources.cpus_per_worker)}
        if use_gpu:
            worker_resources["GPU"] = resources.gpus_per_worker or 1
        TorchTrainer(
            _worker_loop,
            train_loop_config={"run": run.model_dump(mode="json")},
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=use_gpu,
                resources_per_worker=worker_resources,
            ),
            run_config=RayRunConfig(
                storage_path=str(run.outputs.run_dir.parent),
                name=run.outputs.run_dir.name,
            ),
        ).fit()
    except BaseException as exc:
        manifest = load_manifest(run.outputs.run_dir, name=run.outputs.manifest_name)
        if manifest is None or manifest.status != "failed":
            write_config(run)
            record_status(
                run,
                "failed",
                failure=f"{type(exc).__name__}: {exc}",
                tracking_mode=tracking_mode(),
            )
        raise
    summary = summarize_run(run.outputs.run_dir)
    if summary is None:
        raise RuntimeError(
            f"Ray run finished without a GraphIDS manifest: {run.outputs.run_dir}"
        )
    return summary
