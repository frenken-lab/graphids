"""Ray Train launcher for GraphIDS experiment YAML."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from graphids.exp import runtime
from graphids.exp.config import RunConfig
from graphids.exp.journal import (
    EventRecord,
    append_event,
    load_events,
    load_manifest,
    write_manifest,
)
from graphids.exp.runtime import (
    _compile_model_if_requested as _compile_model_if_requested,
)
from graphids.exp.runtime import (
    _scalar_metrics as _scalar_metrics,
)
from graphids.exp.runtime import (
    build_component as build_component,
)
from graphids.exp.runtime import (
    build_model,
    seed_run,
    trainer_components,
)
from graphids.exp.runtime import (
    resolve_spec as resolve_spec,
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

    seed_run(cfg)
    data = runtime.build_component(cfg["data"])
    if hasattr(data, "setup"):
        data.setup(None)
    model = build_model(cfg, data)
    trainer_kwargs = trainer_components(cfg)
    if os.environ.get("GRAPHIDS_RUN_DIR"):
        trainer_kwargs.setdefault("default_root_dir", os.environ["GRAPHIDS_RUN_DIR"])
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
        trainer_kwargs.setdefault("enable_checkpointing", False)
    trainer_kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(**trainer_kwargs)
    if ray:
        trainer = prepare_trainer(trainer)

    if stage == "fit":
        trainer.fit(model, datamodule=data)
        metrics = _scalar_metrics(trainer.callback_metrics)
        if logger is not None and metrics:
            logger.log_metrics(metrics, step=trainer.global_step)
        return {
            "stage": "fit",
            "trainer": trainer.__class__.__name__,
            "metrics": metrics,
        }

    if stage == "test":
        ckpt_path = cfg.get("ckpt_path") or None
        trainer.test(model, datamodule=data, ckpt_path=ckpt_path)
        metrics = _scalar_metrics(trainer.callback_metrics)
        if logger is not None and metrics:
            logger.log_metrics(metrics, step=trainer.global_step)
        return {
            "stage": "test",
            "trainer": trainer.__class__.__name__,
            "ckpt_path": ckpt_path,
            "metrics": metrics,
        }

    raise ValueError(f"unknown stage: {stage!r}")


def _worker_loop(config: dict[str, Any]) -> None:
    from ray import train

    from graphids.exp.ingest import load_ingest_payload, write_ingest_payload

    mode = _tracking_mode()
    if mode not in {"disabled", "online"}:
        os.environ["GRAPHIDS_MLFLOW_MODE"] = "offline"
    run = RunConfig.model_validate(config["run"])
    os.environ["GRAPHIDS_RUN_DIR"] = str(run.outputs.run_dir)
    writer = train.get_context().get_world_rank() == 0
    if writer:
        run.write_config()

    if run.stage in {"fit", "test"} and run.resources.accelerator == "gpu":
        if run.payload.model.get("type") == "vehicle_graph_mamba_joint":
            from graphids.runtime_checks import assert_mamba_backend

            if run.payload.model.get("backend", "official") == "official":
                assert_mamba_backend()

    logger = _make_logger(run) if mode == "online" and writer else None
    if logger is not None:
        logger.log_hyperparams(run.mlflow_hparams())
    elif mode == "offline" and writer:
        write_ingest_payload(run, status="RUNNING")

    manifest = run.journal_manifest(status="running")
    if writer:
        write_manifest(run.outputs.run_dir, manifest, name=run.outputs.manifest_name)
        append_event(
            run.outputs.run_dir,
            EventRecord(status="running", stage=run.stage, message="worker_started"),
        )

    try:
        result = _build_lightning_run(
            run.stage, run.payload.model_dump(mode="json"), logger=logger, ray=True
        )
        metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
        if logger is not None and logger.run_id is not None:
            logger.experiment.set_terminated(logger.run_id, status="FINISHED")
        elif mode == "offline" and writer:
            write_ingest_payload(
                run, status="FINISHED", metrics=metrics, result=_jsonish(result)
            )

        if writer:
            append_event(
                run.outputs.run_dir,
                EventRecord(
                    status="finished",
                    stage=run.stage,
                    message="run_finished",
                    details=_jsonish(result),
                ),
                name=run.outputs.events_name,
            )
            write_manifest(
                run.outputs.run_dir,
                manifest.model_copy(update={"status": "finished"}),
                name=run.outputs.manifest_name,
            )
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
        elif mode == "offline" and writer:
            write_ingest_payload(run, status="FAILED", failure=failure)
        if writer:
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
    """Run a GraphIDS experiment through Ray Train."""
    try:
        import ray
        from ray.train import RunConfig as RayRunConfig
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ray is not installed. Install the project dependencies before launching experiments."
        ) from exc

    if not ray.is_initialized():
        ray.init(address=address, ignore_reinit_error=True)

    resources = run.resources
    devices = run.payload.trainer.get("devices")
    num_workers = devices if isinstance(devices, int) and devices > 1 else 1
    use_gpu = resources.accelerator == "gpu" or float(resources.gpus_per_worker) > 0
    resources_per_worker: dict[str, float | int] = {
        "CPU": max(1, int(resources.cpus_per_worker))
    }
    if float(resources.gpus_per_worker) > 0:
        resources_per_worker["GPU"] = float(resources.gpus_per_worker)

    run_dir = Path(run.outputs.run_dir)
    ray_run_kwargs: dict[str, Any] = {
        "storage_path": str(run_dir.parent),
        "name": run_dir.name,
    }

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
            details={
                "path": str(getattr(result, "path", "") or ""),
                "metrics": _jsonish(getattr(result, "metrics", {}) or {}),
            },
        ),
        name=run.outputs.events_name,
    )
    summary = summarize_run(run.outputs.run_dir)
    if summary is None:
        raise RuntimeError(
            f"Ray run finished without a GraphIDS manifest: {run.outputs.run_dir}"
        )
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
