from __future__ import annotations

import pytest


def test_launch_run_builds_real_ray_trainer_config(monkeypatch, tmp_path):
    import ray
    import ray.train
    import ray.train.torch

    from graphids.exp import run as runner
    from graphids.exp.config import (
        OutputConfig,
        ResourceConfig,
        RunConfig,
        RunPayload,
    )
    from graphids.exp.records import RunManifest, RunSummary, write_manifest

    captured: dict[str, object] = {}

    class FakeScalingConfig:
        def __init__(self, **kwargs):
            captured["scaling_config"] = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs):
            captured["run_config"] = kwargs

    class FakeTorchTrainer:
        def __init__(self, train_loop_per_worker, **kwargs):
            captured["train_loop_per_worker"] = train_loop_per_worker
            captured["trainer_kwargs"] = kwargs

        def fit(self):
            return type(
                "FakeResult",
                (),
                {"path": "/tmp/ray-result", "metrics": {"val_loss": 1.25}},
            )()

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        ray, "init", lambda **kwargs: captured.setdefault("ray_init", kwargs)
    )
    monkeypatch.setattr(ray.train, "ScalingConfig", FakeScalingConfig)
    monkeypatch.setattr(ray.train, "RunConfig", FakeRunConfig)
    monkeypatch.setattr(ray.train.torch, "TorchTrainer", FakeTorchTrainer)
    monkeypatch.setattr(
        runner,
        "summarize_run",
        lambda run_dir: RunSummary(
            run_dir=str(run_dir), status="finished", stage="fit", name="ray-demo"
        ),
    )

    run = RunConfig(
        name="ray-demo",
        stage="fit",
        resources=ResourceConfig(
            accelerator="gpu", cpus_per_worker=4, gpus_per_worker=1.0
        ),
        outputs=OutputConfig(run_dir=tmp_path / "ray-demo"),
        payload=RunPayload(
            trainer={"devices": 1},
            callbacks={
                "best": {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 3,
                }
            },
        ),
    )
    write_manifest(
        run.outputs.run_dir,
        RunManifest(
            run_id=run.name,
            name=run.name,
            stage=run.stage,
            git_sha=run.git_sha,
            run_dir=str(run.outputs.run_dir),
        ),
    )

    summary = runner.launch_run(run, address="local")

    assert summary.status == "finished"
    assert captured["ray_init"] == {"address": "local", "ignore_reinit_error": True}
    assert captured["scaling_config"] == {
        "num_workers": 1,
        "use_gpu": True,
        "resources_per_worker": {"CPU": 4, "GPU": 1.0},
    }
    ray_run_config = captured["run_config"]
    assert ray_run_config["storage_path"] == str(run.outputs.run_dir.parent)
    assert ray_run_config["name"] == run.outputs.run_dir.name
    assert "checkpoint_config" not in ray_run_config
    assert captured["trainer_kwargs"]["train_loop_config"]["run"]["name"] == "ray-demo"


@pytest.mark.parametrize(
    "ray_enabled, stage, with_logger",
    [
        (False, "fit", True),
        (False, "test", False),
        (True, "fit", False),
        (True, "test", True),
    ],
)
def test_build_lightning_run_wires_components_compile_logger_and_ray(
    monkeypatch, tmp_path, ray_enabled, stage, with_logger
):
    """Pairwise coverage of stage, logger, and Ray preserves runtime wiring."""
    import sys
    import types

    from lightning.pytorch.callbacks import ModelCheckpoint

    from graphids._mlflow import MLflowSystemMetricsCallback
    from graphids.exp import run as runner

    captured = {}

    class DummyData:
        def setup(self, stage):
            captured["setup_stage"] = stage

    class DummyModel:
        def prepare_from_datamodule(self, data):
            captured["prepared_data"] = data

        def compile(self, **kwargs):
            captured["compile_kwargs"] = kwargs

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.callback_metrics = {"quality": 0.75}
            self.global_step = 3

        def fit(self, model, datamodule, ckpt_path=None):
            captured["run"] = ("fit", model, datamodule, ckpt_path)

        def test(self, model, datamodule, ckpt_path=None):
            captured["run"] = ("test", model, datamodule, ckpt_path)

    data, model = DummyData(), DummyModel()
    checkpoint = ModelCheckpoint(dirpath=tmp_path) if stage == "fit" else None
    logger = (
        types.SimpleNamespace(
            log_metrics=lambda metrics, step: captured.update(logged=(metrics, step))
        )
        if with_logger
        else None
    )
    components = {"dummy_data": data, "dummy_model": model, "checkpoint": checkpoint}
    monkeypatch.setattr(
        runner, "build_component", lambda spec: components[spec["type"]]
    )
    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    pytorch.Trainer = DummyTrainer
    pytorch.callbacks = types.SimpleNamespace(ModelCheckpoint=ModelCheckpoint)
    pytorch.seed_everything = lambda seed, workers: captured.update(
        seed=seed, workers=workers
    )
    lightning.pytorch = pytorch
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pytorch)

    class FakeRayDDPStrategy:
        pass

    class FakeRayLightningEnvironment:
        pass

    class FakeRayTrainReportCallback:
        pass

    if ray_enabled:
        import ray.train

        ray_lightning = types.ModuleType("ray.train.lightning")
        ray_lightning.RayDDPStrategy = FakeRayDDPStrategy
        ray_lightning.RayLightningEnvironment = FakeRayLightningEnvironment
        ray_lightning.RayTrainReportCallback = FakeRayTrainReportCallback
        ray_lightning.prepare_trainer = lambda trainer: captured.setdefault(
            "prepared_trainer", trainer
        )
        monkeypatch.setitem(sys.modules, "ray.train.lightning", ray_lightning)
        monkeypatch.setattr(
            ray.train,
            "get_context",
            lambda: types.SimpleNamespace(get_world_rank=lambda: 0),
        )

    from graphids.exp.config import OutputConfig, RunConfig, RunPayload

    payload = {
        "data": {"type": "dummy_data"},
        "model": {"type": "dummy_model"},
        "trainer": {"max_epochs": 1},
        "callbacks": {"checkpoint": {"type": "checkpoint"}} if checkpoint else {},
        "seed_everything": 7,
        "ckpt_path": "resume.ckpt",
        "compile": {"backend": "eager", "fullgraph": False},
    }
    result = runner.run_lightning(
        RunConfig(
            name="wiring",
            stage=stage,
            payload=RunPayload.model_validate(payload),
            outputs=OutputConfig(run_dir=tmp_path),
        ),
        logger=logger,
        ray=ray_enabled,
    )
    expected = {"stage": stage, "trainer": "DummyTrainer", "metrics": {"quality": 0.75}}
    if stage == "test":
        expected["ckpt_path"] = "resume.ckpt"
    assert result == expected
    assert captured["setup_stage"] is None
    assert captured["prepared_data"] is data
    assert captured["run"] == (stage, model, data, "resume.ckpt")
    if with_logger:
        assert captured["logged"] == ({"quality": 0.75}, 3)
    else:
        assert "logged" not in captured
    assert captured["seed"] == 7 and captured["workers"] is True
    assert captured["compile_kwargs"] == {"backend": "eager", "fullgraph": False}
    kwargs = captured["trainer_kwargs"]
    assert kwargs["logger"] is (logger if with_logger else False)
    assert (
        any(isinstance(cb, MLflowSystemMetricsCallback) for cb in kwargs["callbacks"])
        is with_logger
    )
    if ray_enabled:
        assert kwargs["enable_checkpointing"] is (checkpoint is not None)
        assert kwargs["accelerator"] == kwargs["devices"] == "auto"
        assert isinstance(kwargs["strategy"], FakeRayDDPStrategy)
        assert any(
            isinstance(p, FakeRayLightningEnvironment) for p in kwargs["plugins"]
        )
        assert any(
            isinstance(cb, FakeRayTrainReportCallback) for cb in kwargs["callbacks"]
        )
        assert captured["prepared_trainer"] is not None


def test_execute_run_saves_one_authoritative_result_and_reports_worker_metrics(
    monkeypatch, tmp_path
):
    import ray.train

    from graphids.exp import run as runner
    from graphids.exp.config import OutputConfig, RunConfig
    from graphids.exp.records import load_events, load_manifest
    from graphids.exp.tracking import load_ingest_payload

    run = RunConfig(
        name="one-result", stage="fit", outputs=OutputConfig(run_dir=tmp_path)
    )
    result = dict(stage="fit", trainer="fixture", metrics={"val/ap": 0.75})
    monkeypatch.setenv("GRAPHIDS_MLFLOW_MODE", "offline")
    monkeypatch.setattr(runner, "run_lightning", lambda actual, **kwargs: result)
    monkeypatch.setattr(
        ray.train,
        "get_context",
        lambda: type("Context", (), {"get_world_rank": lambda self: 0})(),
    )
    reported = []
    monkeypatch.setattr(ray.train, "report", reported.append)
    runner._worker_loop({"run": run.model_dump(mode="json")})
    manifest = load_manifest(tmp_path)
    assert manifest.schema_version == 2
    assert manifest.status == "finished"
    assert manifest.metrics == {"val/ap": 0.75}
    assert manifest.result == {"stage": "fit", "trainer": "fixture"}
    assert [event.status for event in load_events(tmp_path)] == ["running", "finished"]
    assert not (tmp_path / ".graphids/mlflow_ingest.json").exists()
    assert load_ingest_payload(tmp_path)["metrics"] == manifest.metrics
    assert reported == [{"val/ap": 0.75, "graphids_finished": 1.0}]


@pytest.mark.parametrize("failure_at", ["logger", "lightning"])
def test_startup_and_training_errors_leave_failed_records(
    monkeypatch, tmp_path, failure_at
):
    from graphids.exp import run as runner
    from graphids.exp.config import OutputConfig, RunConfig
    from graphids.exp.records import load_manifest
    from graphids.exp.tracking import load_ingest_payload

    error = RuntimeError(f"failure in {failure_at}")

    def broken(*args, **kwargs):
        raise error

    run = RunConfig(name="failed", stage="fit", outputs=OutputConfig(run_dir=tmp_path))
    monkeypatch.setenv(
        "GRAPHIDS_MLFLOW_MODE", "online" if failure_at == "logger" else "offline"
    )
    monkeypatch.setattr(
        runner, "make_logger" if failure_at == "logger" else "run_lightning", broken
    )
    with pytest.raises(RuntimeError) as caught:
        runner.execute_run(run)
    assert caught.value is error
    manifest = load_manifest(tmp_path)
    assert manifest.status == "failed"
    assert manifest.failure == f"RuntimeError: {error}"
    assert load_ingest_payload(tmp_path)["status"] == "FAILED"


def test_ray_initialization_failure_is_recorded(monkeypatch, tmp_path):
    import ray

    from graphids.exp.config import OutputConfig, RunConfig
    from graphids.exp.records import load_manifest
    from graphids.exp.run import launch_run

    monkeypatch.setattr(ray, "is_initialized", lambda: False)

    def broken(**kwargs):
        raise RuntimeError("cluster unavailable")

    monkeypatch.setattr(ray, "init", broken)
    run = RunConfig(name="startup", stage="fit", outputs=OutputConfig(run_dir=tmp_path))
    with pytest.raises(RuntimeError, match="cluster unavailable"):
        launch_run(run)
    assert load_manifest(tmp_path).status == "failed"


@pytest.mark.parametrize("mode", ["disabled", "off", "none"])
def test_disabled_tracking_keeps_results_local(monkeypatch, tmp_path, mode):
    from graphids.exp import run as runner
    from graphids.exp.config import OutputConfig, RunConfig
    from graphids.exp.records import load_manifest
    from graphids.exp.tracking import discover_runs, ingest_many

    monkeypatch.setenv("GRAPHIDS_MLFLOW_MODE", mode)
    monkeypatch.setattr(
        runner, "run_lightning", lambda *args, **kwargs: {"metrics": {"ap": 0.5}}
    )
    run = RunConfig(
        name="local", stage="fit", outputs=OutputConfig(run_dir=tmp_path / "run")
    )
    runner.execute_run(run)
    assert load_manifest(run.outputs.run_dir).tracking_mode == "disabled"
    results = ingest_many(discover_runs(tmp_path))
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].run_id is None
