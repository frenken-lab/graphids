from __future__ import annotations

import pytest


def test_launch_run_builds_real_ray_trainer_config(monkeypatch, tmp_path):
    import ray
    import ray.train
    import ray.train.torch

    from graphids.exp import ray_backend
    from graphids.exp.config import (
        FitRunPayload,
        OutputConfig,
        ResourceConfig,
        RunConfig,
    )
    from graphids.exp.journal import RunManifest, write_manifest
    from graphids.exp.ray_backend import RunSummary

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
        ray_backend,
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
        payload=FitRunPayload(
            trainer={"devices": 2},
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

    summary = ray_backend.launch_run(run, address="local")

    assert summary.status == "finished"
    assert captured["ray_init"] == {"address": "local", "ignore_reinit_error": True}
    assert captured["scaling_config"] == {
        "num_workers": 2,
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
    monkeypatch, ray_enabled, stage, with_logger
):
    """Pairwise coverage of stage, logger, and Ray preserves runtime wiring."""
    import sys
    import types

    from graphids._mlflow import MLflowSystemMetricsCallback
    from graphids.exp import ray_backend, runtime

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
    logger = (
        types.SimpleNamespace(
            log_metrics=lambda metrics, step: captured.update(logged=(metrics, step))
        )
        if with_logger
        else None
    )
    monkeypatch.setattr(
        runtime,
        "build_component",
        lambda spec: data if spec["type"] == "dummy_data" else model,
    )
    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    pytorch.Trainer = DummyTrainer
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

    result = ray_backend._build_lightning_run(
        stage,
        {
            "data": {"type": "dummy_data"},
            "model": {"type": "dummy_model"},
            "trainer": {"max_epochs": 1},
            "seed_everything": 7,
            "ckpt_path": "resume.ckpt",
            "compile": {"backend": "eager", "fullgraph": False},
        },
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
        assert kwargs["accelerator"] == kwargs["devices"] == "auto"
        assert isinstance(kwargs["strategy"], FakeRayDDPStrategy)
        assert any(
            isinstance(p, FakeRayLightningEnvironment) for p in kwargs["plugins"]
        )
        assert any(
            isinstance(cb, FakeRayTrainReportCallback) for cb in kwargs["callbacks"]
        )
        assert captured["prepared_trainer"] is not None
