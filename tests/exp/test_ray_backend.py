from __future__ import annotations


def test_launch_run_builds_real_ray_trainer_config(monkeypatch, tmp_path):
    import ray
    import ray.train
    import ray.train.torch

    from graphids.exp import ray_backend, runtime
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
            return type("FakeResult", (), {"path": "/tmp/ray-result", "metrics": {"val_loss": 1.25}})()

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.setdefault("ray_init", kwargs))
    monkeypatch.setattr(ray.train, "ScalingConfig", FakeScalingConfig)
    monkeypatch.setattr(ray.train, "RunConfig", FakeRunConfig)
    monkeypatch.setattr(ray.train.torch, "TorchTrainer", FakeTorchTrainer)
    monkeypatch.setattr(
        ray_backend,
        "summarize_run",
        lambda run_dir: RunSummary(run_dir=str(run_dir), status="finished", stage="fit", name="ray-demo"),
    )

    run = RunConfig(
        name="ray-demo",
        stage="fit",
        resources=ResourceConfig(accelerator="gpu", cpus_per_worker=4, gpus_per_worker=1.0),
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


def test_build_lightning_run_configures_lightning_for_ray(monkeypatch):
    import sys
    import types

    from graphids.exp import ray_backend, runtime

    captured: dict[str, object] = {}

    class DummyData:
        def setup(self, stage):
            captured["setup_stage"] = stage

    class DummyModel:
        pass

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.callback_metrics = {}
            self.global_step = 0

        def fit(self, model, datamodule, ckpt_path=None):
            captured["fit"] = (model, datamodule)

    class DummyLightning:
        Trainer = DummyTrainer

        @staticmethod
        def seed_everything(seed, workers):
            captured["seed"] = seed

    class FakeRayDDPStrategy:
        pass

    class FakeRayLightningEnvironment:
        pass

    class FakeRayTrainReportCallback:
        pass

    lightning_pkg = types.ModuleType("lightning")
    lightning_pkg.__path__ = []
    lightning_pkg.pytorch = DummyLightning
    ray_lightning_mod = types.ModuleType("ray.train.lightning")
    ray_lightning_mod.RayDDPStrategy = FakeRayDDPStrategy
    ray_lightning_mod.RayLightningEnvironment = FakeRayLightningEnvironment
    ray_lightning_mod.RayTrainReportCallback = FakeRayTrainReportCallback
    ray_lightning_mod.prepare_trainer = lambda trainer: captured.setdefault("prepared", trainer)

    monkeypatch.setitem(sys.modules, "lightning", lightning_pkg)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", DummyLightning)
    monkeypatch.setitem(sys.modules, "ray.train.lightning", ray_lightning_mod)
    monkeypatch.setattr(runtime, "build_component", lambda spec: DummyData())

    def resolve_spec(spec):
        if isinstance(spec, dict) and spec.get("type") == "dummy_model":
            return DummyModel()
        return spec

    monkeypatch.setattr(runtime, "resolve_spec", resolve_spec)

    result = ray_backend._build_lightning_run(
        "fit",
        {
            "data": {"type": "dummy_data"},
            "model": {"type": "dummy_model"},
            "trainer": {"max_epochs": 1},
            "seed_everything": 7,
        },
        logger=None,
        ray=True,
    )

    assert result == {"stage": "fit", "trainer": "DummyTrainer", "metrics": {}}
    kwargs = captured["trainer_kwargs"]
    assert kwargs["accelerator"] == "auto"
    assert kwargs["devices"] == "auto"
    assert isinstance(kwargs["strategy"], FakeRayDDPStrategy)
    assert any(isinstance(plugin, FakeRayLightningEnvironment) for plugin in kwargs["plugins"])
    assert any(isinstance(callback, FakeRayTrainReportCallback) for callback in kwargs["callbacks"])
    assert captured["prepared"] is not None


def test_build_lightning_run_uses_lightning_module_compile(monkeypatch):
    import sys
    import types

    from graphids.exp import ray_backend, runtime

    captured: dict[str, object] = {}

    class DummyData:
        def setup(self, stage):
            captured["setup_stage"] = stage

    class DummyModel:
        def compile(self, **kwargs):
            captured["compile_kwargs"] = kwargs

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.callback_metrics = {}
            self.global_step = 0

        def fit(self, model, datamodule, ckpt_path=None):
            captured["fit"] = (model, datamodule)

    class DummyLightning:
        Trainer = DummyTrainer

        @staticmethod
        def seed_everything(seed, workers):
            del seed, workers

    lightning_pkg = types.ModuleType("lightning")
    lightning_pkg.__path__ = []
    lightning_pkg.pytorch = DummyLightning
    monkeypatch.setitem(sys.modules, "lightning", lightning_pkg)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", DummyLightning)
    monkeypatch.setattr(runtime, "build_component", lambda spec: DummyData())

    def resolve_spec(spec):
        if isinstance(spec, dict) and spec.get("type") == "dummy_model":
            return DummyModel()
        return spec

    monkeypatch.setattr(runtime, "resolve_spec", resolve_spec)

    ray_backend._build_lightning_run(
        "fit",
        {
            "data": {"type": "dummy_data"},
            "model": {"type": "dummy_model"},
            "compile": {
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": False,
            },
        },
        logger=None,
        ray=False,
    )

    assert captured["compile_kwargs"] == {
        "backend": "inductor",
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": False,
    }
    assert captured["fit"][0].__class__.__name__ == "DummyModel"
