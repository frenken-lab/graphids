"""Tests for the new experiment manifest + event journal seam."""

from __future__ import annotations

from typer.testing import CliRunner


def test_mlflow_system_metrics_callback_lifecycle(monkeypatch):
    import sys
    import types

    from graphids import _mlflow

    events: list[tuple[str, object]] = []

    class FakeMonitor:
        def __init__(self, run_id, **kwargs):
            events.append(("init", run_id, kwargs))

        def start(self):
            events.append(("start", None))

        def finish(self):
            events.append(("finish", None))

    monitor_mod = types.ModuleType("mlflow.system_metrics.system_metrics_monitor")
    monitor_mod.SystemMetricsMonitor = FakeMonitor
    system_metrics_mod = types.ModuleType("mlflow.system_metrics")
    system_metrics_mod.system_metrics_monitor = monitor_mod
    monkeypatch.setitem(sys.modules, "mlflow.system_metrics", system_metrics_mod)
    monkeypatch.setitem(sys.modules, "mlflow.system_metrics.system_metrics_monitor", monitor_mod)
    monkeypatch.setattr(_mlflow.mlflow, "get_tracking_uri", lambda: "sqlite:////tmp/mlflow.db")

    trainer = types.SimpleNamespace(logger=types.SimpleNamespace(run_id="abc123"))
    cb = _mlflow.MLflowSystemMetricsCallback(sampling_interval=7, samples_before_logging=2)

    cb.on_train_start(trainer, object())
    cb.on_fit_end(trainer, object())
    cb.on_test_start(trainer, object())
    cb.on_exception(trainer, object(), RuntimeError("boom"))

    assert events == [
        (
            "init",
            "abc123",
            {
                "sampling_interval": 7,
                "samples_before_logging": 2,
                "tracking_uri": "sqlite:////tmp/mlflow.db",
            },
        ),
        ("start", None),
        ("finish", None),
        (
            "init",
            "abc123",
            {
                "sampling_interval": 7,
                "samples_before_logging": 2,
                "tracking_uri": "sqlite:////tmp/mlflow.db",
            },
        ),
        ("start", None),
        ("finish", None),
    ]


def test_run_stage_attaches_launch_logger(monkeypatch, tmp_path):
    import sys
    import types

    from graphids._mlflow import MLflowSystemMetricsCallback
    from graphids.exp import runtime
    from graphids.exp.config import (
        FitRunPayload,
        OutputConfig,
        ResourceConfig,
        RunConfig,
    )

    captured: dict[str, object] = {}

    class DummyData:
        def setup(self, stage):
            captured["setup_stage"] = stage

    class DummyModel:
        def prepare_from_datamodule(self, data):
            captured["prepared_with"] = data

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def fit(self, model, datamodule):
            captured["fit_model"] = model
            captured["fit_datamodule"] = datamodule

    class DummyLightning:
        Trainer = DummyTrainer

        @staticmethod
        def seed_everything(seed, workers):
            captured["seed"] = seed
            captured["workers"] = workers

    lightning_pkg = types.ModuleType("lightning")
    lightning_pkg.__path__ = []
    lightning_pkg.pytorch = DummyLightning
    monkeypatch.setitem(sys.modules, "lightning", lightning_pkg)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", DummyLightning)
    monkeypatch.setattr(runtime, "_build_component", lambda spec: DummyData())
    monkeypatch.setattr(runtime, "_resolve_spec", lambda spec: {} if spec == {} else DummyModel())

    logger = object()
    run = RunConfig(
        name="demo",
        stage="fit",
        dataset="hcrl_sa",
        payload=FitRunPayload(
            model={"type": "dummy_model"},
            data={"type": "dummy_data"},
            seed_everything=123,
        ),
        resources=ResourceConfig(),
        outputs=OutputConfig(run_dir=tmp_path / "run"),
    )

    result = runtime.run_stage(run, logger=logger)

    assert result == {"stage": "fit", "trainer": "DummyTrainer"}
    assert captured["trainer_kwargs"]["logger"] is logger
    callbacks = captured["trainer_kwargs"]["callbacks"]
    assert any(isinstance(cb, MLflowSystemMetricsCallback) for cb in callbacks)
    assert captured["seed"] == 123
    assert captured["workers"] is True


def test_ray_stage_binds_existing_mlflow_run(monkeypatch, tmp_path):
    from graphids.exp import runtime
    from graphids.exp.config import (
        FitRunPayload,
        OutputConfig,
        ResourceConfig,
        RunConfig,
    )

    captured: dict[str, object] = {}

    def fake_make_logger(**kwargs):
        captured["logger_kwargs"] = kwargs
        return object()

    def fake_run_stage(run, logger=None):
        captured["run"] = run
        captured["logger"] = logger
        return {"stage": run.stage}

    monkeypatch.setattr(runtime, "make_logger", fake_make_logger)
    monkeypatch.setattr(runtime, "run_stage", fake_run_stage)

    run = RunConfig(
        name="demo",
        stage="fit",
        dataset="hcrl_sa",
        payload=FitRunPayload(
            model={"type": "dummy_model"},
            data={"type": "dummy_data"},
        ),
        resources=ResourceConfig(backend="ray"),
        outputs=OutputConfig(run_dir=tmp_path / "run"),
    )

    result = runtime._run_stage_with_existing_mlflow_run(run, "existing-run-id")

    assert result == {"stage": "fit"}
    assert captured["run"] is run
    assert captured["logger"] is not None
    assert captured["logger_kwargs"]["run_id"] == "existing-run-id"
    assert captured["logger_kwargs"]["experiment_name"] == "graphids/hcrl_sa/fit"


def test_run_stage_cache_builds_data_only(monkeypatch, tmp_path):
    from graphids.exp import runtime
    from graphids.exp.config import (
        CacheRunPayload,
        OutputConfig,
        ResourceConfig,
        RunConfig,
    )

    captured: dict[str, object] = {}

    class DummySource:
        cache_key = "dummy-cache-key"

        def cache_root_path(self):
            return tmp_path / "cache-root"

        def cache_ready(self):
            return True

    class DummyData:
        source = DummySource()

        def setup(self, stage):
            captured["setup_stage"] = stage

    monkeypatch.setattr(runtime, "_build_component", lambda spec: DummyData())

    run = RunConfig(
        name="cache-demo",
        stage="cache",
        dataset="set_01",
        payload=CacheRunPayload(data={"type": "dummy_data"}),
        resources=ResourceConfig(),
        outputs=OutputConfig(run_dir=tmp_path / "run"),
    )

    assert runtime.run_stage(run) == {
        "stage": "cache",
        "cache_key": "dummy-cache-key",
        "cache_root": str(tmp_path / "cache-root"),
        "cache_ready": True,
    }
    assert captured["setup_stage"] is None


def test_manifest_and_events_round_trip(tmp_path):
    from graphids.exp.config import (
        FitRunPayload,
        OutputConfig,
        ResourceConfig,
        RunConfig,
    )
    from graphids.exp.journal import (
        EventRecord,
        RunManifest,
        append_event,
        load_events,
        load_manifest,
        write_manifest,
    )

    run_dir = tmp_path / "run"
    run = RunConfig(
        name="demo",
        stage="fit",
        dataset="hcrl_sa",
        seed=42,
        git_sha="abc123",
        payload=FitRunPayload(
            model={"class_path": "graphids.primitives_models.GATCfg"},
            data={"class_path": "graphids.primitives_data.CANBusCfg"},
        ),
        resources=ResourceConfig(),
        outputs=OutputConfig(run_dir=str(run_dir)),
    )

    manifest = RunManifest(
        run_id=run.name,
        name=run.name,
        stage=run.stage,
        git_sha=run.git_sha,
        run_dir=str(run.outputs.run_dir),
        config={"payload": run.payload.model_dump(mode="json")},
        outputs={"run_dir": str(run.outputs.run_dir)},
    )
    write_manifest(run.outputs.run_dir, manifest)
    append_event(run.outputs.run_dir, EventRecord(status="running", stage="fit", message="launch_started"))
    append_event(run.outputs.run_dir, EventRecord(status="finished", stage="fit", message="fit_finished"))

    loaded = load_manifest(run.outputs.run_dir)
    events = load_events(run.outputs.run_dir)
    assert loaded is not None
    assert loaded.name == "demo"
    assert loaded.status == "created"
    assert [e.message for e in events] == ["launch_started", "fit_finished"]


def test_exp_status_prints_summary(tmp_path):
    from graphids.exp.journal import (
        EventRecord,
        RunManifest,
        append_event,
        write_manifest,
    )

    run_dir = tmp_path / "run"
    manifest = RunManifest(
        run_id="demo",
        name="demo",
        stage="fit",
        git_sha="abc123",
        run_dir=str(run_dir),
        config={},
        outputs={"run_dir": str(run_dir)},
        status="running",
    )
    write_manifest(run_dir, manifest)
    append_event(run_dir, EventRecord(status="failed", stage="fit", message="fit_failed"))

    runner = CliRunner()
    from graphids.cli.app import app

    result = runner.invoke(app, ["exp", "status", str(run_dir)])
    assert result.exit_code == 0, result.stderr
    assert "demo" in result.stdout
    assert "fit_failed" in result.stdout


def test_exp_launch_loads_yaml_and_invokes_runtime(monkeypatch, tmp_path):
    import graphids.cli.exp as exp_cli
    import graphids.paths as paths
    from graphids.cli.app import app
    from graphids.exp.config import RunSummary

    captured: dict[str, object] = {}
    run_root = tmp_path / "runs"
    monkeypatch.setattr(paths, "trial_dir", lambda: run_root)

    def fake_launch_run(run):
        captured["run"] = run
        return RunSummary(
            run_dir=str(run.outputs.run_dir),
            status="finished",
            stage=run.stage,
            name=run.name,
            last_event="run_finished",
        )

    monkeypatch.setattr(exp_cli, "launch_run", fake_launch_run)

    cfg = tmp_path / "experiment.yaml"
    cfg.write_text(
        """
experiment_name: smoke
dataset: hcrl_sa
seed: 123
stage: fit
resources:
  backend: local
config:
  seed_everything: 123
  model:
    type: dummy_model
  data:
    type: dummy_data
  trainer:
    max_epochs: 1
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(app, ["exp", "launch", str(cfg)])

    assert result.exit_code == 0, result.stderr
    assert '"status": "finished"' in result.stdout
    run = captured["run"]
    assert run.name == "smoke"
    assert run.stage == "fit"
    assert run.dataset == "hcrl_sa"
    assert run.seed == 123
    assert run.payload.model == {"type": "dummy_model"}
    assert run.payload.trainer == {"max_epochs": 1}
    assert run.outputs.run_dir == run_root / "hcrl_sa" / "smoke" / "smoke"
