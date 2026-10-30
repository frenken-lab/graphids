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
    monkeypatch.setitem(
        sys.modules, "mlflow.system_metrics.system_metrics_monitor", monitor_mod
    )
    monkeypatch.setattr(
        _mlflow.mlflow, "get_tracking_uri", lambda: "sqlite:////tmp/mlflow.db"
    )

    trainer = types.SimpleNamespace(
        logger=types.SimpleNamespace(run_id="abc123"),
        default_root_dir="/tmp/graphids-mlflow-test",
        global_step=0,
    )
    cb = _mlflow.MLflowSystemMetricsCallback(
        sampling_interval=7, samples_before_logging=2
    )

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


def test_manifest_and_events_round_trip(tmp_path):
    from graphids.exp.config import (
        OutputConfig,
        ResourceConfig,
        RunConfig,
        RunPayload,
    )
    from graphids.exp.records import (
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
        payload=RunPayload(
            model={"class_path": "graphids.models.config.EventGraphMambaConfig"},
            data={"class_path": "graphids.data.config.CANEventDataConfig"},
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
    append_event(
        run.outputs.run_dir,
        EventRecord(status="running", stage="fit", message="worker_started"),
    )
    append_event(
        run.outputs.run_dir,
        EventRecord(status="finished", stage="fit", message="fit_finished"),
    )

    loaded = load_manifest(run.outputs.run_dir)
    events = load_events(run.outputs.run_dir)
    assert loaded is not None
    assert loaded.name == "demo"
    assert loaded.status == "created"
    assert [e.message for e in events] == ["worker_started", "fit_finished"]


def test_exp_status_prints_summary(tmp_path):
    from graphids.exp.records import (
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
    append_event(
        run_dir, EventRecord(status="failed", stage="fit", message="fit_failed")
    )

    runner = CliRunner()
    from graphids.cli.app import app

    result = runner.invoke(app, ["exp", "status", str(run_dir)])
    assert result.exit_code == 0, result.stderr
    assert "demo" in result.stdout
    assert "fit_failed" in result.stdout


def test_exp_launch_loads_yaml_and_invokes_ray_backend(monkeypatch, tmp_path):
    import graphids.cli.exp as exp_cli
    import graphids.paths as paths
    from graphids.cli.app import app
    from graphids.exp.records import RunSummary

    captured: dict[str, object] = {}
    run_root = tmp_path / "runs"
    monkeypatch.setattr(paths, "trial_dir", lambda: run_root)

    def fake_launch_run(run, *, address=None):
        captured["run"] = run
        captured["address"] = address
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
    assert captured["address"] is None
    assert run.name == "smoke"
    assert run.stage == "fit"
    assert run.dataset == "hcrl_sa"
    assert run.seed == 123
    assert run.payload.model == {"type": "dummy_model"}
    assert run.payload.trainer == {"max_epochs": 1}
    assert run.outputs.run_dir == run_root / "hcrl_sa" / "smoke" / "smoke"


def test_custom_journal_names_are_used_by_status_and_exports(tmp_path):
    from graphids.exp.config import OutputConfig, RunConfig
    from graphids.exp.records import record_status, summarize_run, write_config
    from graphids.exp.tracking import discover_runs, load_ingest_payload

    run = RunConfig(
        name="custom",
        stage="test",
        outputs=OutputConfig(
            run_dir=tmp_path / "run",
            manifest_name="custom.json",
            events_name="history.jsonl",
        ),
    )
    write_config(run)
    record_status(run, "finished", result={"metrics": {"ap": 0.8}})
    summary = summarize_run(run.outputs.run_dir)
    assert summary.status == "finished"
    assert summary.last_event == "run_finished"
    assert discover_runs(tmp_path) == [run.outputs.run_dir]
    assert load_ingest_payload(run.outputs.run_dir)["metrics"] == {"ap": 0.8}
