"""Tests for the new experiment manifest + event journal seam."""

from __future__ import annotations

from typer.testing import CliRunner


def test_manifest_and_events_round_trip(tmp_path):
    from graphids.exp.config import OutputConfig, ResourceConfig, RunConfig
    from graphids.exp.config import FitRunPayload
    from graphids.exp.journal import EventRecord, RunManifest, append_event, load_events, load_manifest, write_manifest

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
        run_dir=run.outputs.run_dir,
        config={"payload": run.payload.model_dump(mode="json")},
        outputs={"run_dir": run.outputs.run_dir},
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
    from graphids.exp.journal import EventRecord, RunManifest, append_event, write_manifest

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
