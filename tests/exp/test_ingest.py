from __future__ import annotations

import json


def _run_config(tmp_path):
    from graphids.exp.config import FitRunPayload, OutputConfig, RunConfig

    return RunConfig(
        name="offline-smoke",
        stage="fit",
        dataset="synthetic",
        plan_id="offline-plan",
        payload=FitRunPayload(
            data={"type": "dummy_data"},
            model={"type": "dummy_model"},
            trainer={"max_epochs": 1},
        ),
        outputs=OutputConfig(run_dir=tmp_path / "offline-smoke"),
    )


def test_write_ingest_payload_records_run_identity(tmp_path):
    from graphids.exp.ingest import ingest_payload_path, write_ingest_payload

    run = _run_config(tmp_path)
    path = write_ingest_payload(run, status="FINISHED", metrics={"val_loss": 0.25})

    assert path == ingest_payload_path(run.outputs.run_dir)
    payload = json.loads(path.read_text())
    assert payload["experiment_name"] == "graphids/synthetic/fit"
    assert payload["run_name"] == "offline-smoke"
    assert payload["metrics"]["val_loss"] == 0.25
    assert payload["tags"]["graphids.run_dir"] == str(run.outputs.run_dir)
    assert payload["tags"]["graphids.tracking_mode"] == "offline"


def test_ingest_run_creates_mlflow_run(tmp_path):
    from mlflow.tracking import MlflowClient

    from graphids.exp.ingest import ingest_run, write_ingest_payload

    run = _run_config(tmp_path)
    write_ingest_payload(run, status="FINISHED", metrics={"val_loss": 0.25})
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = ingest_run(run.outputs.run_dir, tracking_uri=tracking_uri, log_artifacts=False)

    assert result.status == "ingested"
    assert result.run_id
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("graphids/synthetic/fit")
    assert exp is not None
    rows = client.search_runs([exp.experiment_id], filter_string="tags.`graphids.tracking_mode` = 'offline'")
    assert len(rows) == 1
    assert rows[0].data.metrics["val_loss"] == 0.25


def test_exp_ingest_cli_json(tmp_path):
    from typer.testing import CliRunner

    from graphids.cli.app import app
    from graphids.exp.ingest import write_ingest_payload

    run = _run_config(tmp_path)
    write_ingest_payload(run, status="FINISHED", metrics={"val_loss": 0.25})
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = CliRunner().invoke(
        app,
        [
            "exp",
            "ingest",
            str(run.outputs.run_dir),
            "--tracking-uri",
            tracking_uri,
            "--no-artifacts",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert '"status": "ingested"' in result.stdout
