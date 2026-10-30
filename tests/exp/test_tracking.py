from __future__ import annotations

import json


def _run_config(tmp_path):
    from graphids.exp.config import OutputConfig, RunConfig, RunPayload

    return RunConfig(
        name="offline-smoke",
        stage="fit",
        dataset="synthetic",
        plan_id="offline-plan",
        payload=RunPayload(
            data={"type": "dummy_data"},
            model={"type": "dummy_model"},
            trainer={"max_epochs": 1},
        ),
        outputs=OutputConfig(run_dir=tmp_path / "offline-smoke"),
    )


def test_manifest_export_preserves_run_identity(tmp_path):
    from graphids.exp.records import record_status, write_config
    from graphids.exp.tracking import ingest_payload_path, load_ingest_payload

    run = _run_config(tmp_path)
    write_config(run)
    record_status(run, "finished", result={"metrics": {"val_loss": 0.25}})
    path = ingest_payload_path(run.outputs.run_dir)

    assert path == ingest_payload_path(run.outputs.run_dir)
    assert not path.exists()
    payload = load_ingest_payload(run.outputs.run_dir)
    assert payload["experiment_name"] == "graphids/synthetic/fit"
    assert payload["run_name"] == "offline-smoke"
    assert payload["metrics"]["val_loss"] == 0.25
    assert payload["tags"]["graphids.run_dir"] == str(run.outputs.run_dir)
    assert payload["tags"]["graphids.tracking_mode"] == "offline"


def test_ingest_run_creates_mlflow_run(tmp_path):
    from mlflow.tracking import MlflowClient
    from typer.testing import CliRunner

    from graphids.cli.app import app
    from graphids.exp.records import record_status, write_config
    from graphids.exp.tracking import ingest_run

    run = _run_config(tmp_path)
    write_config(run)
    record_status(run, "finished", result={"metrics": {"val_loss": 0.25}})
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = ingest_run(
        run.outputs.run_dir, tracking_uri=tracking_uri, log_artifacts=False
    )

    assert result.status == "ingested"
    assert result.run_id
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("graphids/synthetic/fit")
    assert exp is not None
    rows = client.search_runs(
        [exp.experiment_id], filter_string="tags.`graphids.tracking_mode` = 'offline'"
    )
    assert len(rows) == 1
    assert rows[0].data.metrics["val_loss"] == 0.25
    cli = CliRunner().invoke(
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

    assert cli.exit_code == 0, cli.stderr
    assert '"status": "skipped"' in cli.stdout


def test_legacy_payloads_and_manifests_remain_readable_and_current_state_wins(tmp_path):
    from graphids.exp.records import load_manifest, record_status, write_config
    from graphids.exp.tracking import (
        build_ingest_payload,
        discover_runs,
        ingest_payload_path,
        load_ingest_payload,
    )

    run = _run_config(tmp_path)
    legacy = ingest_payload_path(run.outputs.run_dir)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            build_ingest_payload(run, status="FINISHED", metrics={"val_loss": 99.0})
        )
    )
    # A historical manifest has neither a version nor saved result fields.
    (legacy.parent / "manifest.json").write_text(
        json.dumps(
            dict(
                run_id=run.name,
                name=run.name,
                stage="fit",
                git_sha="old",
                run_dir=str(run.outputs.run_dir),
                status="finished",
            )
        )
    )
    assert load_manifest(run.outputs.run_dir).schema_version == 1
    assert load_ingest_payload(run.outputs.run_dir)["metrics"] == {"val_loss": 99.0}
    write_config(run)
    record_status(run, "finished", result={"metrics": {"val_loss": 0.25}})
    assert load_ingest_payload(run.outputs.run_dir)["metrics"] == {"val_loss": 0.25}
    assert discover_runs(tmp_path) == [run.outputs.run_dir]


def test_ingestion_refuses_running_current_records(tmp_path):
    import pytest

    from graphids.exp.records import record_status, write_config
    from graphids.exp.tracking import ingest_run

    run = _run_config(tmp_path)
    write_config(run)
    record_status(run, "running")
    with pytest.raises(RuntimeError, match="not terminal"):
        ingest_run(
            run.outputs.run_dir, tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
        )


def test_discovery_keeps_corrupt_runs_visible_to_batch_error_reporting(tmp_path):
    from graphids.exp.records import write_config
    from graphids.exp.tracking import discover_runs, ingest_many

    run = _run_config(tmp_path)
    write_config(run)
    journal = run.outputs.run_dir / ".graphids"
    journal.mkdir()
    (journal / "manifest.json").write_text("broken JSON")
    assert discover_runs(tmp_path) == [run.outputs.run_dir]
    results = ingest_many(
        discover_runs(tmp_path), tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    assert len(results) == 1
    assert results[0].status == "failed"
    assert "JSONDecodeError" in results[0].message
