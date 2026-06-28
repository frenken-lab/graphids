from __future__ import annotations

import mlflow
from typer.testing import CliRunner


def _log_temporal_run(
    *,
    tracking_uri: str,
    dataset: str,
    variant: str,
    mcc: float,
    fuzzing: float | None = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"graphids/{dataset}/temporal")
    with mlflow.start_run(run_name=f"temporal_{variant}_{dataset}_seed42") as run:
        mlflow.set_tags(
            {
                "graphids.phase": "test",
                "graphids.group": "temporal",
                "graphids.variant": variant,
                "graphids.dataset": dataset,
            }
        )
        mlflow.log_metric("test/test/mcc", mcc)
        mlflow.log_metric("test/test/auroc_macro", 0.9)
        if fuzzing is not None:
            mlflow.log_metric("test/test/auroc_per_attack/fuzzing", fuzzing)
        return run.info.run_id


def test_query_result_view_latest_per_variant(tmp_path):
    from graphids.exp.results import query_result_view

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    old_run = _log_temporal_run(tracking_uri=tracking_uri, dataset="hcrl_sa", variant="temporal_gat", mcc=0.1)
    latest_run = _log_temporal_run(
        tracking_uri=tracking_uri,
        dataset="hcrl_sa",
        variant="temporal_gat",
        mcc=0.9,
        fuzzing=0.8,
    )
    mlflow.set_tracking_uri(tracking_uri)

    rows = query_result_view(view="any", datasets=["hcrl_sa"])

    assert len(rows) == 1
    assert rows[0].run_id == latest_run
    assert rows[0].run_id != old_run
    assert rows[0].metrics["test/test/mcc"] == 0.9
    assert rows[0].metrics["test/test/auroc_per_attack/fuzzing"] == 0.8


def test_exp_results_cli_json(tmp_path):
    from graphids.cli.app import app

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = _log_temporal_run(
        tracking_uri=tracking_uri,
        dataset="hcrl_sa",
        variant="temporal_gat",
        mcc=0.7,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "exp",
            "results",
            "--tracking-uri",
            tracking_uri,
            "--dataset",
            "hcrl_sa",
            "--variant",
            "temporal_gat",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert run_id in result.stdout
    assert '"test/test/mcc": 0.7' in result.stdout
