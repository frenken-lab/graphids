"""Single-writer MLflow ingestion for completed GraphIDS runs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from graphids.exp.config import RunConfig
from graphids.exp.records import journal_dir, load_config, load_manifest

INGEST_PAYLOAD_NAME = "mlflow_ingest.json"


def tracking_mode() -> str:
    mode = os.environ.get("GRAPHIDS_MLFLOW_MODE", "offline").strip().lower()
    if mode in {"online", "live"}:
        return "online"
    return "disabled" if mode in {"off", "none", "disabled"} else "offline"


def mlflow_tags(run: RunConfig) -> dict[str, str]:
    tags = {
        "graphids.phase": run.stage,
        "graphids.group": run.stage,
        "graphids.variant": run.name,
        "graphids.stage": run.stage,
        "graphids.run_dir": str(run.outputs.run_dir),
        "graphids.git_sha": run.git_sha,
        "graphids.seed": str(run.seed),
        "graphids.representation": run.representation_cfg.kind,
    }
    if run.dataset is not None:
        tags["graphids.dataset"] = run.dataset
    if run.plan_id is not None:
        tags["graphids.plan_id"] = run.plan_id
    return tags


def mlflow_hparams(run: RunConfig) -> dict[str, Any]:
    return {
        "graphids.stage": run.stage,
        "graphids.dataset": run.dataset or "",
        "graphids.seed": run.seed,
        "graphids.plan_id": run.plan_id or "",
        "graphids.git_sha": run.git_sha,
        "graphids.run_dir": str(run.outputs.run_dir),
        "graphids.representation": run.representation_cfg.kind,
        "graphids.representation_cfg": asdict(run.representation_cfg),
        "graphids.payload": run.payload.model_dump(mode="json"),
        **{
            f"graphids.resource.{key}": value
            for key, value in run.resources.model_dump(mode="json").items()
        },
    }


def make_logger(run: RunConfig):
    from graphids._mlflow import make_logger as build_logger

    logger = build_logger(
        experiment_name=f"graphids/{run.dataset or 'unknown'}/{run.stage}",
        run_name=run.name,
        tags=mlflow_tags(run),
        artifact_location=mlflow_artifact_location(run),
    )
    logger.log_hyperparams(mlflow_hparams(run))
    return logger


@dataclass(frozen=True)
class IngestResult:
    run_dir: str
    run_id: str | None
    status: str
    message: str


def ingest_payload_path(run_dir: str | Path) -> Path:
    return journal_dir(run_dir) / INGEST_PAYLOAD_NAME


def mlflow_artifact_location(run: RunConfig) -> str:
    try:
        from graphids.paths import lake_root

        root = Path(lake_root())
    except RuntimeError:
        return str(run.outputs.run_dir / ".mlflow_artifacts")
    dataset = run.dataset or "unknown"
    return str(root / "mlartifacts" / dataset / run.stage)


def build_ingest_payload(
    run: RunConfig,
    *,
    status: str,
    metrics: Mapping[str, float] | None = None,
    result: Mapping[str, Any] | None = None,
    failure: str | None = None,
) -> dict[str, Any]:
    """Build the durable payload later replayed into MLflow."""
    tags = mlflow_tags(run)
    tags.setdefault("graphids.phase", run.stage)
    tags.setdefault("graphids.group", run.stage)
    tags.setdefault("graphids.variant", run.name)
    tags.setdefault("graphids.tracking_mode", "offline")
    return {
        "schema_version": 1,
        "experiment_name": f"graphids/{run.dataset or 'unknown'}/{run.stage}",
        "run_name": run.name,
        "run_dir": str(run.outputs.run_dir),
        "stage": run.stage,
        "dataset": run.dataset,
        "artifact_location": mlflow_artifact_location(run),
        "status": status,
        "failure": failure,
        "tags": tags,
        "params": mlflow_hparams(run),
        "metrics": dict(metrics or {}),
        "result": dict(result or {}),
        "artifacts": {
            "graphids": str(run.outputs.journal_dir()),
            "artifacts": str(run.outputs.artifact_path()),
            "checkpoints": str(run.outputs.checkpoint_path()),
        },
    }


def load_ingest_payload(run_dir: str | Path) -> dict[str, Any]:
    """Derive current exports from saved records; read legacy payloads as fallback."""
    manifest = load_manifest(run_dir)
    if manifest is not None and manifest.schema_version == 2:
        payload = build_ingest_payload(
            load_config(run_dir),
            status={"finished": "FINISHED", "failed": "FAILED", "killed": "KILLED"}.get(
                manifest.status, "RUNNING"
            ),
            metrics=manifest.metrics,
            result={**manifest.result, "metrics": manifest.metrics},
            failure=manifest.failure,
        )
        payload["tags"]["graphids.tracking_mode"] = manifest.tracking_mode
        return payload
    path = ingest_payload_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no MLflow ingest payload found: {path}")
    return json.loads(path.read_text())


def discover_runs(root: str | Path) -> list[Path]:
    """Find current run records and historical offline payloads without duplicates."""
    root = Path(root)
    runs = {
        path.parent.parent for path in root.glob(f"**/.graphids/{INGEST_PAYLOAD_NAME}")
    }
    for path in root.glob("**/resolved-run.yml"):
        # Defer parsing to ingest_many so one corrupt run cannot hide other runs.
        if (path.parent / ".graphids").is_dir():
            runs.add(path.parent)
    return sorted(runs)


def _flatten_param_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _flatten_params(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_params(value, prefix=name))
        else:
            out[name] = _flatten_param_value(value)
    return out


def _existing_run_id(client: Any, experiment_id: str, run_dir: str) -> str | None:
    escaped = run_dir.replace("\\", "\\\\").replace("'", "\\'")
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`graphids.run_dir` = '{escaped}'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _ensure_experiment(client: Any, *, name: str, artifact_location: str | None) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(name, artifact_location=artifact_location)


def _log_artifacts(client: Any, run_id: str, artifacts: Mapping[str, str]) -> None:
    for artifact_path, local in artifacts.items():
        path = Path(local)
        if path.is_dir() and any(path.iterdir()):
            client.log_artifacts(run_id, str(path), artifact_path=artifact_path)
        elif path.is_file():
            client.log_artifact(run_id, str(path), artifact_path=artifact_path)


def ingest_run(
    run_dir: str | Path,
    *,
    tracking_uri: str | None = None,
    log_artifacts: bool = True,
    skip_existing: bool = True,
) -> IngestResult:
    """Export one terminal run record or historical payload into MLflow.

    This function is intentionally single-process oriented. Use it from a
    post-run Slurm job, not from every training job.
    """
    manifest = load_manifest(run_dir)
    if manifest is not None and manifest.tracking_mode == "disabled":
        return IngestResult(str(run_dir), None, "skipped", "MLflow tracking disabled")

    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        from graphids._mlflow import configure_tracking_uri

        configure_tracking_uri()

    payload = load_ingest_payload(run_dir)
    status = str(payload.get("status") or "FINISHED").upper()
    if status in {"RUNNING", "SCHEDULED"}:
        raise RuntimeError(f"run is not terminal yet: {status}")
    if status not in {"FINISHED", "FAILED", "KILLED"}:
        status = "FINISHED"

    client = MlflowClient()
    experiment_id = _ensure_experiment(
        client,
        name=str(payload["experiment_name"]),
        artifact_location=payload.get("artifact_location"),
    )
    existing = _existing_run_id(client, experiment_id, str(payload["run_dir"]))
    if existing and skip_existing:
        return IngestResult(
            run_dir=str(payload["run_dir"]),
            run_id=existing,
            status="skipped",
            message="run already ingested",
        )

    tags = {str(k): str(v) for k, v in dict(payload.get("tags") or {}).items()}
    tags["mlflow.runName"] = str(payload.get("run_name") or Path(run_dir).name)
    run = client.create_run(experiment_id, tags=tags)
    run_id = run.info.run_id

    for key, value in _flatten_params(dict(payload.get("params") or {})).items():
        client.log_param(run_id, key, value)
    for key, value in dict(payload.get("metrics") or {}).items():
        if value is not None:
            client.log_metric(run_id, str(key), float(value))
    if log_artifacts:
        _log_artifacts(client, run_id, dict(payload.get("artifacts") or {}))

    client.set_terminated(run_id, status=status)
    return IngestResult(
        run_dir=str(payload["run_dir"]),
        run_id=run_id,
        status="ingested",
        message=f"logged MLflow run with status {status}",
    )


def ingest_many(
    paths: Iterable[str | Path],
    *,
    tracking_uri: str | None = None,
    log_artifacts: bool = True,
    skip_existing: bool = True,
) -> list[IngestResult]:
    results: list[IngestResult] = []
    for path in paths:
        run_dir = Path(path)
        if run_dir.name in (INGEST_PAYLOAD_NAME, "manifest.json"):
            run_dir = run_dir.parent.parent
        try:
            results.append(
                ingest_run(
                    run_dir,
                    tracking_uri=tracking_uri,
                    log_artifacts=log_artifacts,
                    skip_existing=skip_existing,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch ingest moving
            results.append(
                IngestResult(
                    run_dir=str(run_dir),
                    run_id=None,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
