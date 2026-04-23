"""MLflow lifecycle + sink for training runs.

Owns the MLflow run lifecycle during ``stage.train`` / ``stage.evaluate``:
the run opens at fit-start, per-epoch metrics stream in through the
``MLflowTrainingCallback`` in ``core/mlflow_callback.py``, the run closes
at fit-end (normal) or on exception (FAILED). A separate, self-contained
MLflow run is written for ``stage.evaluate`` (test phase), linked back to
the fit run via the identity-derived ``run_name`` shared between them.

Backend: SQLite at ``{lake_root}/mlflow.db``. Artifacts at
``file://{lake_root}/mlartifacts``. System metrics (GPU utilization,
VRAM, CPU, memory, disk, network) are captured automatically by
MLflow's background sampling thread while any run is active.

Every MLflow call is wrapped in try/swallow: a logging hiccup must not
fail a training job.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphids._otel import get_logger

log = get_logger(__name__)

_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
_BACKEND_DB_SUBPATH = "mlflow.db"
_ARTIFACT_SUBPATH = "mlartifacts"
_SYSTEM_METRICS_INTERVAL_S = 5

# MLflow 3.x limits.
_MAX_PARAM_KEY = 250
_MAX_PARAM_VALUE = 6000
_MAX_TAG_VALUE = 500

_system_metrics_configured = False


@dataclass(frozen=True)
class RunIdentity:
    """The ``(group, variant, dataset, seed)`` tuple that identifies a run."""

    group: str
    variant: str
    dataset: str
    seed: int


def parse_run_dir(run_dir: Path) -> RunIdentity | None:
    """Return identity for an ablation run_dir, or ``None`` if off-tree.

    Expected shape: ``.../<dataset>/ablations/<group>/<variant>/seed_<N>``.
    """
    parts = Path(run_dir).parts
    if len(parts) < 5:
        return None
    seed_part, variant, group, ablations_marker, dataset = (
        parts[-1],
        parts[-2],
        parts[-3],
        parts[-4],
        parts[-5],
    )
    if ablations_marker != "ablations" or not seed_part.startswith("seed_"):
        return None
    try:
        seed = int(seed_part.removeprefix("seed_"))
    except ValueError:
        return None
    return RunIdentity(group=group, variant=variant, dataset=dataset, seed=seed)


def run_name_for(identity: RunIdentity, cluster: str | None = None) -> str:
    """Build the deterministic MLflow ``run_name`` for an identity."""
    base = f"{identity.group}_{identity.variant}_{identity.dataset}_seed{identity.seed}"
    return f"{base}_{cluster}" if cluster else base


def _default_tracking_uri() -> str | None:
    from graphids.config.constants import LAKE_ROOT

    if not LAKE_ROOT:
        return None
    return f"sqlite:///{Path(LAKE_ROOT) / _BACKEND_DB_SUBPATH}"


def _default_artifact_location() -> str | None:
    """Declared-but-unused artifact root.

    MLflow requires ``artifact_location`` per experiment. Graphids keeps
    checkpoints on the filesystem under ``{run_dir}/checkpoints/`` (linked
    to MLflow rows via the ``graphids.run_dir`` + ``graphids.ckpt_sha256``
    tags), so ``mlartifacts/`` exists on disk but stays empty. Reserved
    for future ``mlflow.log_artifact`` calls if any emerge.
    """
    from graphids.config.constants import LAKE_ROOT

    if not LAKE_ROOT:
        return None
    return f"file://{Path(LAKE_ROOT) / _ARTIFACT_SUBPATH}"


def ensure_tracking_uri() -> str | None:
    """Set ``MLFLOW_TRACKING_URI`` in env if unset. Safe to call from workers."""
    uri = os.environ.get(_TRACKING_URI_ENV)
    if uri:
        return uri
    default = _default_tracking_uri()
    if default:
        os.environ[_TRACKING_URI_ENV] = default
    return default


def _configure_system_metrics() -> None:
    """Enable MLflow's background system metrics sampler. Idempotent per process."""
    global _system_metrics_configured
    if _system_metrics_configured:
        return
    try:
        import mlflow

        mlflow.config.enable_system_metrics_logging()
        mlflow.config.set_system_metrics_sampling_interval(_SYSTEM_METRICS_INTERVAL_S)
        _system_metrics_configured = True
    except Exception as exc:
        log.warning("mlflow_system_metrics_config_failed", error=str(exc))


def _ensure_experiment(client, name: str) -> None:
    if client.get_experiment_by_name(name) is not None:
        return
    client.create_experiment(name, artifact_location=_default_artifact_location())


def _flatten_params(obj: Any, parent: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent}.{k}" if parent else str(k)
            out.update(_flatten_params(v, key))
    elif isinstance(obj, (list, tuple)):
        out[parent] = repr(obj)[:_MAX_PARAM_VALUE]
    else:
        value = "" if obj is None else str(obj)
        out[parent[:_MAX_PARAM_KEY]] = value[:_MAX_PARAM_VALUE]
    return out


def _sanitize_metric_name(name: str) -> str:
    """Make a metric key safe for MLflow's name validator.

    MLflow allows alphanumerics plus ``_ - . : / <space>`` only. Operating-point
    keys like ``test/precision@0.95recall`` embed ``@`` and would otherwise
    fail the whole ``log_metrics`` call (and thus the whole test-phase row).
    """
    return name.replace("@", "_at_")


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten trainer metrics dict to ``{name: float}``.

    Accepts flat and one-deep nested (per-test-subdir) shapes. Non-numeric
    values are skipped. Metric names are sanitized for MLflow's validator.
    """
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                    out[_sanitize_metric_name(f"{k}/{sk}")] = float(sv)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[_sanitize_metric_name(k)] = float(v)
    return out


def _slurm_tags() -> dict[str, str]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CLUSTER_NAME",
        "SLURMD_NODENAME",
    )
    return {f"slurm.{k.lower()}": os.environ[k][:_MAX_TAG_VALUE] for k in keys if k in os.environ}


def _cache_digest_tags(resolved_config: dict[str, Any]) -> dict[str, str]:
    """Read ``cache_metadata.json`` for the dataset and tag the cache digest.

    Cheap provenance — ties the run to a specific cache version without
    the full ``mlflow.data`` dataset-object machinery.
    """
    from graphids.config.constants import LAKE_ROOT

    data_init = (resolved_config.get("data") or {}).get("init_args") or {}
    # ``dataset`` is a class_path wrapper: {class_path: CANBusSource, init_args: {name: ...}}.
    # Older plain-string configs are still accepted.
    ds_field = data_init.get("dataset")
    if isinstance(ds_field, dict):
        dataset = (ds_field.get("init_args") or {}).get("name")
    elif isinstance(ds_field, str):
        dataset = ds_field
    else:
        dataset = None
    cache_version = data_init.get("cache_version") or data_init.get("version")
    if not (LAKE_ROOT and dataset):
        return {}
    candidates = []
    if cache_version:
        candidates.append(
            Path(LAKE_ROOT) / "cache" / f"v{cache_version}" / dataset / "cache_metadata.json"
        )
    candidates.append(Path(LAKE_ROOT) / "cache" / dataset / "cache_metadata.json")
    for path in candidates:
        if path.exists():
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                return {
                    "graphids.cache_digest": digest[:16],
                    "graphids.cache_metadata_path": str(path)[:_MAX_TAG_VALUE],
                }
            except OSError:
                break
    return {}


def _checkpoint_hash_tag(run_dir: Path) -> dict[str, str]:
    """Read ``.sha256`` sidecar for ``best_model.ckpt`` if present."""
    sidecar = run_dir / "checkpoints" / "best_model.ckpt.sha256"
    if sidecar.exists():
        try:
            return {"graphids.ckpt_sha256": sidecar.read_text().strip().split()[0][:_MAX_TAG_VALUE]}
        except OSError:
            pass
    return {}


def _identity_tags(identity: RunIdentity, run_dir: Path, cluster: str | None) -> dict[str, str]:
    return {
        "graphids.run_id": run_name_for(identity, cluster=cluster),
        "graphids.run_dir": str(run_dir)[:_MAX_TAG_VALUE],
        "graphids.dataset": identity.dataset,
        "graphids.seed": str(identity.seed),
        "graphids.group": identity.group,
        "graphids.variant": identity.variant,
        "graphids.cluster": cluster or "",
    }


def _git_sha_tag() -> dict[str, str]:
    """Grab current HEAD SHA. Swallows failure (detached head, no git, etc)."""
    try:
        import subprocess

        from graphids.config.constants import PROJECT_ROOT

        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"git_sha": sha}
    except Exception:
        return {}


def start_training_run(run_dir: Path, resolved_config: dict[str, Any]) -> str | None:
    """Open an MLflow run for the fit phase. Returns ``run_name`` or ``None``.

    Logs params, identity tags, SLURM provenance, cache digest, and git
    SHA up-front. System metrics sampling is enabled for the process.
    Per-epoch metrics are appended later by ``MLflowTrainingCallback``.
    """
    identity = parse_run_dir(run_dir)
    if identity is None:
        log.info("mlflow_skip_non_ablation", run_dir=str(run_dir))
        return None

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        log.warning("mlflow_skip_no_install")
        return None

    try:
        uri = ensure_tracking_uri()
        if not uri:
            log.warning("mlflow_skip_no_uri")
            return None
        _configure_system_metrics()
        mlflow.set_tracking_uri(uri)
        from graphids.config.settings import get_settings

        cluster = get_settings().cluster or None
        run_name = run_name_for(identity, cluster=cluster)
        experiment = f"graphids/{identity.group}/{identity.variant}"

        client = MlflowClient(tracking_uri=uri)
        _ensure_experiment(client, experiment)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name, tags={"graphids.phase": "fit"})
        mlflow.log_params(_flatten_params(resolved_config))
        tags = {
            **_identity_tags(identity, run_dir, cluster),
            **_cache_digest_tags(resolved_config),
            **_git_sha_tag(),
            **_slurm_tags(),
        }
        mlflow.set_tags(tags)
        log.info("mlflow_fit_run_started", run_name=run_name, experiment=experiment)
        return run_name
    except Exception as exc:
        log.warning("mlflow_start_failed", error=str(exc), run_dir=str(run_dir))
        return None


def log_epoch_metrics(epoch: int, metrics: dict[str, float]) -> None:
    """Log per-epoch scalar metrics to the active MLflow run. No-op if none."""
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if clean:
            mlflow.log_metrics(clean, step=epoch)
    except Exception as exc:
        log.warning("mlflow_epoch_log_failed", error=str(exc), epoch=epoch)


def log_final_fit(
    *,
    peak_vram_mb: float,
    epochs_run: int,
    best_ckpt_path: str,
    run_dir: Path,
) -> None:
    """Stamp peak VRAM + epochs run + checkpoint hash on the active run.

    Called from ``MLflowTrainingCallback.on_fit_end`` before the run closes.
    """
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        mlflow.log_metrics(
            {
                "peak_vram_mb": float(peak_vram_mb),
                "epochs_run": float(epochs_run),
            }
        )
        tags: dict[str, str] = {}
        if best_ckpt_path:
            tags["graphids.best_ckpt_path"] = best_ckpt_path[:_MAX_TAG_VALUE]
        tags.update(_checkpoint_hash_tag(run_dir))
        if tags:
            mlflow.set_tags(tags)
    except Exception as exc:
        log.warning("mlflow_final_fit_log_failed", error=str(exc))


def end_training_run(status: str = "FINISHED") -> None:
    """End the active MLflow run. Idempotent; swallowed on error."""
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except Exception as exc:
        log.warning("mlflow_end_failed", error=str(exc))


def log_test_run(
    run_dir: Path,
    *,
    resolved_config: dict[str, Any],
    metrics: dict[str, Any],
) -> str | None:
    """Self-contained MLflow run for the test phase (post-hoc sink)."""
    identity = parse_run_dir(run_dir)
    if identity is None:
        log.info("mlflow_test_skip_non_ablation", run_dir=str(run_dir))
        return None

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        log.warning("mlflow_test_skip_no_install")
        return None

    try:
        uri = ensure_tracking_uri()
        if not uri:
            log.warning("mlflow_test_skip_no_uri")
            return None
        mlflow.set_tracking_uri(uri)
        from graphids.config.settings import get_settings

        cluster = get_settings().cluster or None
        run_name = run_name_for(identity, cluster=cluster)
        experiment = f"graphids/{identity.group}/{identity.variant}"
        client = MlflowClient(tracking_uri=uri)
        _ensure_experiment(client, experiment)
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=run_name, tags={"graphids.phase": "test"}):
            scalars = _scalar_metrics(metrics)
            if scalars:
                mlflow.log_metrics(scalars)
            tags = {
                **_identity_tags(identity, run_dir, cluster),
                **_checkpoint_hash_tag(run_dir),
                **_git_sha_tag(),
                **_slurm_tags(),
                "status": "ok",
            }
            mlflow.set_tags(tags)
        log.info("mlflow_test_run_logged", run_name=run_name, experiment=experiment)
        return run_name
    except Exception as exc:
        log.warning("mlflow_test_failed", error=str(exc), run_dir=str(run_dir))
        return None
