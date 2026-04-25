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

# MLflow's own limits — import so we track the installed version.
from mlflow.utils.validation import (  # noqa: E402
    MAX_ENTITY_KEY_LENGTH as _MAX_PARAM_KEY,
)
from mlflow.utils.validation import (
    MAX_PARAM_VAL_LENGTH as _MAX_PARAM_VALUE,
)
from mlflow.utils.validation import (
    MAX_TAG_VAL_LENGTH as _MAX_TAG_VALUE,
)

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


def build_search_filter(
    *,
    dataset: str | None = None,
    group: str | None = None,
    variant: str | None = None,
    seed: int | str | None = None,
    phase: str | None = None,
    cluster: str | None = None,
    run_name: str | None = None,
    run_dir: str | None = None,
    status: str | None = None,
) -> str:
    """Compose an ``AND``-joined ``mlflow.search_runs`` filter_string.

    Superset of the filters used by :func:`graphids.slurm.dag.fit_is_complete`,
    :func:`graphids.slurm.sizing.estimate_walltime_minutes`, and the future
    resume / compare / upstream-lineage lookups. Any field left ``None`` is
    not filtered. ``cluster`` matches the ``slurm.slurm_cluster_name`` tag
    (always set in-job); ``seed`` is coerced to string since MLflow tag
    values are strings.
    """
    clauses: list[str] = []
    if dataset:
        clauses.append(f"tags.`graphids.dataset` = '{dataset}'")
    if group:
        clauses.append(f"tags.`graphids.group` = '{group}'")
    if variant:
        clauses.append(f"tags.`graphids.variant` = '{variant}'")
    if seed is not None:
        clauses.append(f"tags.`graphids.seed` = '{seed}'")
    if phase:
        clauses.append(f"tags.`graphids.phase` = '{phase}'")
    if cluster:
        clauses.append(f"tags.`slurm.slurm_cluster_name` = '{cluster}'")
    if run_name:
        clauses.append(f"attributes.run_name = '{run_name}'")
    if run_dir:
        clauses.append(f"tags.`graphids.run_dir` = '{run_dir}'")
    if status:
        clauses.append(f"attributes.status = '{status}'")
    return " AND ".join(clauses)


def ensure_tracking_uri() -> str | None:
    """Set ``MLFLOW_TRACKING_URI`` in env if unset. Safe to call from workers."""
    uri = os.environ.get(_TRACKING_URI_ENV)
    if uri:
        return uri
    from graphids.config.constants import LAKE_ROOT

    if not LAKE_ROOT:
        return None
    default = f"sqlite:///{Path(LAKE_ROOT) / _BACKEND_DB_SUBPATH}"
    os.environ[_TRACKING_URI_ENV] = default
    return default


def _ensure_experiment(client, name: str) -> None:  # noqa: ANN001 — MlflowClient
    """Create experiment if missing. ``mlartifacts/`` stays empty (data-layout.md)."""
    if client.get_experiment_by_name(name) is not None:
        return
    from graphids.config.constants import LAKE_ROOT

    artifact_location = f"file://{Path(LAKE_ROOT) / _ARTIFACT_SUBPATH}" if LAKE_ROOT else None
    client.create_experiment(name, artifact_location=artifact_location)


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


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten trainer metrics dict to ``{name: float}``.

    Accepts flat and one-deep nested (per-test-subdir) shapes. Non-numeric
    values are skipped. ``@`` in operating-point names (e.g.
    ``test/precision@0.95recall``) is rewritten to ``_at_`` because
    MLflow's metric-name validator rejects it and would otherwise fail
    the whole ``log_metrics`` call.
    """
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                    out[f"{k}/{sk}".replace("@", "_at_")] = float(sv)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k.replace("@", "_at_")] = float(v)
    return out


def _collect_ckpt_paths(obj: Any) -> list[str]:
    """Walk a rendered config and return every string value ending in ``.ckpt``.

    Keys are ignored — the convention is that teacher checkpoint references
    are string values matching ``*.ckpt``, regardless of where they sit in
    the tree (``data.init_args.scorer.init_args.ckpt_path`` for
    ``curriculum_vgae``; other keys in other presets).
    """
    hits: list[str] = []

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)
        elif isinstance(o, str) and o.endswith(".ckpt"):
            hits.append(o)

    _walk(obj)
    return hits


def _upstream_tags(resolved_config: dict[str, Any]) -> dict[str, str]:
    """Derive upstream-lineage tags from every ``.ckpt`` path in the config.

    Tag the upstream **run_dir** (filesystem path), not the MLflow ``run_id``:
    path is stable identity, doesn't require an MLflow query at submit-time,
    and handles the fan-in shape of fusion (vgae + focal upstreams — parent_run_id
    is single-parent). Role comes from :func:`parse_run_dir`; off-tree paths
    fall back to ``u0`` / ``u1``.
    """
    tags: dict[str, str] = {}
    for idx, ckpt in enumerate(_collect_ckpt_paths(resolved_config)):
        run_dir = Path(ckpt).parent.parent  # strip /checkpoints/best_model.ckpt
        identity = parse_run_dir(run_dir)
        role = f"{identity.group}_{identity.variant}" if identity else f"u{idx}"
        tags[f"graphids.upstream.{role}.run_dir"] = str(run_dir)[:_MAX_TAG_VALUE]
        tags[f"graphids.upstream.{role}.ckpt_path"] = ckpt[:_MAX_TAG_VALUE]
    return tags


def _dataset_for(resolved_config: dict[str, Any]):
    """Return a :class:`mlflow.data.MetaDataset` for this run's cache, or ``None``.

    Metadata-only dataset entity. Digest = SHA256 of ``cache_metadata.json``
    (same content-addressing the old ``_cache_digest_tags`` used).
    ``mlflow.log_input(ds, context=...)`` at run start stamps dataset identity
    as a first-class UI entity + filter surface (``dataset.name`` /
    ``dataset.digest`` in ``search_runs``) without calling the banned
    ``log_artifact`` path.
    """
    from graphids.config.constants import LAKE_ROOT

    data_init = (resolved_config.get("data") or {}).get("init_args") or {}
    ds_field = data_init.get("dataset")
    if isinstance(ds_field, dict):
        dataset = (ds_field.get("init_args") or {}).get("name")
    elif isinstance(ds_field, str):
        dataset = ds_field
    else:
        dataset = None
    cache_version = data_init.get("cache_version") or data_init.get("version")
    if not (LAKE_ROOT and dataset):
        return None
    candidates = []
    if cache_version:
        candidates.append(Path(LAKE_ROOT) / "cache" / f"v{cache_version}" / dataset)
    candidates.append(Path(LAKE_ROOT) / "cache" / dataset)
    cache_dir: Path | None = None
    for candidate in candidates:
        if (candidate / "cache_metadata.json").exists():
            cache_dir = candidate
            break
    if cache_dir is None:
        return None
    try:
        digest = hashlib.sha256((cache_dir / "cache_metadata.json").read_bytes()).hexdigest()[:16]
    except OSError:
        return None
    try:
        from mlflow.data.meta_dataset import MetaDataset
        from mlflow.data.sources import LocalArtifactDatasetSource
    except ImportError:
        return None
    return MetaDataset(
        source=LocalArtifactDatasetSource(uri=f"file://{cache_dir}"),
        name=dataset,
        digest=digest,
    )


def _checkpoint_hash_tag(run_dir: Path) -> dict[str, str]:
    """Read ``.sha256`` sidecar for ``best_model.ckpt`` if present."""
    sidecar = run_dir / "checkpoints" / "best_model.ckpt.sha256"
    if sidecar.exists():
        try:
            return {"graphids.ckpt_sha256": sidecar.read_text().strip().split()[0][:_MAX_TAG_VALUE]}
        except OSError:
            pass
    return {}


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


_SLURM_TAG_ENVS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CLUSTER_NAME",
    "SLURMD_NODENAME",
)


def _build_tags(
    identity: RunIdentity,
    run_dir: Path,
    resolved_config: dict[str, Any],
) -> dict[str, str]:
    """All tags for a fit/test run — identity, ckpt hash, git, env, SLURM, upstream.

    Cache digest is captured by ``_dataset_for`` + ``mlflow.log_input`` as a
    first-class MetaDataset entity, not a tag.
    """
    import sys

    from graphids.config.constants import PROJECT_ROOT

    tags: dict[str, str] = {
        "graphids.run_dir": str(run_dir)[:_MAX_TAG_VALUE],
        "graphids.dataset": identity.dataset,
        "graphids.seed": str(identity.seed),
        "graphids.group": identity.group,
        "graphids.variant": identity.variant,
        "graphids.python_version": sys.version.split()[0],
    }
    tags.update(_checkpoint_hash_tag(run_dir))
    tags.update(_git_sha_tag())
    tags.update(_upstream_tags(resolved_config))
    for k in _SLURM_TAG_ENVS:
        if k in os.environ:
            tags[f"slurm.{k.lower()}"] = os.environ[k][:_MAX_TAG_VALUE]
    try:
        lock = Path(PROJECT_ROOT) / "uv.lock"
        if lock.exists():
            tags["graphids.uv_lock_hash"] = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return tags


_FORCE_RESUME_ENV = "GRAPHIDS_FORCE_RESUME"


def _resume_decision(existing_run: Any, cur_git_sha: str | None, force: bool) -> str:
    """Return ``'resume'`` | ``'new'`` | ``'refuse'`` for an existing run.

    Status table (moderate plan Q5) + git-SHA discontinuity (review Q6, option b):

    | status     | no force                      | force  |
    |------------|-------------------------------|--------|
    | FAILED     | resume                        | resume |
    | KILLED     | resume                        | resume |
    | TERMINATED | new (reaper owns tombstone)   | resume |
    | RUNNING    | refuse (live writer / zombie) | refuse |
    | FINISHED   | refuse (use force to override)| resume |

    Git SHA change always forces a new run (option b: treat SHA flip as a
    discontinuity rather than silently mixing commits in one row) — unless
    ``force=True`` explicitly opts into cross-SHA resume.
    """
    status = existing_run.info.status
    existing_sha = (existing_run.data.tags or {}).get("git_sha")

    if cur_git_sha and existing_sha and cur_git_sha != existing_sha and not force:
        return "new"

    if status in ("FAILED", "KILLED"):
        return "resume"
    if status == "TERMINATED":
        return "resume" if force else "new"
    if status == "RUNNING":
        return "refuse"
    if status == "FINISHED":
        return "resume" if force else "refuse"
    return "new"


def start_training_run(run_dir: Path, resolved_config: dict[str, Any]) -> str | None:
    """Open an MLflow run for the fit phase. Returns ``run_name`` or ``None``.

    **Idempotent.** Searches for an existing fit run with this ``run_name``
    in the per-axis experiment; resumes FAILED/KILLED, refuses RUNNING/
    FINISHED (unless ``GRAPHIDS_FORCE_RESUME=1``), creates a fresh run for
    TERMINATED (reaper owns) and for git-SHA discontinuities. Returns
    ``None`` on refusal so callers know no MLflow row is active.

    Logs params, identity tags, SLURM provenance, reproducibility tags,
    and git SHA up-front. System metrics sampling is enabled for the
    process. Per-epoch metrics are appended later by
    ``MLflowTrainingCallback``.
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
        global _system_metrics_configured  # noqa: PLW0603
        if not _system_metrics_configured:
            try:
                mlflow.config.enable_system_metrics_logging()
                mlflow.config.set_system_metrics_sampling_interval(_SYSTEM_METRICS_INTERVAL_S)
                _system_metrics_configured = True
            except Exception as exc:
                log.warning("mlflow_system_metrics_config_failed", error=str(exc))
        mlflow.set_tracking_uri(uri)
        from graphids.config.settings import get_settings

        cluster = get_settings().cluster or None
        run_name = run_name_for(identity, cluster=cluster)
        experiment = f"graphids/{identity.dataset}/{identity.group}"

        client = MlflowClient(tracking_uri=uri)
        _ensure_experiment(client, experiment)
        exp = client.get_experiment_by_name(experiment)
        mlflow.set_experiment(experiment)

        force = os.environ.get(_FORCE_RESUME_ENV, "").lower() in ("1", "true", "yes")
        cur_git_sha = _git_sha_tag().get("git_sha")
        hits = client.search_runs(
            experiment_ids=[exp.experiment_id] if exp else None,
            filter_string=build_search_filter(run_name=run_name, phase="fit"),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        resume_run_id: str | None = None
        if hits:
            decision = _resume_decision(hits[0], cur_git_sha, force)
            if decision == "refuse":
                log.warning(
                    "mlflow_skip_refuse_existing",
                    status=hits[0].info.status,
                    run_id=hits[0].info.run_id,
                    run_name=run_name,
                )
                return None
            if decision == "resume":
                resume_run_id = hits[0].info.run_id

        if resume_run_id:
            mlflow.start_run(run_id=resume_run_id)
            log.info("mlflow_fit_run_resumed", run_id=resume_run_id, run_name=run_name)
        else:
            mlflow.start_run(run_name=run_name, tags={"graphids.phase": "fit"})

        try:
            mlflow.log_params(_flatten_params(resolved_config))
        except Exception as exc:
            # Resuming with altered config hits ``ParamValueAlreadyLoggedWithDifferentValueError``.
            # Not fatal — the original params stay; log a warning so drift is visible.
            log.warning("mlflow_log_params_failed", error=str(exc))
        mlflow.set_tags(_build_tags(identity, run_dir, resolved_config))
        if force and resume_run_id:
            mlflow.set_tag("graphids.resume.forced", "true")
        dataset = _dataset_for(resolved_config)
        if dataset is not None:
            mlflow.log_input(dataset, context="train")
        log.info(
            "mlflow_fit_run_started",
            run_name=run_name,
            experiment=experiment,
            resumed=bool(resume_run_id),
        )
        return run_name
    except Exception as exc:
        log.warning(
            "mlflow_start_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            run_dir=str(run_dir),
        )
        # Outer failure path must not leave a dangling active run — otherwise
        # the callback-free exit path logs against whatever happened to be
        # open and confuses downstream queries.
        try:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
        except Exception:
            pass
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


def _register_logged_model(
    run_id: str,
    experiment_id: str,
    identity: RunIdentity,
    run_dir: Path,
    best_ckpt_path: str,
) -> str | None:
    """Register a metadata-only ``LoggedModel`` pointing at this run's ckpt.

    ``create_logged_model`` takes metadata only (no artifact bytes), so this
    honors the data-layout.md ban on ``log_artifact`` / ``log_model`` while
    still producing a first-class MLflow-3 entity that the UI renders in a
    dedicated panel and that ``search_logged_models`` can filter on.

    Downstream lineage: a fusion / curriculum_vgae run's
    ``graphids.upstream.<role>.run_dir`` tag → ``search_runs(tags.graphids.run_dir = ...)``
    → upstream ``run_id`` → ``search_logged_models(source_run_id=upstream_run_id,
    model_type='<group>_<variant>')`` returns this LoggedModel.
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        sha_tag = _checkpoint_hash_tag(run_dir)
        lm = client.create_logged_model(
            experiment_id=experiment_id,
            name=f"{identity.variant}_seed{identity.seed}",
            source_run_id=run_id,
            model_type=f"{identity.group}_{identity.variant}",
            tags={
                "graphids.ckpt_path": best_ckpt_path[:_MAX_TAG_VALUE],
                "graphids.run_dir": str(run_dir)[:_MAX_TAG_VALUE],
                **sha_tag,
            },
        )
        return lm.model_id
    except Exception as exc:
        log.warning("mlflow_logged_model_failed", error=str(exc))
        return None


def log_final_fit(
    *,
    peak_vram_mb: float,
    epochs_run: int,
    best_ckpt_path: str,
    run_dir: Path,
) -> None:
    """Stamp peak VRAM + epochs run + checkpoint hash + LoggedModel on the active run.

    Called from ``MLflowTrainingCallback.on_fit_end`` before the run closes.
    """
    try:
        import mlflow

        active = mlflow.active_run()
        if active is None:
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

        identity = parse_run_dir(run_dir)
        if identity is not None and best_ckpt_path:
            model_id = _register_logged_model(
                active.info.run_id,
                active.info.experiment_id,
                identity,
                run_dir,
                best_ckpt_path,
            )
            if model_id:
                tags["graphids.logged_model_id"] = model_id
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
        experiment = f"graphids/{identity.dataset}/{identity.group}"
        client = MlflowClient(tracking_uri=uri)
        _ensure_experiment(client, experiment)
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=run_name, tags={"graphids.phase": "test"}):
            scalars = _scalar_metrics(metrics)
            if scalars:
                mlflow.log_metrics(scalars)
            mlflow.set_tags({**_build_tags(identity, run_dir, resolved_config), "status": "ok"})
            dataset = _dataset_for(resolved_config)
            if dataset is not None:
                mlflow.log_input(dataset, context="test")
        log.info("mlflow_test_run_logged", run_name=run_name, experiment=experiment)
        return run_name
    except Exception as exc:
        log.warning("mlflow_test_failed", error=str(exc), run_dir=str(run_dir))
        return None
