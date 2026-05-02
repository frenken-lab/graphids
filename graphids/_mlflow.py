"""All MLflow surface for graphids — lifecycle (write) + filter helpers (read).

Thin wrapper around :class:`mlflow.tracking.MlflowClient` — no custom primitives,
no fluent-API magic. Public surfaces:

Lifecycle (write):
- ``ensure_tracking_uri()`` — set tracking URI from $MLFLOW_TRACKING_URI (fail-fast)
- ``start_training_run(row, phase)`` / ``end_training_run(run_id, status)``
- ``MLflowTrainingCallback`` — :class:`graphids.core.callbacks.CallbackBase`
  subclass that forwards per-epoch metrics via ``client.log_batch`` (one RPC/epoch);
  reads ``run_id`` from ``$GRAPHIDS_MLFLOW_RUN_ID`` (set by orchestrate)

Read helpers (no-op against MLflow on their own — pair with `client.search_runs`):
- ``build_search_filter(...)`` — compose `filter_string` from graphids tag schema
- ``resume_state(client, ...)`` → ``ResumeDecision`` — status-gated resume policy

Mandatory tags written at run open (per ``data-layout.md``):
  graphids.phase, graphids.run_dir, graphids.dataset, graphids.group,
  graphids.variant, graphids.seed, graphids.model_type, graphids.scale
SLURM env adds slurm.job_id + slurm.cluster_name when set.

Experiment shape: ``graphids/{dataset}/{group}`` (per-axis, post-2026-04-24).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from graphids.blueprint import TrainRow
from graphids.core.callbacks import CallbackBase

_TRACKING_SET = False


def ensure_tracking_uri() -> None:
    """Set tracking URI from $MLFLOW_TRACKING_URI. Idempotent; fail-fast on miss."""
    global _TRACKING_SET  # noqa: PLW0603
    if _TRACKING_SET:
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI unset — point it at sqlite:///<lake>/mlflow.db or http://..."
        )
    mlflow.set_tracking_uri(uri)
    _TRACKING_SET = True


def _experiment_id(client: MlflowClient, dataset: str, group: str) -> str:
    """Get-or-create the per-axis experiment `graphids/{dataset}/{group}`."""
    name = f"graphids/{dataset}/{group}"
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else client.create_experiment(name)


def start_training_run(row: TrainRow, phase: str) -> str:
    """Open an MLflow run for ``row`` + ``phase``; return run_id.

    System-metrics sampler attaches at 5s intervals (per data-layout.md). The
    caller (orchestrate.train / .evaluate) must close via ``end_training_run``.
    """
    ensure_tracking_uri()
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(5)
    m = row.meta
    client = MlflowClient()
    tags = {
        "graphids.phase": phase,
        "graphids.run_dir": row.identity.run_dir,
        "graphids.dataset": m.dataset,
        "graphids.group": m.group,
        "graphids.variant": m.variant,
        "graphids.seed": str(m.seed),
        "graphids.model_type": m.model_type,
        "graphids.scale": m.scale,
    }
    if jid := os.environ.get("SLURM_JOB_ID"):
        tags["slurm.job_id"] = jid
    if cluster := os.environ.get("SLURM_CLUSTER_NAME"):
        tags["slurm.cluster_name"] = cluster
    run = client.create_run(
        _experiment_id(client, m.dataset, m.group),
        run_name=row.identity.run_name,
        tags=tags,
    )
    return run.info.run_id


def end_training_run(run_id: str, status: str = "FINISHED") -> None:
    """Close an MLflow run with status FINISHED / FAILED / KILLED."""
    MlflowClient().set_terminated(run_id, status=status)


class MLflowTrainingCallback(CallbackBase):
    """Forward `trainer.callback_metrics` to MLflow + record run-scoped state.

    Per-epoch: one ``log_batch`` RPC per epoch (single round-trip for every metric).
    Pairs with :func:`start_training_run` (caller passes the run_id back in).

    Run-scoped (epoch 0 / fit-end): records non-derivable graphids state with
    the right MLflow primitive per value:

    - ``params`` — immutable per-run config: ``graphids.budget_target_bytes``,
      ``graphids.num_workers``, ``graphids.prefetch_factor``.
    - ``tags`` — categorical: ``graphids.budget_binding``,
      ``graphids.num_workers_source``.
    - ``metrics`` — numeric, threshold-filterable: ``graphids.peak_vram_mb``
      (logged once at fit-end at ``step=current_epoch``).

    Things the docs say MLflow gives for free, so NOT logged here:
    peak system VRAM time-series (``system/gpu_0_memory_usage_megabytes`` from
    the system sampler), epochs_run (``max(step)`` on any per-epoch metric),
    run wall-time (``end_time - start_time``).
    """

    def __init__(self, run_id: str | None = None) -> None:
        # run_id is set by orchestrate via ``GRAPHIDS_MLFLOW_RUN_ID`` before
        # ``_instantiate`` runs, so the libsonnet's empty ``init_args: {}`` is
        # honest. Direct callers (tests) may pass ``run_id`` explicitly.
        self.run_id = run_id or os.environ["GRAPHIDS_MLFLOW_RUN_ID"]
        self._client = MlflowClient()
        self._run_config_stamped = False

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if not self._run_config_stamped:
            self._stamp_run_config(trainer)
            self._run_config_stamped = True

        raw = {k: float(v) for k, v in trainer.callback_metrics.items() if v is not None}
        if not raw:
            return
        ts = int(time.time() * 1000)
        ms = [
            Metric(key=k, value=v, timestamp=ts, step=trainer.current_epoch)
            for k, v in _scalar_metrics(raw).items()
        ]
        self._client.log_batch(self.run_id, metrics=ms)

    def on_fit_end(self, trainer: Any, model: Any) -> None:
        peak_mb = _peak_vram_mb(model)
        if peak_mb > 0:
            # Single-point series — last==only, so threshold filtering on this
            # metric works as a one-shot "actual peak vs. probed budget" lookup.
            # Step = current_epoch so resumed fits append a new point per resume.
            self._client.log_metric(
                self.run_id,
                "graphids.peak_vram_mb",
                peak_mb,
                step=trainer.current_epoch,
            )
        budget = getattr(model, "_budget_cache", None)
        if budget is not None:
            self._client.set_tag(self.run_id, "graphids.budget_binding", budget.binding)
        self._register_logged_model(trainer, model)

    def _register_logged_model(self, trainer: Any, model: Any) -> None:
        """Register the best ckpt as a metadata-only ``LoggedModel`` (no bytes).

        Per data-layout.md: ckpt bytes stay on disk; LoggedModel is a secondary
        handle that exposes the run→ckpt link to ``search_logged_models``
        (cross-run ckpt queries by tag, dataset linkage). Idempotent on resume:
        query by ``source_run_id`` first, ``set_logged_model_tags`` if found,
        else ``create_logged_model``. Finalized to READY at the end.
        """
        from pathlib import Path

        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb is None or not getattr(ckpt_cb, "best_model_path", ""):
            return
        best_path = Path(ckpt_cb.best_model_path)
        if not best_path.exists():
            return
        sha_sidecar = best_path.with_suffix(best_path.suffix + ".sha256")
        sha = sha_sidecar.read_text().strip() if sha_sidecar.exists() else ""

        run = self._client.get_run(self.run_id)
        run_tags = run.data.tags
        tags = {
            "graphids.ckpt_path": str(best_path),
            "graphids.ckpt_sha256": sha,
            "graphids.dataset": run_tags["graphids.dataset"],
            "graphids.group": run_tags["graphids.group"],
            "graphids.variant": run_tags["graphids.variant"],
            "graphids.seed": run_tags["graphids.seed"],
        }
        params = {"graphids.run_dir": run_tags["graphids.run_dir"]}
        model_type = f"{type(model).__module__}.{type(model).__name__}"

        existing = self._client.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"source_run_id = '{self.run_id}'",
            max_results=1,
        )
        if existing:
            model_id = existing[0].model_id
            self._client.set_logged_model_tags(model_id, tags)
        else:
            lm = self._client.create_logged_model(
                experiment_id=run.info.experiment_id,
                name=run.info.run_name,
                source_run_id=self.run_id,
                model_type=model_type,
                tags=tags,
                params=params,
            )
            model_id = lm.model_id
        self._client.finalize_logged_model(model_id, status="READY")

    def _stamp_run_config(self, trainer: Any) -> None:
        """One-shot at epoch 0 — DM and probe state are populated by then.

        Idempotency: params reject same-key/different-value rewrites. Resume
        with a different cluster (different probe target_bytes) would error
        loudly here — surfaced rather than silently shadowed.
        """
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return

        # Budget target — read off the model (probe owner after the disentangle).
        m = getattr(dm, "_model", None)
        budget = getattr(m, "_budget_cache", None) if m is not None else None
        if budget is not None and budget.target_bytes > 0:
            self._client.log_param(
                self.run_id,
                "graphids.budget_target_bytes",
                str(budget.target_bytes),
            )

        # DataLoader autosize — DM populates ``_autosize_info`` on first
        # ``train_dataloader()`` build; epoch 0 is past that point.
        autosize = getattr(dm, "_autosize_info", None)
        if autosize is not None:
            self._client.log_param(
                self.run_id, "graphids.num_workers", str(autosize["num_workers"])
            )
            self._client.log_param(
                self.run_id, "graphids.prefetch_factor", str(autosize["prefetch_factor"])
            )
            self._client.set_tag(
                self.run_id, "graphids.num_workers_source", autosize["source"]
            )


def _peak_vram_mb(model: Any) -> float:
    """Peak CUDA-allocator high-water mark for the model's device, in MB.

    torch is imported lazily so non-training callers of :mod:`graphids._mlflow`
    (``cli/export.py``, ``analysis/compare.py``) don't pay the import cost.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        dev = getattr(model, "device", None)
        idx = dev.index if dev is not None and dev.index is not None else 0
        return torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
    except (AttributeError, RuntimeError):
        return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ---------------------------------------------------------------------------
# Read helpers — `filter_string` composer + status-gated resume policy.
# Callers run the actual `client.search_runs(...)` so the read path stays
# explicit at the call site; we only own the bits that have policy or
# schema knowledge (tag-key spelling, status decision).
# ---------------------------------------------------------------------------

# Tag keys with dots in them (every graphids.* + slurm.*) need backtick quoting
# in MLflow filter_string syntax. Quoting unconditionally is safer than
# enumerating which keys need it.
_TAG_PREDICATE = "tags.`{key}` = '{value}'"


def _scalar_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Rename keys for MLflow's metric-name alphabet (``[A-Za-z0-9_\\-. :/]``).

    Operating-point metrics like ``test/precision@0.95recall`` (emitted by
    ``core/models/base.py::_log_operating_points``) embed ``@`` and would
    otherwise trip ``log_metrics`` / ``log_batch``, killing the whole row.
    """
    return {k.replace("@", "_at_"): v for k, v in metrics.items()}


def build_search_filter(
    *,
    dataset: str | None = None,
    group: str | None = None,
    variant: str | None = None,
    seed: int | None = None,
    phase: str | None = None,
    status: str | None = None,
    cluster: str | None = None,
) -> str:
    """Compose an MLflow ``filter_string`` from graphids tag predicates.

    All keys map to the schema written by :func:`start_training_run`. ``status``
    becomes ``attributes.status`` (run-level, not a tag); the rest are tags
    quoted with backticks. Empty filter returns ``""``.
    """
    parts: list[str] = []
    if dataset is not None:
        parts.append(_TAG_PREDICATE.format(key="graphids.dataset", value=dataset))
    if group is not None:
        parts.append(_TAG_PREDICATE.format(key="graphids.group", value=group))
    if variant is not None:
        parts.append(_TAG_PREDICATE.format(key="graphids.variant", value=variant))
    if seed is not None:
        parts.append(_TAG_PREDICATE.format(key="graphids.seed", value=str(seed)))
    if phase is not None:
        parts.append(_TAG_PREDICATE.format(key="graphids.phase", value=phase))
    if cluster is not None:
        parts.append(_TAG_PREDICATE.format(key="slurm.cluster_name", value=cluster))
    if status is not None:
        parts.append(f"attributes.status = '{status}'")
    return " and ".join(parts)


@dataclass(frozen=True)
class ResumeDecision:
    """Outcome of consulting MLflow for a (variant, seed) tuple at fit-submit time.

    ``action``:
        ``new``    — open a fresh run_id (no prior, or prior FINISHED → re-train)
        ``resume`` — re-open ``run_id`` (prior FAILED/KILLED, or RUNNING + force)
        ``refuse`` — caller should abort (prior RUNNING; set GRAPHIDS_FORCE_RESUME=1 to override)
    """

    action: Literal["new", "resume", "refuse"]
    run_id: str | None
    reason: str


def resume_state(
    client: MlflowClient,
    *,
    dataset: str,
    group: str,
    variant: str,
    seed: int,
) -> ResumeDecision:
    """Status-gated decision: should this fit re-open the prior run, start fresh, or refuse?

    Policy (per data-layout.md §2):
      no prior run    → new
      FINISHED        → new (resubmit means redo)
      FAILED / KILLED → resume same run_id
      RUNNING         → refuse, unless ``GRAPHIDS_FORCE_RESUME=1`` is set
    """
    exp = client.get_experiment_by_name(f"graphids/{dataset}/{group}")
    if exp is None:
        return ResumeDecision(action="new", run_id=None, reason="no experiment yet")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=build_search_filter(
            dataset=dataset, group=group, variant=variant, seed=seed, phase="fit"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return ResumeDecision(action="new", run_id=None, reason="no prior fit run")
    run = runs[0]
    status = run.info.status
    if status in ("FAILED", "KILLED"):
        return ResumeDecision(action="resume", run_id=run.info.run_id, reason=f"prior fit {status}")
    if status == "RUNNING":
        if os.environ.get("GRAPHIDS_FORCE_RESUME") == "1":
            return ResumeDecision(
                action="resume",
                run_id=run.info.run_id,
                reason="RUNNING + GRAPHIDS_FORCE_RESUME=1",
            )
        return ResumeDecision(
            action="refuse",
            run_id=run.info.run_id,
            reason="prior fit RUNNING — set GRAPHIDS_FORCE_RESUME=1 to override",
        )
    if status == "FINISHED":
        return ResumeDecision(
            action="new", run_id=None, reason="prior fit FINISHED — new run for re-train"
        )
    return ResumeDecision(action="new", run_id=None, reason=f"unexpected status {status}")
