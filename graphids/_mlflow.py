"""MLflow surface for graphids — run open, callback, LM lookup, filter helpers.

Tracking URI + system-metrics sampler are configured process-once in
:func:`graphids.orchestrate.setup`; this module assumes both are live.

Public surfaces:
- ``identity_tags(row, phase)`` — tag dict for an ``MLFlowLogger`` to
  open a run with. The logger owns lifecycle (lazy open + finalize).
- ``MLflowTrainingCallback`` — Lightning callback for catalog ``LoggedModel``
  lifecycle + graphids-specific run state. Per-epoch metric forwarding is
  delegated to ``lightning.pytorch.loggers.MLFlowLogger`` (wired in
  :func:`graphids.orchestrate._make_trainer`); the callback pulls run_id +
  client from ``trainer.logger`` in ``on_train_start``.
- ``_find_logged_model`` / ``_find_logged_model_by_ckpt`` — LM lookup by
  identity tags or ckpt path (used for upstream lineage).
- ``build_search_filter(...)`` — compose ``filter_string`` from the
  graphids tag schema; single source of truth per ``data-layout.md``.

Mandatory tags written at run open: ``graphids.{phase, run_dir, dataset,
group, variant, seed, model_type, scale}`` + ``slurm.{job_id, cluster_name}``
when set. Experiment shape: ``graphids/{dataset}/{group}``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import lightning.pytorch as pl
import mlflow
from mlflow.entities import LoggedModelOutput, Metric
from mlflow.tracking import MlflowClient


def configure_tracking_uri() -> None:
    """Set the MLflow tracking URI to the graphids default if unset.

    ``mlflow.config.is_tracking_uri_set()`` natively checks both
    ``$MLFLOW_TRACKING_URI`` and any prior ``set_tracking_uri`` call —
    if either is in effect, MLflow already knows where to talk. Otherwise
    fall back to the canonical ``sqlite:///{lake_root}/mlflow.db``.
    """
    if mlflow.config.is_tracking_uri_set():
        return
    from graphids.config.catalog import lake_root

    mlflow.set_tracking_uri(f"sqlite:///{lake_root().rstrip('/')}/mlflow.db")

from graphids.blueprint import TrainRow

def _find_logged_model(
    client: MlflowClient,
    experiment_id: str,
    *,
    dataset: str,
    group: str,
    variant: str,
    seed: int | str,
) -> Any | None:
    """Resolve the catalog ``LoggedModel`` for a hyperparam-identity tuple.

    The catalog key is ``(dataset, group, variant, seed)`` — same hyperparams
    map to the same ``model_id`` across re-fits. Used at three sites:
    fit-start (create-or-find), fit-end (finalize), test-open (link).
    """
    res = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.`graphids.dataset` = '{dataset}' AND "
            f"tags.`graphids.group` = '{group}' AND "
            f"tags.`graphids.variant` = '{variant}' AND "
            f"tags.`graphids.seed` = '{seed}'"
        ),
        max_results=1,
    )
    return res[0] if res else None


def _find_logged_model_by_ckpt(
    client: MlflowClient, dataset: str, ckpt_path: str
) -> Any | None:
    """Resolve a ``LoggedModel`` from the ckpt path it cataloged.

    Used for upstream lineage: a fusion fit knows the vgae/gat ckpt paths
    from ``row.upstreams`` and needs each upstream's ``model_id`` to call
    ``log_inputs``. Per-axis experiments (``graphids/{dataset}/{group}``)
    mean a fusion fit and its upstream vgae/gat live in different
    experiments under the same dataset prefix; we enumerate via
    ``search_experiments(name LIKE ...)`` because passing
    ``experiment_ids=[]`` returns zero results, not "all".
    Most-recent match wins under resume-driven duplicates.
    """
    exps = client.search_experiments(filter_string=f"name LIKE 'graphids/{dataset}/%'")
    if not exps:
        return None
    res = client.search_logged_models(
        experiment_ids=[e.experiment_id for e in exps],
        filter_string=f"tags.`graphids.ckpt_path` = '{ckpt_path}'",
        order_by=[{"field_name": "creation_time", "ascending": False}],
        max_results=1,
    )
    return res[0] if res else None


def identity_tags(row: TrainRow, phase: str) -> dict[str, str]:
    """Tag dict for the run that an ``MLFlowLogger`` will open.

    Mandatory graphids identity (per ``data-layout.md``) + phase + SLURM
    context when present. ``MLFlowLogger(tags=identity_tags(row, "fit"))``
    opens the run with these tags on first lazy access.
    """
    m = row.meta
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
    return tags


class MLflowTrainingCallback(pl.Callback):
    """Catalog ``LoggedModel`` lifecycle + graphids-specific run state.

    Per-epoch metric forwarding is owned by Lightning's
    :class:`lightning.pytorch.loggers.MLFlowLogger` (wired in
    :func:`graphids.orchestrate._make_trainer`). This callback owns the
    bits the logger doesn't:

    - ``on_train_start`` — create-or-find the catalog ``LoggedModel`` keyed
      by ``(dataset, group, variant, seed)``. Same hyperparams → same
      ``model_id`` across re-fits.
    - ``on_train_epoch_end`` first call — stamp graphids-specific run state
      (params: ``budget_target_bytes`` / ``num_workers`` / ``prefetch_factor``;
      tag: ``num_workers_source``). Deferred to epoch 0 so DM
      ``_autosize_info`` and probe state are populated.
    - ``on_fit_end`` — final summary metric ``graphids.peak_vram_mb`` (with
      ``model_id`` so it lands on the LM), tag ``graphids.budget_binding``,
      and finalize the LM (``set_tags`` ckpt_path + sha256, mark READY).

    Run_id and the MLflow client are pulled from ``trainer.logger`` rather
    than env vars or constructor args — Lightning's logger is the single
    source of truth for both.

    Things MLflow / Lightning give for free, NOT logged here: per-epoch
    ``trainer.callback_metrics`` (logger), peak system VRAM time-series
    (system-metrics sampler), epochs_run (``max(step)`` on any metric),
    run wall-time (``end_time - start_time``).
    """

    def __init__(self) -> None:
        # run_id + client are rebound in ``on_train_start`` from
        # ``trainer.logger``. Eager defaults keep the types simple — the
        # tracking URI is already set process-wide by ``runtime.setup``,
        # so a stub MlflowClient() costs nothing.
        self.run_id: str = ""
        self._client: MlflowClient = MlflowClient()
        self._run_config_stamped = False
        self._lm_model_id: str | None = None

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Bind run_id from MLFlowLogger; create-or-find catalog ``LoggedModel``."""
        self.run_id = trainer.logger.run_id
        # ``MLFlowLogger.experiment`` returns the underlying ``MlflowClient``.
        self._client = trainer.logger.experiment

        run = self._client.get_run(self.run_id)
        run_tags = run.data.tags
        existing = _find_logged_model(
            self._client,
            run.info.experiment_id,
            dataset=run_tags["graphids.dataset"],
            group=run_tags["graphids.group"],
            variant=run_tags["graphids.variant"],
            seed=run_tags["graphids.seed"],
        )
        if existing is not None:
            self._lm_model_id = existing.model_id
            return
        identity_tags = {
            "graphids.dataset": run_tags["graphids.dataset"],
            "graphids.group": run_tags["graphids.group"],
            "graphids.variant": run_tags["graphids.variant"],
            "graphids.seed": run_tags["graphids.seed"],
        }
        params = {"graphids.run_dir": run_tags["graphids.run_dir"]}
        model_type = f"{type(pl_module).__module__}.{type(pl_module).__name__}"
        lm = self._client.create_logged_model(
            experiment_id=run.info.experiment_id,
            name=run.info.run_name,
            source_run_id=self.run_id,
            model_type=model_type,
            tags=identity_tags,
            params=params,
        )
        self._lm_model_id = lm.model_id

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if not self._run_config_stamped:
            self._stamp_run_config(trainer, pl_module)
            self._run_config_stamped = True

    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        """Bind the test run to its catalog ``LoggedModel`` (created by fit).

        Fail-fast if no LM exists for these hyperparams — testing without
        a successful prior fit has no catalog entity to attribute to.
        ``log_outputs`` declares the run→model lineage; ``model_id`` is
        cached so ``on_test_end`` can route final metrics onto the LM.
        """
        self.run_id = trainer.logger.run_id
        self._client = trainer.logger.experiment

        run = self._client.get_run(self.run_id)
        run_tags = run.data.tags
        lm = _find_logged_model(
            self._client,
            run.info.experiment_id,
            dataset=run_tags["graphids.dataset"],
            group=run_tags["graphids.group"],
            variant=run_tags["graphids.variant"],
            seed=run_tags["graphids.seed"],
        )
        if lm is None:
            raise RuntimeError(
                f"No LoggedModel for {run.info.run_name} "
                f"(dataset={run_tags['graphids.dataset']}, "
                f"group={run_tags['graphids.group']}, "
                f"variant={run_tags['graphids.variant']}, "
                f"seed={run_tags['graphids.seed']}). Run fit before test."
            )
        self._lm_model_id = lm.model_id
        self._client.log_outputs(
            run_id=self.run_id,
            models=[LoggedModelOutput(model_id=lm.model_id, step=0)],
        )

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        """Route final test metrics onto the LM via per-Metric ``model_id``."""
        if self._lm_model_id is None:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        if not metrics:
            return
        ts = int(time.time() * 1000)
        ms = [
            Metric(k, v, ts, 0, model_id=self._lm_model_id)
            for k, v in metrics.items()
        ]
        self._client.log_batch(self.run_id, metrics=ms)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        peak_mb = _peak_vram_mb(pl_module)
        if peak_mb > 0:
            # Single-point series — last==only, so threshold filtering on this
            # metric works as a one-shot "actual peak vs. probed budget" lookup.
            # Step = current_epoch so resumed fits append a new point per resume.
            # ``model_id`` routes the value onto the LM so the catalog has it.
            self._client.log_metric(
                self.run_id,
                "graphids.peak_vram_mb",
                peak_mb,
                step=trainer.current_epoch,
                model_id=self._lm_model_id,
            )
        budget = getattr(pl_module, "_budget_cache", None)
        if budget is not None:
            self._client.set_tag(self.run_id, "graphids.budget_binding", budget.binding)
        self._register_logged_model(trainer, pl_module)

    def _register_logged_model(self, trainer: Any, pl_module: Any) -> None:
        """Finalize the catalog ``LoggedModel``: stamp ckpt tags, mark READY.

        The LM was created in ``on_train_start`` (so its ``model_id`` is
        available for every per-epoch ``Metric``). Here we add the ckpt
        path + sha256 (only known at fit-end) and promote to READY. No
        artifact bytes are uploaded — the path tag is the load-bearing
        link, per data-layout.md.
        """
        from pathlib import Path

        if self._lm_model_id is None:
            return
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb is None or not getattr(ckpt_cb, "best_model_path", ""):
            return
        best_path = Path(ckpt_cb.best_model_path)
        if not best_path.exists():
            return
        sha_sidecar = best_path.with_suffix(best_path.suffix + ".sha256")
        sha = sha_sidecar.read_text().strip() if sha_sidecar.exists() else ""
        self._client.set_logged_model_tags(
            self._lm_model_id,
            {"graphids.ckpt_path": str(best_path), "graphids.ckpt_sha256": sha},
        )
        self._client.finalize_logged_model(self._lm_model_id, status="READY")

    def _stamp_run_config(self, trainer: Any, pl_module: Any) -> None:
        """One-shot at epoch 0 — DM and probe state are populated by then.

        Idempotency: params reject same-key/different-value rewrites. Resume
        with a different cluster (different probe target_bytes) would error
        loudly here — surfaced rather than silently shadowed.
        """
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return

        # Budget target — read off the model (probe owner after the disentangle).
        budget = getattr(pl_module, "_budget_cache", None)
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

    All keys map to the schema written by :func:`identity_tags`. ``status``
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


