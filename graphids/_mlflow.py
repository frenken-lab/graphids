"""MLflow integration for graphids — callback + LM lookup + filter composer.

Tracking URI + system-metrics sampler are configured process-once in
``orchestrate.setup``. Run lifecycle is owned by Lightning's
``MLFlowLogger`` (wired in ``orchestrate._make_trainer``); this module
plugs the bits that aren't its concern: catalog ``LoggedModel`` lifecycle,
graphids-specific run-config stamping, peak-VRAM at fit-end.

What MLflow / Lightning give for free, NOT logged here:
- per-epoch ``trainer.callback_metrics`` → ``MLFlowLogger``
- system VRAM time-series → MLflow system-metrics sampler
- ``epochs_run`` / wall-time → derivable from any metric step / run timestamps

Every MLflow API call here goes through ``MlflowClient`` directly:
``search_logged_models``, ``search_experiments``, ``create_logged_model``,
``log_outputs``, ``log_batch``, ``set_logged_model_tags``,
``finalize_logged_model``, ``log_param`` / ``log_metric`` / ``set_tag``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import mlflow
from mlflow.entities import LoggedModelOutput, Metric
from mlflow.tracking import MlflowClient

from graphids.plan.schema import TrainRow

# Identity attribute names on ``TrainRow.meta``; surfaced as ``graphids.{key}`` tags.
_IDENTITY_KEYS = ("dataset", "group", "variant", "seed", "model_type", "scale")

# Search-filter kwargs → MLflow tag keys. Single source of truth used by both
# ``build_search_filter`` (run search) and ``_find_logged_model`` (LM search).
_TAG_KEYS = {
    "dataset": "graphids.dataset",
    "group": "graphids.group",
    "variant": "graphids.variant",
    "seed": "graphids.seed",
    "phase": "graphids.phase",
    "cluster": "slurm.cluster_name",
    "plan_id": "graphids.plan_id",
    "plan_module": "graphids.plan_module",
    "git_sha": "graphids.git_sha",
    "row_name": "graphids.row_name",
}


def configure_tracking_uri() -> None:
    """Default URI to ``sqlite:///{lake_root}/mlflow.db`` if neither
    ``$MLFLOW_TRACKING_URI`` nor a prior ``set_tracking_uri`` is in effect.
    """
    if mlflow.config.is_tracking_uri_set():
        return
    from graphids.paths import lake_root

    mlflow.set_tracking_uri(f"sqlite:///{lake_root().rstrip('/')}/mlflow.db")


def identity_tags(row: TrainRow, phase: str) -> dict[str, str]:
    """Mandatory run-open tags. Reproduction contract is

        ``git checkout <graphids.git_sha> && graphids run <graphids.plan_module>
            --dataset <graphids.dataset> --seed <graphids.seed>
            --filter <graphids.row_name>``

    so all five tags must be present on every fit/test run.
    """
    tags = {
        "graphids.phase": phase,
        "graphids.run_dir": row.identity.run_dir,
        "graphids.plan_id": row.plan_id,
        "graphids.plan_module": row.plan_module,
        "graphids.git_sha": row.git_sha,
        "graphids.row_name": row.name,
        **{f"graphids.{k}": str(getattr(row.meta, k)) for k in _IDENTITY_KEYS},
    }
    if jid := os.environ.get("SLURM_JOB_ID"):
        tags["slurm.job_id"] = jid
    if cluster := os.environ.get("SLURM_CLUSTER_NAME"):
        tags["slurm.cluster_name"] = cluster
    return tags


def build_search_filter(
    *,
    dataset: str | None = None,
    group: str | None = None,
    variant: str | None = None,
    seed: int | None = None,
    phase: str | None = None,
    cluster: str | None = None,
    plan_id: str | None = None,
    plan_module: str | None = None,
    git_sha: str | None = None,
    row_name: str | None = None,
    status: str | None = None,
) -> str:
    """Compose ``MlflowClient.search_runs(filter_string=...)`` from
    graphids tag predicates. ``status`` is run-level (``attributes.status``);
    everything else is a backtick-quoted tag.
    """
    items = {
        "dataset": dataset,
        "group": group,
        "variant": variant,
        "seed": str(seed) if seed is not None else None,
        "phase": phase,
        "cluster": cluster,
        "plan_id": plan_id,
        "plan_module": plan_module,
        "git_sha": git_sha,
        "row_name": row_name,
    }
    parts = [f"tags.`{_TAG_KEYS[k]}` = '{v}'" for k, v in items.items() if v is not None]
    if status is not None:
        parts.append(f"attributes.status = '{status}'")
    return " and ".join(parts)


def _find_logged_model(
    client: MlflowClient,
    experiment_id: str,
    *,
    dataset: str,
    group: str,
    variant: str,
    seed: int | str,
) -> Any | None:
    """Catalog LM keyed by ``(dataset, group, variant, seed)``. Same hyperparams
    map to the same ``model_id`` across re-fits.
    """
    res = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=build_search_filter(
            dataset=dataset,
            group=group,
            variant=variant,
            seed=int(seed) if isinstance(seed, str) else seed,
        ),
        max_results=1,
    )
    return res[0] if res else None


def _find_logged_model_by_ckpt(client: MlflowClient, dataset: str, ckpt_path: str) -> Any | None:
    """LM resolved by ``graphids.ckpt_path`` tag across per-axis experiments
    ``graphids/{dataset}/*``. Most-recent wins under resume duplicates.

    Per-axis experiments (``graphids/{dataset}/{group}``) mean a fusion fit
    and its upstream vgae/gat live in different experiments under the same
    dataset prefix; ``search_experiments(name LIKE ...)`` enumerates them
    because ``experiment_ids=[]`` returns zero, not "all".
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


def _peak_vram_mb(model: Any) -> float:
    """Peak CUDA-allocator high-water mark on the model's device, in MB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        dev = getattr(model, "device", None)
        idx = dev.index if dev is not None and dev.index is not None else 0
        return torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
    except (AttributeError, RuntimeError):
        return torch.cuda.max_memory_allocated() / (1024 * 1024)


class MLflowTrainingCallback(pl.Callback):
    """Catalog ``LoggedModel`` lifecycle + graphids-specific run state.

    - ``on_train_start``: create-or-find LM keyed by identity tags.
    - ``on_train_epoch_end``: stamp budget_target_bytes / num_workers /
      prefetch_factor / num_workers_source at epoch 0 (DM populates
      ``_autosize_info`` and probe state by then).
    - ``on_fit_end``: log peak_vram_mb (with ``model_id`` so it lands on the
      LM), tag ``budget_binding``, finalize LM (ckpt_path + sha256, READY).
    - ``on_test_start``: bind run to the LM created by fit (fail-fast if
      missing). ``log_outputs`` declares run→model lineage.
    - ``on_test_end``: route final test metrics onto the LM via per-Metric
      ``model_id``.

    ``run_id`` and the ``MlflowClient`` are pulled from ``trainer.logger``
    every callback (Lightning's logger is the SoT for both).
    """

    def __init__(self, *, system_metrics_interval: int = 10) -> None:
        self._lm_model_id: str | None = None
        self._stamped = False
        # SystemMetricsMonitor is keyed by run_id and runs as a daemon thread.
        # Lightning's MLFlowLogger creates runs via MlflowClient (not the fluent
        # mlflow.start_run path), so mlflow.enable_system_metrics_logging()
        # alone never spawns a sampler — fluent.start_run is the only place
        # that reads MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING. Wire it manually.
        self._sysmon: Any = None
        self._sm_interval = system_metrics_interval

    def _start_sysmon(self, run_id: str) -> None:
        try:
            from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
        except ImportError:
            return  # mlflow build without system_metrics; no-op
        self._sysmon = SystemMetricsMonitor(run_id, sampling_interval=self._sm_interval)
        self._sysmon.start()

    def _stop_sysmon(self) -> None:
        if self._sysmon is not None:
            try:
                self._sysmon.finish()
            finally:
                self._sysmon = None

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _bind(trainer: Any) -> tuple[str, MlflowClient]:
        return trainer.logger.run_id, trainer.logger.experiment

    @staticmethod
    def _identity(client: MlflowClient, run_id: str) -> tuple[Any, dict[str, str]]:
        run = client.get_run(run_id)
        return run, {
            k: run.data.tags[f"graphids.{k}"] for k in ("dataset", "group", "variant", "seed")
        }

    # ── train ────────────────────────────────────────────────────────
    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        run_id, client = self._bind(trainer)
        run, ident = self._identity(client, run_id)
        existing = _find_logged_model(client, run.info.experiment_id, **ident)
        if existing is not None:
            self._lm_model_id = existing.model_id
        else:
            lm = client.create_logged_model(
                experiment_id=run.info.experiment_id,
                name=run.info.run_name,
                source_run_id=run_id,
                model_type=f"{type(pl_module).__module__}.{type(pl_module).__name__}",
                tags={f"graphids.{k}": v for k, v in ident.items()},
                params={"graphids.run_dir": run.data.tags["graphids.run_dir"]},
            )
            self._lm_model_id = lm.model_id
        self._start_sysmon(run_id)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self._stamped:
            return
        run_id, client = self._bind(trainer)
        budget = getattr(pl_module, "_budget_cache", None)
        if budget is not None and budget.target_bytes > 0:
            client.log_param(run_id, "graphids.budget_target_bytes", str(budget.target_bytes))
        info = getattr(getattr(trainer, "datamodule", None), "_autosize_info", None)
        if info is not None:
            client.log_param(run_id, "graphids.num_workers", str(info["num_workers"]))
            client.log_param(run_id, "graphids.prefetch_factor", str(info["prefetch_factor"]))
            client.set_tag(run_id, "graphids.num_workers_source", info["source"])
        self._stamped = True

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self._stop_sysmon()
        run_id, client = self._bind(trainer)
        peak_mb = _peak_vram_mb(pl_module)
        if peak_mb > 0:
            client.log_metric(
                run_id,
                "graphids.peak_vram_mb",
                peak_mb,
                step=trainer.current_epoch,
                model_id=self._lm_model_id,
            )
        budget = getattr(pl_module, "_budget_cache", None)
        if budget is not None:
            client.set_tag(run_id, "graphids.budget_binding", budget.binding)

        if self._lm_model_id is None:
            return
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        best = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
        if not best or not Path(best).exists():
            return
        sha_path = Path(best).with_suffix(Path(best).suffix + ".sha256")
        sha = sha_path.read_text().strip() if sha_path.exists() else ""
        client.set_logged_model_tags(
            self._lm_model_id,
            {"graphids.ckpt_path": best, "graphids.ckpt_sha256": sha},
        )
        client.finalize_logged_model(self._lm_model_id, status="READY")

    # ── test ─────────────────────────────────────────────────────────
    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        run_id, client = self._bind(trainer)
        run, ident = self._identity(client, run_id)
        lm = _find_logged_model(client, run.info.experiment_id, **ident)
        if lm is None:
            raise RuntimeError(
                f"no LoggedModel for {run.info.run_name} (ident={ident}); fit before test"
            )
        self._lm_model_id = lm.model_id
        client.log_outputs(run_id=run_id, models=[LoggedModelOutput(model_id=lm.model_id, step=0)])
        self._start_sysmon(run_id)

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        self._stop_sysmon()
        if self._lm_model_id is None:
            return
        run_id, client = self._bind(trainer)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        if not metrics:
            return
        ts = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[Metric(k, v, ts, 0, model_id=self._lm_model_id) for k, v in metrics.items()],
        )
