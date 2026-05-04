"""Dict→objects→Lightning bridge. importlib-instantiates from
``rendered_config``, opens/closes the MLflow run, and dispatches on
``row.action``.

Lightning owns the train/val loop, AMP autocast, gradient clipping,
optimizer state, scheduler stepping, and the callback lifecycle.
graphids only owns:

- ``dm.setup("fit")`` BEFORE ``trainer.fit`` so ``prepare_from_datamodule``
  can read DM-supplied vocab/channel sizes for the lazy ``_build()``.
  ``dm.setup`` is idempotent — Lightning re-invokes it inside fit harmlessly.
- ``model.prepare_from_datamodule(dm)`` lazy-builds parameters before
  Lightning's ``configure_optimizers`` snapshots the parameter list.
- VGAE/DGI calibration via ``model.on_test_setup(dm, device)`` after
  ckpt load, before the test loop.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import signal
import sys
from typing import Any, Literal

import lightning.pytorch as pl
import structlog
import torch
import torch_geometric
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.entities import LoggedModelInput

from graphids._fs import atomic_load
from graphids._mlflow import _find_logged_model_by_ckpt, identity_tags
from graphids.blueprint import ExtractRow, Row, TrainRow
from graphids.core.models.base import strip_orig_mod_prefix


def _instantiate(spec: dict[str, Any]) -> Any:
    """Build ``{class_path, init_args}``; recurses on nested class_path blocks
    in init_args (e.g. GAT's ``loss_fn``)."""
    rec = lambda v: _instantiate(v) if isinstance(v, dict) and "class_path" in v else v  # noqa: E731
    ia = {k: rec(v) for k, v in spec.get("init_args", {}).items()}
    mod, _, attr = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(mod), attr)(**ia)


def _build(row: TrainRow) -> tuple[Any, Any, list, dict]:
    """Instantiate model + datamodule + callbacks + trainer kwargs.

    All callbacks go through ``_instantiate``; ``MLflowTrainingCallback``
    pulls run_id + client from ``trainer.logger`` in its ``on_train_start``
    hook (the trainer's ``MLFlowLogger`` is the single source of truth).
    """
    rc = row.rendered_config
    model = _instantiate(rc["model"])
    datamodule = _instantiate(rc["data"])
    callbacks = [_instantiate(spec) for spec in rc.get("callbacks", {}).values()]
    trainer_kwargs = {k: v for k, v in rc["trainer"].items() if k != "callbacks"}
    return model, datamodule, callbacks, trainer_kwargs


def _device_from_kwargs(trainer_kwargs: dict) -> torch.device:
    """Resolve the device pl.Trainer will pick, so calibration / ckpt-load
    can run on the same one before fit."""
    if trainer_kwargs.get("accelerator") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_trainer(
    callbacks: list, trainer_kwargs: dict, logger: MLFlowLogger
) -> pl.Trainer:
    """``pl.Trainer`` with graphids defaults that don't belong in jsonnet.

    - ``logger``: Lightning's MLFlowLogger drives the run lifecycle (lazy-
      open with our tags on first access, ``set_terminated`` via
      ``finalize`` on teardown — FINISHED on success, FAILED on exception).
      Graphids owns only the LM lifecycle bits in ``MLflowTrainingCallback``.
    - ``enable_progress_bar=False``: SLURM logs go to ``*_log.err``, the
      tqdm bar would smear stderr.
    - ``num_sanity_val_steps=0``: our val path runs full epochs and our
      DM constructs val loaders only after ``setup``; the default sanity
      pass races that.
    """
    return pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        **trainer_kwargs,
    )


def _load_state_into_model(ckpt_path: str, model: torch.nn.Module) -> dict:
    """Read a graphids/Lightning ckpt, restore weights into ``model``,
    fire ``on_load_checkpoint``. Returns the raw ckpt dict.

    ``strict=False`` tolerates removed buffers (e.g. DGI ``svdd_calibrated``,
    dropped when centroid fit moved from state_dict to test-start).
    """
    ckpt = atomic_load(ckpt_path, map_location="cpu", weights_only=True)
    state = strip_orig_mod_prefix(ckpt.get("state_dict", ckpt))
    # Align ckpt to target's compile-prefix convention.
    remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict().keys()}
    state = {remap.get(k, k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint(ckpt)
    return ckpt


def _logger_for(row: TrainRow, phase: str) -> MLFlowLogger:
    """Construct an ``MLFlowLogger`` carrying graphids identity tags.

    The tracking URI is set process-wide by :func:`setup`.
    The logger lazily opens the run on first ``logger.run_id`` /
    ``logger.experiment`` access with the supplied tags, and finalizes
    (FINISHED/FAILED) automatically in Lightning's teardown.
    """
    return MLFlowLogger(
        experiment_name=f"graphids/{row.meta.dataset}/{row.meta.group}",
        run_name=row.identity.run_name,
        tags=identity_tags(row, phase),
    )


def _log_upstream_inputs(logger: MLFlowLogger, row: TrainRow) -> None:
    """Declare ``row.upstreams`` as model inputs of the open run.

    Lookup each upstream's catalog ``LoggedModel`` by ckpt_path and call
    ``log_inputs`` so lineage queries work without a custom tag scheme.
    Missing upstreams are warn-logged and skipped — degraded lineage is
    preferable to refusing to fit when MLflow is the only complaint.
    """
    if not row.upstreams:
        return
    client = logger.experiment  # MlflowClient
    inputs, missing = [], []
    for u in row.upstreams:
        lm = _find_logged_model_by_ckpt(client, row.meta.dataset, u.ckpt_path)
        if lm is None:
            missing.append(f"{u.role}={u.ckpt_path}")
            continue
        inputs.append(LoggedModelInput(model_id=lm.model_id))
    if inputs:
        client.log_inputs(run_id=logger.run_id, models=inputs)
    if missing:
        from structlog import get_logger

        get_logger(__name__).warning(
            "upstream_lm_missing",
            run_id=logger.run_id,
            dataset=row.meta.dataset,
            missing=missing,
        )


def train(row: TrainRow, *, ckpt_path: str | None = None) -> None:
    """Fit one row. ``MLFlowLogger`` owns the run lifecycle."""
    torch_geometric.seed_everything(row.meta.seed)

    model, dm, callbacks, trainer_kwargs = _build(row)
    device = _device_from_kwargs(trainer_kwargs)

    # Setup the DM BEFORE Lightning so ``prepare_from_datamodule`` can
    # read vocab/channel sizes. ``dm.setup`` is idempotent — Lightning
    # will re-invoke it inside ``trainer.fit`` to no effect.
    dm.setup("fit")
    model.prepare_from_datamodule(dm)
    # Move to device BEFORE dataloader construction so the VRAM probe
    # (``model.compute_budget`` → ``budget.probe``) reads the right
    # ``model.device`` on its first call.
    model.to(device)

    logger = _logger_for(row, phase="fit")
    # Trigger lazy run open so the run_id is available for log_inputs
    # before fit starts. After this, ``logger.run_id`` is a real id.
    _log_upstream_inputs(logger, row)

    trainer = _make_trainer(callbacks, trainer_kwargs, logger)
    # Pass datamodule so Lightning wires ``dm.trainer = trainer`` — the
    # probe path (``_ensure_budget``) reads ``self.trainer.lightning_module``.
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


def evaluate(row: TrainRow, *, ckpt_path: str | None = None) -> dict[str, float]:
    """Test one row. ``MLFlowLogger`` owns the run lifecycle; the LM
    callback's ``on_test_start`` / ``on_test_end`` hooks attach the test
    run to the catalog LoggedModel and route final metrics onto the LM.
    """
    torch_geometric.seed_everything(row.meta.seed)

    model, dm, callbacks, trainer_kwargs = _build(row)
    device = _device_from_kwargs(trainer_kwargs)

    dm.setup("test")
    model.prepare_from_datamodule(dm)

    # Restore weights BEFORE on_test_setup — score-based detectors
    # (VGAE/DGI) need the trained encoder to fit calibration buffers.
    if ckpt_path:
        _load_state_into_model(ckpt_path, model)

    # VGAE/DGI calibration buffers (z-norm stats, SVDD center) are
    # deterministic functions of (trained encoder, fit-phase data) and
    # are NOT persisted through state_dict — refit at test-start.
    dm.setup("fit")
    model.to(device)
    model.on_test_setup(dm, device)

    logger = _logger_for(row, phase="test")
    trainer = _make_trainer(callbacks, trainer_kwargs, logger)
    # ckpt_path NOT passed — we already restored above so the calibration
    # hook saw trained weights. Lightning's ckpt-load would happen too late.
    trainer.test(model, datamodule=dm)

    return {k: float(v) for k, v in trainer.callback_metrics.items()}


def extract(row: ExtractRow) -> None:
    """One-shot fusion-feature extraction. Idempotent on ``row.output_dir``.

    Pure data transform — no MLflow run. The fusion fit that consumes
    these states logs the upstream vgae/gat ``LoggedModel``s as inputs in
    its own run, which is sufficient for catalog lineage. Adding a
    parallel extract-run would be redundant: the artifact (cached states)
    is identified by ``output_dir``, the producers are identified by the
    ckpt paths fusion's plan jsonnet pins.
    """
    from graphids.core.data.extract import extract_states

    extract_states(
        checkpoints=row.extractor_ckpts,
        dataset=row.dataset,
        output_dir=row.output_dir,
        max_samples=row.max_samples,
        max_val_samples=row.max_val_samples,
        batch_size=row.batch_size,
        seed=row.seed,
        window_size=row.window_size,
        stride=row.stride,
        val_fraction=row.val_fraction,
    )


# ---------------------------------------------------------------------------
# Process-level setup — structlog, spawn mp, CPU pinning, MLflow URI, preempt.
# ---------------------------------------------------------------------------

_SPAWN_SET = False
_THREADS_SET = False
_LOGGING_CONFIGURED = False

# SLURM env vars that carry useful identity for log queries. Auto-attached
# to every event by the structlog processor below.
_SLURM_KEYS = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_CPUS_PER_TASK": "slurm.cpus_per_task",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


def _slurm_context(_logger: Any, _method: str, event_dict: dict) -> dict:
    """structlog processor: attach SLURM env vars to every event."""
    for env, key in _SLURM_KEYS.items():
        if (v := os.environ.get(env)) and key not in event_dict:
            event_dict[key] = v
    return event_dict


def _configure_logging() -> None:
    """Install structlog → JSON sync stderr handler. Idempotent."""
    global _LOGGING_CONFIGURED  # noqa: PLW0603
    if _LOGGING_CONFIGURED:
        return
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _slurm_context,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True


def _ensure_spawn() -> None:
    global _SPAWN_SET  # noqa: PLW0603
    if _SPAWN_SET:
        return
    import torch.multiprocessing  # noqa: PLC0415

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")
    _SPAWN_SET = True


def _configure_cpu_threads() -> None:
    global _THREADS_SET  # noqa: PLW0603
    if _THREADS_SET:
        return
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    n = (int(slurm) if slurm and slurm.isdigit() else None) or os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some torch op already launched. Intra-op is still applied; that's
        # where CPU-bound ops get most of their wins.
        structlog.get_logger(__name__).warning("cpu_threads_interop_locked", intra_op=n)
    _THREADS_SET = True


def setup(*, mode: Literal["compute", "ops", "render"] = "compute") -> None:
    """Idempotent process-level setup. The single authoritative entry point.

    Every command (CLI subcommand, SLURM exec body, library caller) goes
    through this. The mode parameter is a property of the entry point —
    each call site declares what it needs:

    - ``"render"`` — pure render or schema-only commands (``graphids run``,
      ``graphids submit`` on login). Logging only — no torch import, no
      MLflow contact.
    - ``"ops"``    — read-only / lightweight ops that hit MLflow (export,
      compare, analyze). Logging + tracking URI.
    - ``"compute"`` (default) — full setup for SLURM jobs running
      ``trainer.fit/test``. Adds spawn mp, CPU pinning, system metrics.
    """
    _configure_logging()
    if mode == "render":
        return

    from graphids._mlflow import configure_tracking_uri  # noqa: PLC0415

    configure_tracking_uri()
    if mode == "ops":
        return

    _ensure_spawn()
    _configure_cpu_threads()
    _enable_system_metrics()


def _enable_system_metrics() -> None:
    """Turn on the MLflow system-metrics sampler (5s interval). Process-once."""
    import mlflow  # noqa: PLC0415

    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(5)


def register_preempt_handler(row: Any) -> None:
    """SIGUSR2 → re-submit row with afterany dep on the current SLURM job.

    Pairs with ``submit_row``'s ``--signal=USR2@N`` directive. SLURM sends
    SIGUSR2 N seconds before walltime; we save no extra state (the trainer's
    ``ModelCheckpoint`` already wrote ``last.ckpt`` on the most recent epoch
    end), then re-submit ``row`` with ``ckpt_path=last.ckpt``. No-op outside
    SLURM (no ``SLURM_JOB_ID`` env var).
    """
    jid = os.environ.get("SLURM_JOB_ID")
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if not jid or not cluster:
        return
    log = structlog.get_logger(__name__)

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        from graphids.slurm import submit_row  # noqa: PLC0415

        last_ckpt = f"{row.identity.run_dir}/checkpoints/last.ckpt"
        try:
            new_jid = submit_row(
                row,
                cluster=cluster,
                length="long",
                ckpt_path=last_ckpt,
                depends_on_afterany=jid,
            )
            log.info("preempt_resume_submitted", original_jid=jid, resume_jid=new_jid)
        except Exception as e:
            log.error("preempt_resume_failed", original_jid=jid, error=str(e))
        sys.exit(0)

    signal.signal(signal.SIGUSR2, _handler)


def run_row(row: Row, *, ckpt_path: str | None = None) -> None:
    setup()
    if isinstance(row, ExtractRow):
        extract(row)
        return
    register_preempt_handler(row)
    {"fit": train, "test": evaluate}[row.action](row, ckpt_path=ckpt_path)
