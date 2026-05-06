"""Dict → objects → Lightning bridge.

Lightning owns: train/val loop, AMP, gradient clipping, optimizer state,
scheduler, callback lifecycle, MLflow run lifecycle (via ``MLFlowLogger``),
SLURM preempt-resume (via ``SLURMEnvironment(auto_requeue=True,
requeue_signal=SIGUSR2)`` — calls ``scontrol requeue``, same job ID,
downstream ``afterok`` deps stay valid).

graphids owns:
- ``dm.setup("fit")`` BEFORE ``trainer.fit`` so ``model.prepare_from_datamodule``
  reads vocab/channel sizes.
- ``model.to(device)`` BEFORE dataloader build so the VRAM probe sees the
  right device.
- VGAE/DGI calibration via ``model.on_test_setup(dm, device)`` after ckpt
  load, before ``trainer.test``.
- Upstream LM lineage via ``UpstreamLineageCallback.on_fit_start`` —
  registered as a callback so the lineage write happens inside Lightning's
  lifecycle after the run is open, not as an explicit pre-fit side-effect.
"""

from __future__ import annotations

import functools
import importlib
import multiprocessing
import os
import signal
from typing import Any

import lightning.pytorch as pl
import torch
import torch_geometric
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from mlflow.entities import LoggedModelInput

from graphids._fs import atomic_load
from graphids._mlflow import _find_logged_model_by_ckpt, identity_tags
from graphids.cli.app import configure_logging
from graphids.plan.schema import AnalyzeRow, CacheRow, ExtractRow, Row, TrainRow
from graphids.core.models.base import strip_orig_mod_prefix


@functools.cache
def _ensure_runtime() -> None:
    """spawn mp + ``file_system`` sharing + structlog + strict cuBLAS reductions.
    CUDA-safe + OSC-safe."""
    import torch.multiprocessing

    configure_logging()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Disallow reduced-precision intermediate reductions in fp32 cuBLAS GEMM.
    # On V100 the default heuristic picks a kernel that saturates to fp32 max
    # for our [300K, 64] @ [64, 1791] sanity probe shape, producing NaN/Inf
    # in canid_logits/nbr_logits despite finite z and finite weights. See
    # docs/empirical-notes/2026-05-06-vgae-cublas-overflow-v100.md.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _instantiate(spec: dict[str, Any]) -> Any:
    """Build ``{class_path, init_args}``; recurses on nested ``class_path`` blocks."""
    rec = lambda v: _instantiate(v) if isinstance(v, dict) and "class_path" in v else v  # noqa: E731
    init_args = {k: rec(v) for k, v in spec.get("init_args", {}).items()}
    mod, _, attr = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(mod), attr)(**init_args)


def _build(row: TrainRow) -> tuple[Any, Any, list, dict]:
    rc = row.rendered_config.model_dump()
    callbacks = [_instantiate(spec) for spec in rc["callbacks"].values()]
    return (
        _instantiate(rc["model"]),
        _instantiate(rc["data"]),
        callbacks,
        {k: v for k, v in rc["trainer"].items() if k != "callbacks"},
    )


def _load_state_into_model(ckpt_path: str, model: torch.nn.Module) -> dict:
    """Restore weights from ckpt; align keys across compile prefix variants."""
    ckpt = atomic_load(ckpt_path, map_location="cpu", weights_only=True)
    state = strip_orig_mod_prefix(ckpt.get("state_dict", ckpt))
    remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict()}
    state = {remap.get(k, k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint(ckpt)
    return ckpt


class UpstreamLineageCallback(Callback):
    """Write MLflow ``LoggedModelInput`` lineage edges at fit start.

    Lives inside the Lightning callback lifecycle so the write happens
    after the MLflow run is open (no need to dance around lazy
    ``logger.experiment`` access from outside the trainer).
    """

    def __init__(self, row: TrainRow) -> None:
        self._row = row

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            return
        client = logger.experiment
        inputs, missing = [], []
        for u in self._row.upstreams:
            lm = _find_logged_model_by_ckpt(client, self._row.meta.dataset, u.ckpt_path)
            if lm is None:
                missing.append(f"{u.role}={u.ckpt_path}")
            else:
                inputs.append(LoggedModelInput(model_id=lm.model_id))
        if inputs:
            client.log_inputs(run_id=logger.run_id, models=inputs)
        if missing:
            import structlog

            structlog.get_logger(__name__).warning(
                "upstream_lm_missing",
                run_id=logger.run_id,
                dataset=self._row.meta.dataset,
                missing=missing,
            )


def _prepare(row: TrainRow) -> tuple[Any, Any, list, dict, torch.device]:
    """Shared bootstrap: seed, build, dm.setup, prepare_from_datamodule.

    `dm.setup` is idempotent and stage-agnostic in this project (always
    materializes train/val/test on first call), so we don't pass a stage.
    """
    torch_geometric.seed_everything(row.meta.seed)
    model, dm, callbacks, kw = _build(row)
    device = torch.device(
        "cpu" if kw.get("accelerator") == "cpu"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    dm.setup(None)
    model.prepare_from_datamodule(dm)
    return model, dm, callbacks, kw, device


def _trainer_kwargs(callbacks: list, row: TrainRow, kw: dict, phase: str) -> dict[str, Any]:
    plugins: list[Any] = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR2))
    logger = MLFlowLogger(
        experiment_name=f"graphids/{row.meta.dataset}/{row.meta.group}",
        run_name=row.identity.run_name,
        tags=identity_tags(row, phase),
    )
    return dict(
        callbacks=callbacks,
        logger=logger,
        plugins=plugins or None,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        **kw,
    )


def train(row: TrainRow, *, ckpt_path: str | None = None) -> None:
    model, dm, callbacks, kw, device = _prepare(row)
    model.to(device)  # before dataloader build — VRAM probe reads model.device
    if row.upstreams:
        callbacks.append(UpstreamLineageCallback(row))
    pl.Trainer(**_trainer_kwargs(callbacks, row, kw, "fit")).fit(
        model, datamodule=dm, ckpt_path=ckpt_path
    )


def evaluate(row: TrainRow, *, ckpt_path: str | None = None) -> dict[str, float]:
    if ckpt_path is not None and not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"test row {row.name!r}: ckpt_path does not exist: {ckpt_path}\n"
            f"  → fit row may not have completed yet, or saved no best_model.ckpt "
            f"(EarlyStopping monitor never improved). Check `gx plans show <plan_id>` "
            f"and `gx plans where <plan_id> --row {row.name.removesuffix('-test') or row.name}`."
        )
    model, dm, callbacks, kw, device = _prepare(row)

    if ckpt_path:
        _load_state_into_model(ckpt_path, model)

    # VGAE/DGI calibration buffers (z-norm stats, SVDD center) refit at
    # test-start from fit-phase data — not persisted in state_dict. The dm
    # already has train/val/test populated from _prepare.
    model.to(device)

    # Build trainer BEFORE on_test_setup so the budget probe inside
    # val_dataloader can reach `dm.trainer.lightning_module`. Lightning's
    # public attach happens in trainer.test(); pre-wire manually.
    trainer = pl.Trainer(**_trainer_kwargs(callbacks, row, kw, "test"))
    dm.trainer = trainer
    trainer.strategy.connect(model)

    model.on_test_setup(dm, device)

    # ckpt_path NOT passed — already restored above so calibration saw
    # trained weights. Lightning's ckpt-load would happen too late.
    trainer.test(model, datamodule=dm)
    return {k: float(v) for k, v in trainer.callback_metrics.items()}


def extract(row: ExtractRow) -> None:
    """One-shot fusion-feature extraction. Pure data transform — no MLflow run."""
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


def analyze(row: AnalyzeRow) -> None:
    """Run the per-checkpoint artifact pipeline on a single ckpt."""
    from graphids.core.artifacts import Analyzer

    Analyzer(row).run()


def cache(row: CacheRow) -> None:
    """One-shot dataset cache build. Idempotent on the partition dir."""
    from graphids.core.data.datasets.can_bus import CANBusSource

    CANBusSource(
        name=row.dataset,
        seed=row.seed,
        window_size=row.window_size,
        stride=row.stride,
        val_fraction=row.val_fraction,
        vocab_scope=row.vocab_scope,
    ).build()


def run_row(row: Row, *, ckpt_path: str | None = None) -> None:
    _ensure_runtime()
    if isinstance(row, ExtractRow):
        extract(row)
        return
    if isinstance(row, AnalyzeRow):
        analyze(row)
        return
    if isinstance(row, CacheRow):
        cache(row)
        return
    {"fit": train, "test": evaluate}[row.action](row, ckpt_path=ckpt_path)
