"""Pure-PyTorch training loop for GraphIDS.

Single-GPU only (project uses 1x V100). Handles:
- AMP via ``torch.amp.autocast`` + ``GradScaler(enabled=...)`` (no-op when disabled)
- Gradient clipping via ``clip_grad_norm_``
- ``automatic_optimization=False`` for RL fusion models
- Metric accumulation and logger dispatch (see :mod:`graphids.core._metric_acc`)
- Callback lifecycle (same hook names as Lightning)
- Checkpoint resume (schema in :mod:`graphids.core._ckpt`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from structlog import get_logger

from graphids.core._ckpt import load_state_into_model, restore_training_state
from graphids.core.callbacks import CallbackBase, EarlyStopping, ModelCheckpoint

Precision = Literal["32", "16-mixed", "bf16-mixed"]

_log = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Flat config matching the jsonnet ``trainer`` section keys."""

    max_epochs: int = 300
    precision: Precision = "16-mixed"
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    accelerator: str = "auto"
    devices: str | int = "auto"
    default_root_dir: str = ""


def _amp_dtype(precision: Precision) -> torch.dtype | None:
    """None ⇒ AMP disabled (run in fp32)."""
    if precision == "16-mixed":
        return torch.float16
    if precision == "bf16-mixed":
        return torch.bfloat16
    return None


class Trainer:
    """Single-GPU training loop with AMP, gradient clipping, and callbacks."""

    def __init__(
        self,
        config: TrainerConfig,
        callbacks: list | None = None,
        logger: list | bool | None = None,
    ) -> None:
        self.config = config
        self.callbacks: list[CallbackBase] = list(callbacks or [])
        self.loggers: list = list(logger) if isinstance(logger, list) else []

        # Public state (read by callbacks + model code)
        self.max_epochs: int = config.max_epochs
        self.current_epoch = self.global_step = 0
        self.callback_metrics: dict[str, float] = {}
        self.default_root_dir: str = config.default_root_dir
        self.should_stop: bool = False
        self.datamodule: Any = None
        # Underscore prefix matches ``_ckpt.build_checkpoint`` / callbacks.py
        # consumers — single attribute name, no public/private split.
        self._optimizers: list[torch.optim.Optimizer] = []
        self._schedulers: list[Any] = []

        self._device = (
            torch.device("cpu")
            if config.accelerator == "cpu"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Cached for callbacks.py (TauNormCallback, MLflowTrainingCallback) —
        # not used inside this file.
        self.checkpoint_callback: ModelCheckpoint | None = next(
            (c for c in self.callbacks if isinstance(c, ModelCheckpoint)), None
        )
        self.early_stopping_callback: EarlyStopping | None = next(
            (c for c in self.callbacks if isinstance(c, EarlyStopping)), None
        )

    # -- shared setup --------------------------------------------------------

    def _prep(self, model: nn.Module, datamodule: Any, stage: str, ckpt_path: str | None) -> None:
        """Wire DM, run setup, move model to device, optionally load weights.

        Every datamodule MUST implement ``bind(*, model, device)``. DMs that
        don't move batches (e.g. ``FusionDataModule``) implement it as a
        no-op — but the method must exist. Silent getattr fallbacks
        previously hid DM-wiring bugs as cpu/cuda mismatches at the first op.
        """
        self.datamodule = datamodule
        datamodule.bind(model=model, device=self._device)
        datamodule.setup(stage)
        model.setup(datamodule)
        # Must move AFTER setup(): _build() creates self.model and may wrap it
        # in torch.compile — moving before setup leaves the inner module on CPU.
        model.to(self._device)
        if ckpt_path:
            load_state_into_model(ckpt_path, model, self._device)
            _log.info("loaded_checkpoint", path=ckpt_path)

    def _amp_dtype(self) -> torch.dtype | None:
        if self._device.type != "cuda":
            return None
        return _amp_dtype(self.config.precision)

    # -- public API ----------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        datamodule: Any,
        ckpt_path: str | None = None,
    ) -> None:
        """Fit the model.

        Wires datamodule → device → model.setup → device.to(), then runs
        the train/val loop up to ``max_epochs`` or until a callback flips
        ``trainer.should_stop``. ``ckpt_path`` resumes weights +
        optimizer + scheduler + AMP scaler state; ``on_exception`` fires
        on any raise so callbacks can close MLflow runs cleanly before
        re-raising.
        """
        self._prep(model, datamodule, "fit", ckpt_path=None)

        opt, sched = model.build_optimizers(self.max_epochs)
        self._optimizers = [opt] if opt else []
        self._schedulers = [sched] if sched else []

        # GradScaler(enabled=False) is a complete no-op passthrough —
        # all methods become identity. No branching needed in the loop.
        amp_dtype = self._amp_dtype()
        use_amp = amp_dtype is not None
        # GradScaler is only correct for fp16; bf16 has fp32-equivalent
        # range and does not need loss scaling.
        scaler = torch.amp.GradScaler(enabled=(amp_dtype is torch.float16))

        if ckpt_path:
            ckpt = load_state_into_model(ckpt_path, model, self._device)
            restore_training_state(ckpt, self, opt, sched, scaler)
            _log.info(
                "resumed_from_checkpoint",
                path=ckpt_path,
                epoch=self.current_epoch,
                global_step=self.global_step,
            )

        self._dispatch("on_fit_start", model)
        self._log_hyperparams(model)

        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                self._train_one_epoch(model, datamodule, opt, scaler, amp_dtype)
                self._validate_one_epoch(model, datamodule, amp_dtype)

                self._dispatch("on_train_epoch_end", model)
                self._step_schedulers_safely()

                if self.should_stop:
                    _log.info("early_stopping", epoch=epoch)
                    break

        except BaseException as exc:
            self._dispatch("on_exception", model, exc)
            raise

        self._dispatch("on_fit_end", model)

    def test(
        self,
        model: nn.Module,
        datamodule: Any,
        ckpt_path: str | None = None,
    ) -> dict[str, float]:
        """Evaluate on all test dataloaders, return aggregated metrics.

        Multiple test loaders (e.g. one per attack subdir) are dispatched
        with a ``dataloader_idx`` so ``test_step`` can name metrics per
        subdir.
        """
        self._prep(model, datamodule, "test", ckpt_path)

        # Score-based detectors (VGAE/DGI) need their calibration buffers
        # (z-norm stats, SVDD center) refit at test-start — they're
        # deterministic functions of (trained encoder, fit-phase data) and
        # were NOT persisted through state_dict (callback/ckpt-save ordering
        # deadlock shipped uncalibrated ckpts; Cardinal jid 8772115). The
        # model's ``on_test_setup`` hook owns this; default is no-op.
        datamodule.setup("fit")
        model.on_test_setup(datamodule, self._device)

        model.eval()
        model.on_test_epoch_start()

        with torch.no_grad():
            test_loaders = datamodule.test_dataloader()
            if not isinstance(test_loaders, list):
                test_loaders = [test_loaders]
            for dl_idx, loader in enumerate(test_loaders):
                for batch_idx, batch in enumerate(loader):
                    model.test_step(batch, batch_idx, dataloader_idx=dl_idx)

        model.on_test_epoch_end()

        self.callback_metrics.update(model._metric_acc.compute())
        model._metric_acc.reset()

        return dict(self.callback_metrics)

    def predict(
        self,
        model: nn.Module,
        datamodule: Any,
        ckpt_path: str | None = None,
    ) -> list:
        """Run ``predict_step`` over every test loader and return the
        concatenated list. Setups with ``"predict"`` so datamodules can
        swap in a predict-specific loader.

        Uses ``inference_mode`` (stricter than ``no_grad``: disables view
        tracking + version counter bumps) — ``predict_step`` never backwards,
        so the stricter context is safe and ~5–10% faster on V100 inference.
        """
        self._prep(model, datamodule, "predict", ckpt_path)
        loaders = datamodule.test_dataloader()
        if not isinstance(loaders, list):
            loaders = [loaders]
        model.eval()
        results: list = []
        with torch.inference_mode():
            for loader in loaders:
                for batch_idx, batch in enumerate(loader):
                    out = model.predict_step(batch, batch_idx)
                    if out is not None:
                        results.append(out)
        return results

    # -- inner loops ---------------------------------------------------------

    def _train_one_epoch(
        self,
        model: nn.Module,
        datamodule: Any,
        opt: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler,
        amp_dtype: torch.dtype | None,
    ) -> None:
        model.train()
        self._dispatch("on_train_epoch_start", model)

        auto_opt = getattr(model, "automatic_optimization", True)
        log_every = self.config.log_every_n_steps
        for batch_idx, batch in enumerate(datamodule.train_dataloader()):
            self._dispatch("on_train_batch_start", model, batch, batch_idx)

            try:
                with torch.amp.autocast(
                    self._device.type,
                    enabled=amp_dtype is not None,
                    dtype=amp_dtype,
                ):
                    output = model.training_step(batch, batch_idx)
            except torch.cuda.OutOfMemoryError:
                output = self._handle_oom(batch_idx, batch)

            if auto_opt and opt is not None:
                self._optimizer_step(model, output, opt, scaler)

            # Flush logger only at log-throttle boundary — compute()
            # walks every accumulated key, not free at sub-second cadence.
            if self.global_step % log_every == 0:
                self.callback_metrics.update(model._metric_acc.compute())
                self._log_metrics(step=self.global_step)

            self._dispatch("on_train_batch_end", model, output, batch, batch_idx)
            self.global_step += 1

        # Epoch-level train metrics
        self.callback_metrics.update(model._metric_acc.compute())
        model._metric_acc.reset()

    def _validate_one_epoch(
        self,
        model: nn.Module,
        datamodule: Any,
        amp_dtype: torch.dtype | None,
    ) -> None:
        val_loader = datamodule.val_dataloader()
        if val_loader is None:
            return
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                with torch.amp.autocast(
                    self._device.type,
                    enabled=amp_dtype is not None,
                    dtype=amp_dtype,
                ):
                    model.validation_step(batch, batch_idx)

        # Flush epoch-level metrics (e.g. AUROC) into _metric_acc before compute.
        model.on_validation_epoch_end()

        self.callback_metrics.update(model._metric_acc.compute())
        model._metric_acc.reset()

        self._log_metrics(step=self.global_step)

    # -- step helpers --------------------------------------------------------

    def _optimizer_step(
        self,
        model: nn.Module,
        output: Any,
        opt: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
    ) -> None:
        loss = (
            output.get("loss")
            if isinstance(output, dict)
            else (output if isinstance(output, torch.Tensor) else None)
        )
        if loss is None:
            return
        # Order: scale→backward→unscale→clip→step→update (PyTorch docs).
        # GradScaler no-ops when enabled=False — no branching needed.
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if self.config.gradient_clip_val:
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
        scaler.step(opt)
        scaler.update()

    def _handle_oom(self, batch_idx: int, batch: Any) -> None:
        """Skip the batch — empty cache so the next batch isn't OOM
        for the same fragmentation reason. Loss not accumulated;
        optimizer not stepped. Cross-cutting runtime concern; lives
        here, not on the model.
        """
        torch.cuda.empty_cache()
        _log.warning(
            "oom_batch_skipped",
            batch_idx=batch_idx,
            num_graphs=getattr(batch, "num_graphs", None),
            num_nodes=getattr(batch, "num_nodes", None),
        )
        return None

    def _step_schedulers_safely(self) -> None:
        """Skip ``scheduler.step()`` when no optimizer stepped this run.

        ``GradScaler`` skips ``opt.step()`` on inf/nan grads (common on
        early fp16 batches while the scale warms up). Stepping the
        scheduler anyway trips PyTorch's "lr_scheduler.step() before
        optimizer.step()" warning and silently burns the first LR value.
        """
        if not any(getattr(o, "_opt_called", False) for o in self._optimizers):
            return
        for s in self._schedulers:
            if s is not None:
                s.step()

    # -- callbacks + logging -------------------------------------------------

    def _dispatch(self, hook: str, model: nn.Module, *args: Any) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                fn(self, model, *args)

    def _log_hyperparams(self, model: nn.Module) -> None:
        hp = vars(model.hparams) if hasattr(model.hparams, "__dict__") else {}
        for lg in self.loggers:
            lg.log_hyperparams(hp)

    def _log_metrics(self, step: int | None = None) -> None:
        for lg in self.loggers:
            lg.log_metrics(dict(self.callback_metrics), step=step)
