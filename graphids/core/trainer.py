"""Pure-PyTorch training loop for GraphIDS.

Single-GPU only (project uses 1x V100). Handles:
- AMP via ``torch.amp.autocast`` + ``GradScaler(enabled=...)`` (no-op when disabled)
- Gradient clipping via ``clip_grad_norm_``
- ``automatic_optimization=False`` for RL fusion models
- Metric accumulation and logger dispatch
- Callback lifecycle (same hook names as Lightning)
- Checkpoint resume
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from graphids._otel import get_logger
from graphids.core.callbacks import CallbackBase, EarlyStopping, ModelCheckpoint

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric accumulator — batch-size-weighted mean per epoch phase
# ---------------------------------------------------------------------------


class MetricAccumulator:
    """Dynamic-keyed batch-weighted mean.

    Plain ``dict[str, (sum, count)]`` — NOT an ``nn.Module``. These are
    transient per-phase accumulators; storing them in a ``ModuleDict``
    both pollutes the parent's ``state_dict`` and rejects keys with
    ``"."`` (add_module's attribute-name check), breaking metric names
    like ``"test/precision@0.95recall"``.

    NaN detection hard-fails the run — under ``precision: 16-mixed`` a
    silent NaN in ``callback_metrics`` fools ``EarlyStopping``
    (``NaN < inf`` is False) and wastes the full patience window.
    """

    def __init__(self, nan_strategy: str = "error") -> None:
        self._nan_strategy = nan_strategy
        self._sums: dict[str, float] = {}
        self._counts: dict[str, float] = {}

    def update(self, name: str, value: float, batch_size: int = 1) -> None:
        v = float(value)
        if math.isnan(v):
            if self._nan_strategy == "error":
                raise ValueError(f"NaN encountered in metric {name!r}")
            return
        self._sums[name] = self._sums.get(name, 0.0) + v * batch_size
        self._counts[name] = self._counts.get(name, 0.0) + batch_size

    def compute(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts.get(k)}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs. ``torch.manual_seed`` covers CPU + CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Flat config matching the jsonnet ``trainer`` section keys."""

    max_epochs: int = 300
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50
    accelerator: str = "auto"
    devices: str | int = "auto"
    default_root_dir: str = ""


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
        # build_loggers returns a list or a falsy (None/False); normalize once.
        self.loggers: list = list(logger) if isinstance(logger, list) else []

        # Public state (read by callbacks + model code)
        self.max_epochs: int = config.max_epochs
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.callback_metrics: dict[str, float] = {}
        self.default_root_dir: str = config.default_root_dir
        self.should_stop: bool = False
        self.datamodule: Any = None

        # Populated during fit()
        self._optimizers: list[torch.optim.Optimizer] = []
        self._schedulers: list[Any] = []

        # Resolve device
        self._device = self._resolve_device(config.accelerator)

        # Find well-known callbacks
        self.checkpoint_callback: ModelCheckpoint | None = None
        self.early_stopping_callback: EarlyStopping | None = None
        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpoint):
                self.checkpoint_callback = cb
            elif isinstance(cb, EarlyStopping):
                self.early_stopping_callback = cb

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return self._optimizers

    # -- device resolution ---------------------------------------------------

    @staticmethod
    def _resolve_device(accelerator: str) -> torch.device:
        if accelerator == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _wire_datamodule(self, datamodule: Any, model: nn.Module) -> None:
        """Push device + model handles into the datamodule.

        Every datamodule MUST implement ``_set_device`` and ``_set_model``.
        DMs that don't move batches (e.g. FusionDataModule) implement them
        as no-ops — but the method must exist. Silent getattr fallbacks
        previously hid DM-wiring bugs as cpu/cuda mismatches at the first op.
        """
        datamodule._set_device(self._device)
        datamodule._set_model(model)

    # -- public API ----------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        datamodule: Any,
        ckpt_path: str | None = None,
    ) -> None:
        self.datamodule = datamodule
        self._wire_datamodule(datamodule, model)

        datamodule.setup("fit")
        model.setup(datamodule)
        # Must move AFTER setup(): _build() creates self.model and may wrap it
        # in torch.compile — moving before setup leaves the inner module on CPU.
        model.to(self._device)

        opt, sched = model.build_optimizers(self.max_epochs)
        self._optimizers = [opt] if opt else []
        self._schedulers = [sched] if sched else []

        # GradScaler(enabled=False) is a complete no-op passthrough —
        # all methods become identity. No branching needed in the loop.
        use_amp = "16" in str(self.config.precision) and self._device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        if ckpt_path:
            self._resume_fit(ckpt_path, model, opt, sched, scaler)

        self._dispatch("on_fit_start", model)
        self._log_hyperparams(model)

        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                self._train_one_epoch(model, datamodule, opt, scaler, use_amp)
                self._validate_one_epoch(model, datamodule, use_amp)

                self._dispatch("on_train_epoch_end", model)

                # Skip scheduler.step() when the optimizer hasn't stepped
                # this run — GradScaler skips opt.step() on inf/nan grads,
                # which is common on early fp16 batches while the scale warms
                # up. Stepping the scheduler anyway trips PyTorch's
                # "lr_scheduler.step() before optimizer.step()" warning and
                # silently burns the first LR value.
                opt_stepped = any(getattr(o, "_opt_called", False) for o in self._optimizers)
                if opt_stepped:
                    for s in self._schedulers:
                        if s is not None:
                            s.step()

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
        self.datamodule = datamodule
        self._wire_datamodule(datamodule, model)

        datamodule.setup("test")
        model.setup(datamodule)
        model.to(self._device)

        if ckpt_path:
            self._load_model_weights(ckpt_path, model)

        # test_dataloader() may trigger BudgetProfiler.probe() which needs
        # autograd for a backward pass — must build it BEFORE torch.no_grad.
        test_loaders = datamodule.test_dataloader()
        if not isinstance(test_loaders, list):
            test_loaders = [test_loaders]

        model.eval()
        model.on_test_epoch_start()

        with torch.no_grad():
            for dl_idx, loader in enumerate(test_loaders):
                for batch_idx, batch in enumerate(loader):
                    model.test_step(batch, batch_idx, dataloader_idx=dl_idx)

        model.on_test_epoch_end()

        self.callback_metrics.update(model._metric_acc.compute())
        model._metric_acc.reset()

        return dict(self.callback_metrics)

    def validate(
        self,
        model: nn.Module,
        datamodule: Any,
        ckpt_path: str | None = None,
    ) -> dict[str, float]:
        self.datamodule = datamodule
        self._wire_datamodule(datamodule, model)

        datamodule.setup("fit")
        model.setup(datamodule)
        model.to(self._device)

        if ckpt_path:
            self._load_model_weights(ckpt_path, model)

        use_amp = "16" in str(self.config.precision) and self._device.type == "cuda"
        self._validate_one_epoch(model, datamodule, use_amp)
        return dict(self.callback_metrics)

    def predict(
        self,
        model: nn.Module,
        datamodule: Any,
        ckpt_path: str | None = None,
    ) -> list:
        self.datamodule = datamodule
        self._wire_datamodule(datamodule, model)

        datamodule.setup("predict")
        model.setup(datamodule)
        model.to(self._device)

        if ckpt_path:
            self._load_model_weights(ckpt_path, model)

        loaders = datamodule.test_dataloader()
        if not isinstance(loaders, list):
            loaders = [loaders]
        results: list = []
        for loader in loaders:
            results.extend(self.predict_on(model, loader))
        return results

    def predict_on(self, model: nn.Module, loader: Any) -> list:
        """Run ``predict_step`` over a single loader. Assumes model/dm set up."""
        model.eval()
        results: list = []
        with torch.no_grad():
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
        use_amp: bool,
    ) -> None:
        model.train()
        self._dispatch("on_train_epoch_start", model)

        train_loader = datamodule.train_dataloader()
        for batch_idx, batch in enumerate(train_loader):
            self._dispatch("on_train_batch_start", model, batch, batch_idx)

            auto_opt = getattr(model, "automatic_optimization", True)
            if auto_opt and opt is not None:
                opt.zero_grad()
                with torch.amp.autocast(self._device.type, enabled=use_amp):
                    output = model.training_step(batch, batch_idx)
                loss = output if isinstance(output, torch.Tensor) else None
                if isinstance(output, dict):
                    loss = output.get("loss")

                if loss is not None:
                    # GradScaler no-ops when enabled=False — no branching needed.
                    # Order: scale→backward→unscale→clip→step→update (PyTorch docs).
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    if self.config.gradient_clip_val:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config.gradient_clip_val,
                        )
                    scaler.step(opt)
                    scaler.update()
            else:
                # Model manages its own backward + step (RL fusion)
                with torch.amp.autocast(self._device.type, enabled=use_amp):
                    output = model.training_step(batch, batch_idx)

            # Flush step-level metrics from model.log()
            self.callback_metrics.update(model._metric_acc.compute())

            if self.global_step % self.config.log_every_n_steps == 0:
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
        use_amp: bool,
    ) -> None:
        # val_dataloader() may trigger BudgetProfiler.probe() which needs
        # autograd for a backward pass — must build it BEFORE torch.no_grad.
        val_loader = datamodule.val_dataloader()
        if val_loader is None:
            return
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                with torch.amp.autocast(self._device.type, enabled=use_amp):
                    model.validation_step(batch, batch_idx)

        self.callback_metrics.update(model._metric_acc.compute())
        model._metric_acc.reset()

        self._log_metrics(step=self.global_step)

    # -- callbacks -----------------------------------------------------------

    def _dispatch(self, hook: str, model: nn.Module, *args: Any) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                fn(self, model, *args)

    # -- logging -------------------------------------------------------------

    def _log_hyperparams(self, model: nn.Module) -> None:
        hp = vars(model.hparams) if hasattr(model.hparams, "__dict__") else {}
        for lg in self.loggers:
            lg.log_hyperparams(hp)

    def _log_metrics(self, step: int | None = None) -> None:
        for lg in self.loggers:
            lg.log_metrics(dict(self.callback_metrics), step=step)

    # -- checkpoint resume ---------------------------------------------------

    def _resume_fit(
        self,
        ckpt_path: str,
        model: nn.Module,
        opt: torch.optim.Optimizer | None,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
    ) -> None:
        """Resume training from a checkpoint."""
        ckpt = self._load_ckpt_into(ckpt_path, model)
        self.current_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)

        if opt and "optimizer_states" in ckpt:
            opt.load_state_dict(ckpt["optimizer_states"][0])
        if scheduler and "lr_schedulers" in ckpt:
            scheduler.load_state_dict(ckpt["lr_schedulers"][0])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        _log.info(
            "resumed_from_checkpoint",
            path=ckpt_path,
            epoch=self.current_epoch,
            global_step=self.global_step,
        )

    def _load_model_weights(self, ckpt_path: str, model: nn.Module) -> None:
        """Load model weights only (for test/validate/predict)."""
        self._load_ckpt_into(ckpt_path, model)
        _log.info("loaded_checkpoint", path=ckpt_path)

    def _load_ckpt_into(self, ckpt_path: str, model: nn.Module) -> dict:
        """Load ckpt, restore weights, fire ``on_load_checkpoint``. Return raw dict."""
        from graphids.core.callbacks import _strip_orig_mod_prefix

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Align ckpt to target's compile-prefix convention. Save strips
        # ``_orig_mod.``; target may or may not have it depending on whether
        # this run has compile_model enabled. Remap via the target's keys.
        stripped = _strip_orig_mod_prefix(state)
        remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict().keys()}
        state = {remap.get(k, k): v for k, v in stripped.items()}
        model.load_state_dict(state)
        model.on_load_checkpoint(ckpt)
        return ckpt
