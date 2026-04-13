"""Training callback and logger protocols — pure-Python replacements for Lightning.

Callback lifecycle mirrors Lightning's hook names so existing OTel and
curriculum callbacks need minimal changes.
"""

from __future__ import annotations

import operator
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from graphids.core.trainer import Trainer


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingCallback(Protocol):
    """Hook protocol called by :class:`Trainer` at lifecycle boundaries."""

    def on_fit_start(self, trainer: Trainer, model: torch.nn.Module) -> None: ...
    def on_fit_end(self, trainer: Trainer, model: torch.nn.Module) -> None: ...
    def on_exception(self, trainer: Trainer, model: torch.nn.Module, exception: BaseException) -> None: ...
    def on_train_epoch_start(self, trainer: Trainer, model: torch.nn.Module) -> None: ...
    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None: ...
    def on_train_batch_start(self, trainer: Trainer, model: torch.nn.Module, batch: Any, batch_idx: int) -> None: ...
    def on_train_batch_end(self, trainer: Trainer, model: torch.nn.Module, outputs: Any, batch: Any, batch_idx: int) -> None: ...


@runtime_checkable
class TrainingLogger(Protocol):
    """Logger protocol — receives metrics from ``model.log()`` calls."""

    def log_metrics(self, metrics_dict: dict[str, float], step: int | None = None) -> None: ...
    def log_hyperparams(self, params: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Callback base class (default no-ops)
# ---------------------------------------------------------------------------


class CallbackBase:
    """Concrete base with no-op defaults so subclasses only override what they need."""

    def on_fit_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_fit_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_exception(self, trainer: Trainer, model: torch.nn.Module, exception: BaseException) -> None:
        pass

    def on_train_epoch_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_train_batch_start(self, trainer: Trainer, model: torch.nn.Module, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(
        self, trainer: Trainer, model: torch.nn.Module, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------

_OPS = {"min": operator.lt, "max": operator.gt}
_WORST = {"min": float("inf"), "max": float("-inf")}


@dataclass
class ModelCheckpoint(CallbackBase):
    """Save best + last checkpoints based on a monitored metric.

    Writes to ``{trainer.default_root_dir}/checkpoints/`` unless an
    explicit ``dirpath`` is set. The ``/checkpoints`` subdir convention
    is owned here so neither jsonnet nor the instantiator has to wire
    it from the trainer's run_dir.
    """

    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: str = "best_model"
    dirpath: str = ""

    best_model_path: str = ""
    best_score: float = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in _OPS:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        self.best_score = _WORST[self.mode]
        self._compare = _OPS[self.mode]

    def _resolve_dirpath(self, trainer: Trainer) -> Path:
        return Path(self.dirpath) if self.dirpath else Path(trainer.default_root_dir) / "checkpoints"

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        dirpath = self._resolve_dirpath(trainer)
        dirpath.mkdir(parents=True, exist_ok=True)

        ckpt = _build_checkpoint(trainer, model)

        if self.save_last:
            last_path = dirpath / "last.ckpt"
            _atomic_save(ckpt, last_path)

        if self._compare(current, self.best_score):
            self.best_score = current
            best_path = dirpath / f"{self.filename}.ckpt"
            _atomic_save(ckpt, best_path)
            self.best_model_path = str(best_path)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping(CallbackBase):
    """Stop training when monitored metric stops improving."""

    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 100

    wait_count: int = field(init=False, default=0)
    best_score: float = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in _OPS:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        self.best_score = _WORST[self.mode]
        self._compare = _OPS[self.mode]

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        if self._compare(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _build_checkpoint(trainer: Trainer, model: torch.nn.Module) -> dict[str, Any]:
    """Build a raw-PyTorch checkpoint dict."""
    cls = type(model)
    ckpt: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "class_path": f"{cls.__module__}.{cls.__name__}",
    }
    if hasattr(model, "hparams"):
        hp = model.hparams
        ckpt["hyper_parameters"] = vars(hp) if hasattr(hp, "__dict__") else dict(hp)
    if trainer.callback_metrics:
        ckpt["metrics"] = {k: float(v) for k, v in trainer.callback_metrics.items()}
    # Let model add custom state (e.g. test_threshold)
    if hasattr(model, "on_save_checkpoint"):
        model.on_save_checkpoint(ckpt)
    # Optimizer + scheduler + scaler state for resume
    if trainer._optimizers:
        ckpt["optimizer_states"] = [opt.state_dict() for opt in trainer._optimizers]
    if trainer._schedulers:
        ckpt["lr_schedulers"] = [s.state_dict() for s in trainer._schedulers if s is not None]
    return ckpt


def _atomic_save(obj: Any, path: Path) -> None:
    """Write checkpoint atomically via temp + fsync + rename (NFS safe)."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, str(tmp))
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# VRAMDriftCallback
# ---------------------------------------------------------------------------


@dataclass
class VRAMDriftCallback(CallbackBase):
    """Warn when free VRAM shrinks past ``threshold`` between epochs.

    The node-budget probe captures ``free`` once at build time. Over a
    long run the actual free pool drifts — co-resident CUDA processes on
    shared nodes, activation checkpoint leaks in PyG, growing OTel
    exporter caches. We capture a baseline at ``on_fit_start`` and
    compare at each epoch boundary. Epoch boundaries deliberately avoid
    transient allocations (teacher params are moved on/off GPU per-step
    in KD; checking between epochs catches persistent leaks only).
    Log-and-warn: re-probing mid-run would race optimizer state, so the
    researcher decides whether to abort.
    """

    threshold: float = 0.20

    baseline_free: int = field(init=False, default=0)
    _warned: bool = field(init=False, default=False)

    def on_fit_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        if not torch.cuda.is_available():
            return
        self.baseline_free = max(1, torch.cuda.mem_get_info()[0])

    def on_train_epoch_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        if not torch.cuda.is_available() or self.baseline_free <= 1 or self._warned:
            return
        current = torch.cuda.mem_get_info()[0]
        drift = (self.baseline_free - current) / self.baseline_free
        if drift > self.threshold:
            from graphids._otel import get_logger
            get_logger(__name__).warning(
                "vram_drift_detected",
                baseline_free=self.baseline_free, current_free=current,
                drift_frac=round(drift, 3),
                threshold=self.threshold, epoch=trainer.current_epoch,
            )
            # Warn once per run — repeated warnings add noise without
            # extra signal. Abort is the researcher's call.
            self._warned = True
