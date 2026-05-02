"""Training callback and logger protocols — pure-Python replacements for Lightning.

Callback lifecycle mirrors Lightning's hook names so existing OTel and
curriculum callbacks need minimal changes.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from graphids.core._ckpt import build_checkpoint, strip_orig_mod_prefix

if TYPE_CHECKING:
    from graphids.core.trainer import Trainer


# ---------------------------------------------------------------------------
# Callback base class (default no-ops)
# ---------------------------------------------------------------------------


class CallbackBase:
    """Concrete base with no-op defaults so subclasses only override what they need."""

    def on_fit_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_fit_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_exception(
        self, trainer: Trainer, model: torch.nn.Module, exception: BaseException
    ) -> None:
        pass

    def on_train_epoch_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        pass

    def on_train_batch_start(
        self, trainer: Trainer, model: torch.nn.Module, batch: Any, batch_idx: int
    ) -> None:
        pass

    def on_train_batch_end(
        self,
        trainer: Trainer,
        model: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------

_OPS = {"min": operator.lt, "max": operator.gt}
_WORST = {"min": float("inf"), "max": float("-inf")}


def _resolve_mode_state(mode: str) -> tuple[Any, float]:
    """Return ``(compare_fn, worst_score)`` for ``mode ∈ {'min', 'max'}``.

    Raises ``ValueError`` for any other value. Shared ``__post_init__``
    helper for ModelCheckpoint and EarlyStopping so both callbacks stay
    in sync on what 'better' means and what the initial sentinel is.
    """
    if mode not in _OPS:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    return _OPS[mode], _WORST[mode]


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
        self._compare, self.best_score = _resolve_mode_state(self.mode)

    def _resolve_dirpath(self, trainer: Trainer) -> Path:
        return (
            Path(self.dirpath) if self.dirpath else Path(trainer.default_root_dir) / "checkpoints"
        )

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        from graphids._fs import atomic_save

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        dirpath = self._resolve_dirpath(trainer)
        dirpath.mkdir(parents=True, exist_ok=True)

        ckpt = build_checkpoint(trainer, model)

        if self.save_last:
            atomic_save(ckpt, dirpath / "last.ckpt")

        if self._compare(current, self.best_score):
            self.best_score = current
            best_path = dirpath / f"{self.filename}.ckpt"
            atomic_save(ckpt, best_path)
            self.best_model_path = str(best_path)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping(CallbackBase):
    """Stop training when monitored metric stops improving.

    Flips ``trainer.should_stop`` at the epoch boundary — doesn't raise.
    The fit loop observes the flag after the scheduler step so the
    current epoch's metrics are logged before exit.
    """

    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 100
    min_delta: float = 0.0

    wait_count: int = field(init=False, default=0)
    best_score: float = field(init=False)

    def __post_init__(self) -> None:
        self._compare, self.best_score = _resolve_mode_state(self.mode)

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        # min_delta: an improvement only counts if it exceeds the prior best
        # by at least this margin. Default 0.0 preserves strict-inequality
        # behavior for callers not specifying it.
        threshold = (
            self.best_score - self.min_delta
            if self.mode == "min"
            else self.best_score + self.min_delta
        )
        if self._compare(current, threshold):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True


# Checkpoint save/load helpers live in :mod:`graphids.core._ckpt`.


# ---------------------------------------------------------------------------
# TauNormCallback — Kang et al. ICLR 2020 classifier-head τ-norm
# ---------------------------------------------------------------------------


def apply_tau_norm(weight: torch.Tensor, tau: float) -> None:
    """In-place row-wise τ-norm of a classifier weight matrix.

    Kang et al., "Decoupling Representation and Classifier for Long-Tailed
    Recognition," ICLR 2020 §3.4 (arXiv 1910.09217). Rescales each row
    ``w_c`` by ``1 / ‖w_c‖^τ`` so majority-class rows (which grow large
    norms under imbalance) are damped relative to minority. ``tau=0``
    is identity; ``tau=1`` produces unit-norm rows. Kang sweeps τ ∈ [0, 1]
    on val.
    """
    if weight.ndim != 2:
        raise ValueError(f"τ-norm expects a 2-D weight matrix, got shape {tuple(weight.shape)}")
    norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    weight.div_(norms.pow(tau))


@dataclass
class TauNormCallback(CallbackBase):
    """Apply Kang ICLR 2020 τ-norm to the GAT classifier head at fit-end.

    Tests whether the supervised classifier is imbalance-bottlenecked
    (Kang et al. 2020 §3.4). Loads the best checkpoint into the live
    model, rescales the final ``fc_layers[-1]`` ``nn.Linear`` weight by
    ``‖w_c‖^τ``, and re-saves the checkpoint. The hidden FC layers
    (``fc_layers[:-1]``) are encoder-side in Kang's framing and are
    left untouched — only the logit-producing weight matrix is normed.

    Ordering: registered under the key ``kang_tau_norm`` so it sorts
    alphabetically before ``mlflow`` in ``$.callbacks`` — the defaults
    libsonnet builds ``trainer.callbacks`` from ``std.objectFields``,
    which preserves the dict-key ordering. Any future post-fit consumer
    that reads the saved checkpoint must run after this callback.
    """

    tau: float = 0.5

    def on_fit_start(self, trainer: Trainer, model: torch.nn.Module) -> None:
        # Tag at start so the τ value is on the run regardless of fit-end ordering.
        try:
            import mlflow

            mlflow.set_tag("graphids.tau_norm.tau", f"{self.tau:.4f}")
        except Exception:  # noqa: BLE001 — MLflow tag is metadata, must not block training
            pass

    def on_fit_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        from structlog import get_logger

        from graphids._fs import atomic_load, atomic_save

        log = get_logger(__name__)

        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None or not ckpt_cb.best_model_path:
            log.warning("tau_norm.no_best_ckpt_skipped", tau=self.tau)
            return

        best_path = Path(ckpt_cb.best_model_path)
        ckpt = atomic_load(best_path, map_location="cpu", weights_only=True)
        state: dict[str, torch.Tensor] = strip_orig_mod_prefix(ckpt["state_dict"])

        weight_key = self._resolve_classifier_key(state)
        weight = state[weight_key]
        original_norm = float(weight.norm().item())
        apply_tau_norm(weight, self.tau)
        new_norm = float(weight.norm().item())

        ckpt["state_dict"] = state
        atomic_save(ckpt, best_path)

        log.info(
            "tau_norm.applied",
            tau=self.tau,
            classifier_key=weight_key,
            ckpt_path=str(best_path),
            weight_norm_before=round(original_norm, 4),
            weight_norm_after=round(new_norm, 4),
        )

    @staticmethod
    def _resolve_classifier_key(state: dict[str, torch.Tensor]) -> str:
        """Return the state-dict key of GAT's final ``nn.Linear`` (logit-producing).

        GATWithJK builds ``self.fc_layers`` as ``ModuleList([Linear, ReLU,
        Dropout, ..., Linear])``. The last entry is always the linear
        that maps to ``num_classes``. We pick the highest-indexed
        ``model.fc_layers.<N>.weight`` whose value has 2 rows
        (``out_channels == num_classes == 2``); intermediate Linear rows
        match ``fc_input_dim``, so 2-row indexing is unambiguous.
        """
        prefix = "model.fc_layers."
        candidates: list[tuple[int, str]] = []
        for k, v in state.items():
            if not (k.startswith(prefix) and k.endswith(".weight")):
                continue
            if v.ndim != 2:
                continue
            try:
                idx = int(k[len(prefix) :].split(".", 1)[0])
            except ValueError:
                continue
            candidates.append((idx, k))
        if not candidates:
            raise KeyError(
                f"τ-norm: no {prefix}<N>.weight entries found in state_dict — "
                "TauNormCallback only supports GAT-shaped models (fc_layers ModuleList)"
            )
        # Highest index = last layer = the classifier head.
        return max(candidates, key=lambda t: t[0])[1]


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
            from structlog import get_logger

            get_logger(__name__).warning(
                "vram_drift_detected",
                baseline_free=self.baseline_free,
                current_free=current,
                drift_frac=round(drift, 3),
                threshold=self.threshold,
                epoch=trainer.current_epoch,
            )
            # Warn once per run — repeated warnings add noise without
            # extra signal. Abort is the researcher's call.
            self._warned = True

# MLflowTrainingCallback lives in :mod:`graphids._mlflow` (single source).
