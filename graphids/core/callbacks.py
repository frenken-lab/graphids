"""graphids-specific Lightning callbacks.

Lightning's stock ``ModelCheckpoint`` / ``EarlyStopping`` cover the
universal trio (checkpoint + early-stop + MLflow forwarding); we only
ship callbacks that encode graphids-specific policy:

- ``Sha256ModelCheckpoint``: ``ModelCheckpoint`` + sha256 sidecar so
  ``_fs.atomic_load`` can verify integrity at load time on GPFS.
- ``TauNormCallback``: Kang ICLR 2020 τ-norm of the GAT classifier head
  at fit-end (rescales final ``fc_layers[-1]`` row-wise by ``‖w_c‖^τ``).
- ``VRAMDriftCallback``: warn-once when free VRAM shrinks past threshold
  across epoch boundaries.

``MLflowTrainingCallback`` lives in ``graphids._mlflow`` (single source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from structlog import get_logger

from graphids._fs import _sha256_file, atomic_load, atomic_save
from graphids.core.models.base import strip_orig_mod_prefix

log = get_logger(__name__)


class Sha256ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """``ModelCheckpoint`` + ``<ckpt>.sha256`` sidecar after every save.

    GPFS truncation surprises happen on OSC; the sidecar is the load-time
    integrity check used by ``_fs.atomic_load``.
    """

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:  # type: ignore[override]
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            p = Path(filepath)
            p.with_suffix(p.suffix + ".sha256").write_text(_sha256_file(p) + "\n")


def apply_tau_norm(weight: torch.Tensor, tau: float) -> None:
    """In-place row-wise τ-norm: ``w_c /= ‖w_c‖^τ``.

    Kang et al. ICLR 2020 §3.4 (arXiv 1910.09217). τ=0 identity, τ=1
    unit-norm rows. Damps majority-class rows under imbalance.
    """
    if weight.ndim != 2:
        raise ValueError(f"τ-norm needs 2-D weight, got shape {tuple(weight.shape)}")
    norms = torch.linalg.vector_norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
    weight.div_(norms.pow(tau))


@dataclass
class TauNormCallback(pl.Callback):
    """Apply Kang τ-norm to GAT's classifier head at fit-end.

    Loads the best ckpt, rescales the highest-indexed ``fc_layers.<N>.weight``
    by ``‖w_c‖^τ``, atomic-saves. Hidden FC layers (``fc_layers[:-1]``) are
    encoder-side per Kang's framing — only the logit-producing matrix is normed.
    """

    tau: float = 0.5

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Tag at start so τ lands on the run regardless of fit-end ordering.
        # Through ``trainer.logger`` (MLFlowLogger), not the fluent API — the
        # logger is the SoT for the active run, no implicit start_run context.
        logger = trainer.logger
        if logger is not None and hasattr(logger, "run_id"):
            logger.experiment.set_tag(logger.run_id, "graphids.tau_norm.tau", f"{self.tau:.4f}")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ckpt_cb = trainer.checkpoint_callback
        best = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
        if not best:
            log.warning("tau_norm.no_best_ckpt_skipped", tau=self.tau)
            return

        ckpt = atomic_load(Path(best), map_location="cpu", weights_only=True)
        state = strip_orig_mod_prefix(ckpt["state_dict"])

        key = _classifier_key(state)
        weight = state[key]
        before = float(weight.norm())
        apply_tau_norm(weight, self.tau)
        ckpt["state_dict"] = state
        atomic_save(ckpt, Path(best))

        log.info(
            "tau_norm.applied",
            tau=self.tau,
            classifier_key=key,
            ckpt_path=best,
            weight_norm_before=round(before, 4),
            weight_norm_after=round(float(weight.norm()), 4),
        )


def _classifier_key(state: dict[str, torch.Tensor]) -> str:
    """Highest-indexed ``fc_layers.<N>.weight`` (the GAT logit head).

    GAT builds ``fc_layers`` as ``ModuleList([Linear, ReLU, Dropout, ..., Linear])``
    directly on the LightningModule. Last entry maps to ``num_classes``.
    """
    prefix = "fc_layers."
    keys = [
        k for k, v in state.items()
        if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2
    ]
    if not keys:
        raise KeyError(
            f"τ-norm: no {prefix}<N>.weight in state_dict — "
            "TauNormCallback supports GAT-shaped models only"
        )
    return max(keys, key=lambda k: int(k.split(".", 2)[1]))


@dataclass
class VRAMDriftCallback(pl.Callback):
    """Warn-once when free VRAM shrinks past ``threshold`` across epochs.

    Budget probe captures free VRAM at build time. Over long runs the pool
    drifts (co-resident processes, PyG activation leaks). Baseline at
    fit-start, check at each epoch start. Warn-only — re-probing mid-run
    would race optimizer state; the researcher decides whether to abort.
    """

    threshold: float = 0.20
    baseline_free: int = field(init=False, default=0)
    _warned: bool = field(init=False, default=False)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if torch.cuda.is_available():
            self.baseline_free = max(1, torch.cuda.mem_get_info()[0])

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not torch.cuda.is_available() or self.baseline_free <= 1 or self._warned:
            return
        current = torch.cuda.mem_get_info()[0]
        drift = (self.baseline_free - current) / self.baseline_free
        if drift > self.threshold:
            log.warning(
                "vram_drift_detected",
                baseline_free=self.baseline_free,
                current_free=current,
                drift_frac=round(drift, 3),
                threshold=self.threshold,
                epoch=trainer.current_epoch,
            )
            self._warned = True
