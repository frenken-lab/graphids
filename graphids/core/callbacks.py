"""graphids-specific Lightning callbacks.

Lightning's stock ``ModelCheckpoint`` / ``EarlyStopping`` cover the universal
trio (checkpoint + early-stop + MLflow forwarding); we only own callbacks that
encode graphids-specific policy:

- :class:`Sha256ModelCheckpoint` — Lightning ``ModelCheckpoint`` + sha256
  sidecar so :func:`graphids._fs.atomic_load` can verify integrity at load
  time on GPFS.
- :class:`TauNormCallback` — Kang et al. ICLR 2020 τ-norm of the GAT
  classifier head at fit-end.
- :class:`VRAMDriftCallback` — warn-once when free VRAM shrinks past a
  threshold across epoch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

from graphids._fs import _sha256_file
from graphids.core.models.base import strip_orig_mod_prefix


# ---------------------------------------------------------------------------
# Sha256ModelCheckpoint — Lightning ModelCheckpoint + sha256 sidecar
# ---------------------------------------------------------------------------


class Sha256ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """``pl.callbacks.ModelCheckpoint`` that writes a ``<ckpt>.sha256``
    sidecar after every save so :func:`graphids._fs.atomic_load` can verify
    bytes on read (GPFS truncates surprise us; sidecar is the load-time
    integrity check)."""

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:  # type: ignore[override]
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            p = Path(filepath)
            p.with_suffix(p.suffix + ".sha256").write_text(_sha256_file(p) + "\n")


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
class TauNormCallback(pl.Callback):
    """Apply Kang ICLR 2020 τ-norm to the GAT classifier head at fit-end.

    Tests whether the supervised classifier is imbalance-bottlenecked
    (Kang et al. 2020 §3.4). Loads the best checkpoint into the live
    model, rescales the final ``fc_layers[-1]`` ``nn.Linear`` weight by
    ``‖w_c‖^τ``, and re-saves the checkpoint. The hidden FC layers
    (``fc_layers[:-1]``) are encoder-side in Kang's framing and are
    left untouched — only the logit-producing weight matrix is normed.

    Ordering: registered under the key ``kang_tau_norm`` so it sorts
    alphabetically before ``mlflow`` in ``$.callbacks`` — Lightning
    invokes callbacks in registration order.
    """

    tau: float = 0.5

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Tag at start so the τ value is on the run regardless of fit-end ordering.
        # Go through ``trainer.logger`` (MLFlowLogger) rather than the fluent
        # ``mlflow.set_tag`` global — the logger is the single source of truth
        # for the active run and works without an implicit ``start_run`` context.
        logger = trainer.logger
        if logger is not None and hasattr(logger, "run_id"):
            logger.experiment.set_tag(
                logger.run_id, "graphids.tau_norm.tau", f"{self.tau:.4f}"
            )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from structlog import get_logger

        from graphids._fs import atomic_load, atomic_save

        log = get_logger(__name__)

        ckpt_cb = trainer.checkpoint_callback
        best_path_str = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
        if not best_path_str:
            log.warning("tau_norm.no_best_ckpt_skipped", tau=self.tau)
            return

        best_path = Path(best_path_str)
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

        GAT builds ``self.fc_layers`` directly on the LightningModule as
        ``ModuleList([Linear, ReLU, Dropout, ..., Linear])`` (the historical
        ``self.model = nn.Module(...)`` indirection was collapsed — see
        ``base.py:516``). The last entry is always the linear that maps to
        ``num_classes``. We pick the highest-indexed ``fc_layers.<N>.weight``
        whose value has 2 rows (``out_channels == num_classes == 2``);
        intermediate Linear rows match ``fc_input_dim``, so 2-row indexing
        is unambiguous.
        """
        prefix = "fc_layers."
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
                f"τ-norm: no top-level {prefix}<N>.weight entries found in state_dict — "
                "TauNormCallback only supports GAT-shaped models (fc_layers ModuleList)"
            )
        # Highest index = last layer = the classifier head.
        return max(candidates, key=lambda t: t[0])[1]


# ---------------------------------------------------------------------------
# VRAMDriftCallback
# ---------------------------------------------------------------------------


@dataclass
class VRAMDriftCallback(pl.Callback):
    """Warn when free VRAM shrinks past ``threshold`` between epochs.

    The node-budget probe captures ``free`` once at build time. Over a
    long run the actual free pool drifts — co-resident CUDA processes on
    shared nodes, activation checkpoint leaks in PyG. We capture a baseline
    at ``on_fit_start`` and compare at each epoch boundary. Epoch boundaries
    deliberately avoid transient allocations (teacher params are moved on/off
    GPU per-step in KD; checking between epochs catches persistent leaks
    only). Log-and-warn: re-probing mid-run would race optimizer state, so
    the researcher decides whether to abort.
    """

    threshold: float = 0.20

    baseline_free: int = field(init=False, default=0)
    _warned: bool = field(init=False, default=False)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not torch.cuda.is_available():
            return
        self.baseline_free = max(1, torch.cuda.mem_get_info()[0])

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
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
