"""MLflow training callback — per-epoch metrics + final VRAM/ckpt stamping.

Replaces ``OTelTrainingCallback``'s ML-specific hooks. Lifecycle:

- ``on_train_epoch_end``: append every metric the model logged via
  ``self.log(...)`` (read from ``trainer.callback_metrics``) plus ``lr`` /
  ``early_stop.wait`` at ``step=epoch``. The model layer gates what gets
  named; this callback does not whitelist.
- ``on_fit_end``: stamp ``peak_vram_mb`` + ``epochs_run`` + best-ckpt hash
  + best-ckpt path tag, then close the run with status ``FINISHED``.
- ``on_exception``: close the run with status ``FAILED``.

Device telemetry (GPU utilization, VRAM allocated/reserved, CPU, memory,
disk, network) is captured by MLflow's background system-metrics sampler
— enabled once per process in ``_mlflow.start_training_run``. No
per-batch hook here; sampling handles it at a fixed interval.
"""

from __future__ import annotations

from pathlib import Path

import torch

from graphids.core.callbacks import CallbackBase


class MLflowTrainingCallback(CallbackBase):
    """Per-epoch + fit-end MLflow logging. Lifecycle opened/closed by this callback.

    The MLflow run is started in ``stage.train`` via
    ``_mlflow.start_training_run`` before ``trainer.fit`` runs. This
    callback only writes into the already-active run, then finalizes and
    closes it. Keeps the trainer + CLI layer free of MLflow knowledge.
    """

    def on_train_epoch_end(self, trainer, model: torch.nn.Module) -> None:
        import mlflow

        from graphids._mlflow import log_epoch_metrics

        # Stamp autosize tags on epoch 0 (the dataloader has now been built,
        # so _autosize_info is populated). Later epochs idempotent-overwrite,
        # which is harmless. Doing this here — not in on_fit_end — means the
        # tags survive walltime kills / preemption, which is the common case
        # for smoke verifications.
        if trainer.current_epoch == 0:
            autosize = getattr(getattr(trainer, "datamodule", None), "_autosize_info", None)
            if autosize is not None:
                mlflow.set_tag("graphids.num_workers", str(autosize["num_workers"]))
                mlflow.set_tag("graphids.num_workers_source", autosize["source"])
                mlflow.set_tag("graphids.prefetch_factor", str(autosize["prefetch_factor"]))

        cb = trainer.callback_metrics
        metrics: dict[str, float] = {k: float(v) for k, v in cb.items() if v is not None}
        if trainer.optimizers:
            metrics["lr"] = float(trainer.optimizers[0].param_groups[0]["lr"])
        es = getattr(trainer, "early_stopping_callback", None)
        if es is not None:
            metrics["early_stop.wait"] = float(es.wait_count)
            if es.best_score is not None:
                metrics["early_stop.best_score"] = float(es.best_score)
        log_epoch_metrics(trainer.current_epoch, metrics)

    def on_fit_end(self, trainer, model: torch.nn.Module) -> None:
        from graphids._mlflow import end_training_run, log_final_fit

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            try:
                dev = getattr(model, "device", None)
                idx = dev.index if dev is not None and dev.index is not None else 0
                peak_vram_mb = torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
            except (AttributeError, RuntimeError):
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        best_path = ""
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb is not None and getattr(ckpt_cb, "best_model_path", ""):
            best_path = str(ckpt_cb.best_model_path)
        run_dir = Path(trainer.default_root_dir) if trainer.default_root_dir else Path()
        log_final_fit(
            peak_vram_mb=peak_vram_mb,
            epochs_run=trainer.current_epoch + 1,
            best_ckpt_path=best_path,
            run_dir=run_dir,
        )
        self._check_budget_utilization(trainer, peak_vram_mb)
        end_training_run(status="FINISHED")

    def _check_budget_utilization(self, trainer, peak_vram_mb: float) -> None:
        """Warn + tag when actual VRAM peak is far below the probed budget's
        target. Catches silently-conservative budgets that would otherwise
        masquerade as a healthy run at 20–30% GPU memory utilization.
        """
        import mlflow
        from structlog import get_logger

        budget = getattr(getattr(trainer, "datamodule", None), "_budget", None)
        if budget is None or budget.target_bytes <= 0 or peak_vram_mb <= 0:
            return
        peak_bytes = int(peak_vram_mb * 1024 * 1024)
        utilization = peak_bytes / budget.target_bytes
        pct = round(utilization * 100, 1)
        mlflow.set_tag("graphids.budget_utilization_pct", str(pct))
        # Tag value domain is the BudgetBinding Literal from budget.py —
        # {"measured", "measured_degenerate_fallback", "opted_in_fallback"}.
        # Pre-Apr-2026 runs stamped {"memory", "memory_fallback", "fallback"};
        # consumers filtering this tag across history must accept both domains.
        mlflow.set_tag("graphids.budget_binding", budget.binding)
        if utilization < 0.4:
            mlflow.set_tag("graphids.budget_underutilized", "true")
            get_logger(__name__).warning(
                "budget_underutilized",
                peak_vram_mb=round(peak_vram_mb, 1),
                target_mb=budget.target_bytes // (1024 * 1024),
                utilization_pct=pct,
                binding=budget.binding,
                threshold_pct=40.0,
            )

    def on_exception(
        self,
        trainer,
        model: torch.nn.Module,
        exception: BaseException,
    ) -> None:
        from graphids._mlflow import end_training_run

        del trainer, model, exception
        end_training_run(status="FAILED")
