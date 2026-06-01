"""Budget planner orchestration."""

from __future__ import annotations

import torch
from structlog import get_logger

from .config import BudgetConfig
from .heuristic import _heuristic_budget
from .probe import probe
from .types import BudgetResult

log = get_logger(__name__)


def _can_probe(model, train_dataset) -> bool:
    return torch.cuda.is_available() and model is not None and train_dataset is not None


def node_budget(
    dataset: str,
    *,
    model=None,
    train_dataset=None,
    conv_type: str | None = None,
    heads: int | None = None,
    min_steps: int | None = None,
    probe_fn=probe,
    config: BudgetConfig | None = None,
) -> BudgetResult:
    cfg = config or BudgetConfig.from_env()
    if conv_type is None and model is not None:
        conv_type = getattr(model.hparams, "conv_type", "gatv2")
    quadratic = conv_type == "gps"

    if cfg.mode in {"probe", "measured", "auto"} and _can_probe(model, train_dataset):
        try:
            return probe_fn(model, train_dataset, quadratic=quadratic, min_steps=min_steps)
        except Exception:
            if cfg.strict_probe or cfg.mode in {"probe", "measured"}:
                raise
            log.warning("budget_probe_failed_using_heuristic", dataset=dataset, exc_info=True)
            return _heuristic_budget(
                dataset,
                train_dataset=train_dataset,
                quadratic=quadratic,
                heads=heads,
                min_steps=min_steps,
                binding="probe_failed_heuristic",
                config=cfg,
            )
    if cfg.mode in {"probe", "measured"} and cfg.strict_probe:
        raise RuntimeError("budget probe requires CUDA + model + train_dataset")
    return _heuristic_budget(
        dataset,
        train_dataset=train_dataset,
        quadratic=quadratic,
        heads=heads,
        min_steps=min_steps,
        config=cfg,
    )
