"""Chain execution — loop train_stage then eval_stage over a stage list.

The driver lives here (not in ``monarch.py``) so it stays testable
without the Monarch ``SlurmJob`` / actor spawn path. It takes an
already-spawned actor and a list of ``StageConfig``s; composition with
the allocation is the ``run_pipeline`` driver's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graphids._otel import get_logger

if TYPE_CHECKING:
    from graphids.orchestrate.planning import StageConfig

log = get_logger(__name__)


@dataclass(frozen=True)
class ChainResult:
    """Outcome of running one chain on a single actor."""

    checkpoints: dict[str, str] = field(default_factory=dict)  # {asset_name: ckpt_path}
    stage_to_asset: dict[str, str] = field(default_factory=dict)  # {stage: asset_name}

    def ckpts_by_stage(self) -> dict[str, str]:
        return {stage: self.checkpoints.get(asset, "") for stage, asset in self.stage_to_asset.items()}


def run_chain(
    actor,
    stages: list["StageConfig"],
    *,
    dataset: str,
    seed: int,
    max_retries: int = 2,
) -> ChainResult:
    """Loop ``train_stage`` over ``stages`` (with retry), then ``eval_stage``.

    Returns a ``ChainResult`` carrying the checkpoint path per asset +
    a stage → asset map so callers can look up by either key.
    """
    checkpoints: dict[str, str] = {}
    stage_to_asset = {cfg.stage: cfg.asset_name for cfg in stages}

    for cfg in stages:
        upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
        for attempt in range(max_retries + 1):
            try:
                ckpt = actor.train_stage.call_one(
                    stage_config=cfg.model_dump(),
                    dataset=dataset,
                    seed=seed,
                    upstream_ckpts=upstream,
                ).get()
                checkpoints[cfg.asset_name] = ckpt
                break
            except Exception as exc:
                log.error(
                    "stage_failed", stage=cfg.stage, attempt=attempt, error=str(exc),
                )
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"{cfg.stage} failed after {max_retries + 1} attempts",
                    ) from exc

    for cfg in stages:
        upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
        try:
            actor.eval_stage.call_one(
                stage_config=cfg.model_dump(),
                dataset=dataset,
                seed=seed,
                upstream_ckpts=upstream,
            ).get()
        except Exception as exc:
            log.warning("eval_failed", stage=cfg.stage, error=str(exc))

    return ChainResult(checkpoints=checkpoints, stage_to_asset=stage_to_asset)
