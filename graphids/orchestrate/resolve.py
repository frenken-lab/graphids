"""Config resolution and cross-field validation for pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphids.config.jsonnet import render
from graphids.config.schemas import validate_config
from graphids.config.topology import TOPOLOGY, PathContext
from graphids._otel import get_logger
from graphids.orchestrate.planning import StageConfig

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TLA dict construction
# ---------------------------------------------------------------------------


def _build_tla_dict(
    stage_cfg: StageConfig,
    *,
    dataset: str,
    seed: int,
    run_dir: str,
    upstream_ckpts: dict[str, str],
    upstream_model_families: dict[str, str],
    kd_overrides: dict[str, Any] | None = None,
    trainer_overrides: dict[str, str] | None = None,
    stage_overrides: dict[str, str] | None = None,
    ckpt_path: str | None = None,
) -> dict[str, Any]:
    """Build the typed TLA dict consumed by the stage's jsonnet function."""
    tla: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "run_dir": run_dir,
        "scale": stage_cfg.scale,
        "trainer_overrides": dict(trainer_overrides or {}),
        "stage_overrides": dict(stage_overrides or {}),
    }
    tla.update(stage_cfg.model_init_overrides)

    stage_def = TOPOLOGY.stages.get(stage_cfg.stage)
    accepted = set(stage_def.stage_tlas) if stage_def else set()

    if "fusion_method" in accepted:
        tla["fusion_method"] = stage_cfg.resource_model or stage_cfg.model_type

    for upstream_asset, ckpt in upstream_ckpts.items():
        family = upstream_model_families.get(upstream_asset)
        if family == "unsupervised" and "vgae_ckpt_path" in accepted:
            tla["vgae_ckpt_path"] = ckpt
        elif family == "supervised" and "gat_ckpt_path" in accepted:
            tla["gat_ckpt_path"] = ckpt

    if "distillation_config" in accepted:
        tla["distillation_config"] = dict(kd_overrides) if kd_overrides else None

    if ckpt_path is not None and "ckpt_path" in accepted:
        tla["ckpt_path"] = ckpt_path

    return tla


@dataclass(frozen=True)
class ResolvedConfig:
    """Rendered, validated config ready for instantiation."""

    paths: PathContext
    validated: Any  # ValidatedConfig
    rendered: dict[str, Any]

    @classmethod
    def resolve(
        cls,
        cfg: StageConfig,
        *,
        lake_root: str,
        user: str,
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> ResolvedConfig:
        """Resolve a StageConfig into a validated, rendered config."""
        upstream_ckpts = upstream_ckpts or {}

        paths = PathContext(
            lake_root=lake_root,
            user=user,
            dataset=dataset,
            model_type=cfg.model_type,
            scale=cfg.scale,
            stage=cfg.stage,
            identity=cfg.identity,
            kd_tag=cfg.kd_tag,
            seed=seed,
        )

        tla = _build_tla_dict(
            cfg,
            dataset=dataset,
            seed=seed,
            run_dir=str(paths.run_dir),
            upstream_ckpts=upstream_ckpts,
            upstream_model_families=cfg.upstream_model_families,
            kd_overrides=cfg.kd_overrides or None,
            trainer_overrides=cfg.trainer_overrides or None,
            stage_overrides=cfg.stage_overrides or None,
        )

        rendered = render(cfg.jsonnet_path, tla)

        try:
            validated = validate_config(rendered)
        except ValueError as e:
            raise ValueError(f"{cfg.asset_name} config validation: {e}") from e

        family = TOPOLOGY.stage_family_map.get(cfg.stage)
        if family is not None:
            exp_monitor, exp_mode = (
                ("val_acc", "max") if family == "fusion" else ("val_loss", "min")
            )
            if validated.checkpoint_monitor != exp_monitor or validated.checkpoint_mode != exp_mode:
                log.warning(
                    "stage_monitor_mismatch",
                    asset=cfg.asset_name,
                    got=f"{validated.checkpoint_monitor}/{validated.checkpoint_mode}",
                    expected=f"{exp_monitor}/{exp_mode}",
                )

        return cls(paths=paths, validated=validated, rendered=rendered)
