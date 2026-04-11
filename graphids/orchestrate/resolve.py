"""Config resolution — Layer 2 of the orchestrate stack.

``resolve_config`` is the single entry point for pipeline-path
resolution: it builds a ``PathContext`` from the ``StageConfig`` +
runtime context, packs the TLA dict via ``StageConfig.to_tla_dict``,
shells out to the jsonnet binary, and gates the output through
``validate_config``. The CLI path constructs ``ResolvedConfig``
directly via ``ResolvedConfig.from_rendered`` and doesn't touch this
module.
"""

from __future__ import annotations

from pathlib import Path

from graphids.config.jsonnet import render
from graphids.config.schemas import validate_config
from graphids.config.topology import PathContext
from graphids.orchestrate.config import ResolvedConfig, StageConfig


def resolve_config(
    cfg: StageConfig,
    *,
    lake_root: str,
    user: str,
    dataset: str,
    seed: int,
    upstream_ckpts: dict[str, str] | None = None,
) -> ResolvedConfig:
    """Render + validate a StageConfig into a ResolvedConfig."""
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
    tla = cfg.to_tla_dict(
        dataset=dataset,
        seed=seed,
        run_dir=str(paths.run_dir),
        upstream_ckpts=upstream_ckpts or {},
    )
    rendered = render(cfg.jsonnet_path, tla)
    try:
        validated = validate_config(rendered)
    except ValueError as e:
        raise ValueError(f"{cfg.asset_name} config validation: {e}") from e
    return ResolvedConfig(
        rendered=rendered,
        validated=validated,
        stage_name=cfg.stage,
        run_dir=Path(str(paths.run_dir)),
        ckpt_file=Path(str(paths.ckpt_file)),
    )
