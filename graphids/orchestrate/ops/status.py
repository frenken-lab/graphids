"""Pipeline status: DuckDB catalog joined with planner topology.

Public entry point: :func:`show_pipeline_status`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from graphids.config.constants import CONFIG_DIR, LAKE_ROOT
from graphids.config.topology import (
    TOPOLOGY,  # noqa: F401 (used in module body)
    catalog_path,
)

RECIPES_DIR = CONFIG_DIR / "recipes"

_FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"})
_RUNNING_STATES = frozenset({"RUNNING", "PENDING"})
_STAGE_ORDER = {s: i for i, s in enumerate(TOPOLOGY.default_stages)}

_CATALOG_STATUS_MAP = {
    "completed": "COMPLETED",
    "failed": "FAILED",
    "started": "RUNNING",
}


@dataclass
class AssetStatus:
    asset: str
    stage: str
    model_type: str
    scale: str
    status: str
    phases: dict[str, bool | None] = field(default_factory=dict)
    wall_time_seconds: float | None = None
    job_id: str = ""
    upstream: list[str] = field(default_factory=list)


def _load_topology(recipe_path: str | Path | None = None) -> list:
    """Render recipe → expand → enumerate_assets → list[StageConfig]."""
    from graphids.config.jsonnet import render
    from graphids.orchestrate.planning import enumerate_assets, expand_recipe_configs

    if recipe_path is None:
        from graphids.config.settings import get_settings

        recipe_env = get_settings().recipe
        recipe_path = Path(recipe_env) if recipe_env else RECIPES_DIR / "ablation.jsonnet"

    raw = render(Path(recipe_path))
    expanded = expand_recipe_configs(raw)
    return enumerate_assets(expanded)


def _collect_statuses(
    *,
    recipe_path: str | Path | None = None,
    dataset: str | None = None,
    seed: int = 42,
) -> list[AssetStatus]:
    """Load topology from planner, join with DuckDB catalog.

    For assets without a catalog row, infers BLOCKED/PENDING/WAITING
    from upstream status.
    """
    import duckdb

    stage_configs = _load_topology(recipe_path)
    config_by_name = {cfg.asset_name: cfg for cfg in stage_configs}

    cat = catalog_path(LAKE_ROOT)
    if not cat.exists():
        raise FileNotFoundError(
            f"Catalog not found at {cat}. Run: python -m graphids rebuild-catalog"
        )

    query = """
        SELECT
            asset_name,
            status,
            to_json(phases) AS phases_json,
            wall_time_seconds,
            CAST(slurm_job_id AS VARCHAR) AS slurm_job_id_str
        FROM runs
        WHERE seed = ?
    """
    params: list = [seed]
    if dataset:
        query += " AND dataset = ?"
        params.append(dataset)

    db = duckdb.connect(str(cat), read_only=True)
    try:
        rows = db.execute(query, params).fetchall()
    finally:
        db.close()

    catalog_by_name: dict[str, dict] = {}
    for asset_name, rec_status, phases_json, wall_s, job_id_str in rows:
        if not asset_name:
            continue
        try:
            phases = json.loads(phases_json) if phases_json else {}
        except (TypeError, ValueError):
            phases = {}
        catalog_by_name[asset_name] = {
            "status": _CATALOG_STATUS_MAP.get(rec_status, (rec_status or "").upper()),
            "phases": phases if isinstance(phases, dict) else {},
            "wall_time_seconds": wall_s,
            "job_id": job_id_str or "",
        }

    status_map: dict[str, str] = {}
    out: list[AssetStatus] = []

    all_names = sorted(
        config_by_name.keys(),
        key=lambda n: (_STAGE_ORDER.get(config_by_name[n].stage, 99), n),
    )

    for name in all_names:
        cfg = config_by_name[name]
        parents = sorted(cfg.upstream_asset_names)

        hit = catalog_by_name.get(name)
        if hit is not None:
            status = hit["status"]
            phases = hit["phases"]
            wall_s = hit["wall_time_seconds"]
            job_id = hit["job_id"]
        else:
            upstream = [status_map.get(p, "WAITING") for p in parents]
            if any(s in _FAILED_STATES for s in upstream):
                status = "BLOCKED"
            elif not upstream or all(s == "COMPLETED" for s in upstream):
                status = "PENDING"
            else:
                status = "WAITING"
            phases = {}
            wall_s = None
            job_id = ""

        status_map[name] = status
        out.append(
            AssetStatus(
                asset=name,
                stage=cfg.stage,
                model_type=cfg.model_type,
                scale=cfg.scale,
                status=status,
                phases=phases,
                wall_time_seconds=wall_s,
                job_id=job_id,
                upstream=parents,
            )
        )

    return out


def show_pipeline_status(
    *,
    recipe_path: str | None = None,
    dataset: str | None = None,
    seed: int = 42,
) -> None:
    """Show pipeline status from DuckDB catalog joined with planner topology."""
    rows = _collect_statuses(recipe_path=recipe_path, dataset=dataset, seed=seed)

    if not rows:
        print("No assets defined in current recipe.")
        return

    # TODO: rebuild Rich table rendering (deleted: _render_grouped_table,
    # _progress_summary, _phase_symbol, _format_wall_time, _asset_description —
    # Rich grouped-by-stage table with colored status, ✓/✗/- phase columns,
    # wall time, job ID, and a one-line progress summary header)
    print(
        json.dumps(
            {"dataset": dataset, "seed": seed, "assets": [asdict(r) for r in rows]},
            indent=2,
        )
    )
