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

_FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "ERROR"})
_RUNNING_STATES = frozenset({"RUNNING", "PENDING", "UNSET"})
_STAGE_ORDER = {s: i for i, s in enumerate(TOPOLOGY.default_stages)}

# OTel StatusCode → pipeline display status
_OTEL_STATUS_MAP = {
    "OK": "COMPLETED",
    "ERROR": "FAILED",
    "UNSET": "RUNNING",
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

    # The catalog now has OTel span data with run_dir encoding identity.
    # run_dir pattern: .../{dataset}/{family}_{scale}_{stage}_{hash}/seed_{N}
    query = """
        SELECT
            run_dir,
            status_code,
            slurm_job_id,
            epochs_run,
            start_time,
            end_time
        FROM runs
        WHERE run_dir LIKE '%/seed_' || CAST(? AS VARCHAR) || '%'
    """
    params: list = [seed]
    if dataset:
        query += " AND run_dir LIKE '%/' || ? || '/%'"
        params.append(dataset)

    db = duckdb.connect(str(cat), read_only=True)
    try:
        rows = db.execute(query, params).fetchall()
    finally:
        db.close()

    # Build lookup keyed on the directory component that matches asset_name
    # (the {family}_{scale}_{stage}_{hash} segment before /seed_N)
    catalog_by_name: dict[str, dict] = {}
    for run_dir, status_code, slurm_jid, _epochs_run, _start_t, _end_t in rows:
        if not run_dir:
            continue
        # Extract the identity segment: parent of seed_N dir
        parts = Path(run_dir).parts
        # Find seed_N component, take its parent name as asset_name
        asset_name = None
        for i, p in enumerate(parts):
            if p.startswith("seed_") and i > 0:
                asset_name = parts[i - 1]
                break
        if not asset_name:
            continue
        catalog_by_name[asset_name] = {
            "status": _OTEL_STATUS_MAP.get(status_code or "", (status_code or "").upper()),
            "phases": {},
            "wall_time_seconds": None,
            "job_id": slurm_jid or "",
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
