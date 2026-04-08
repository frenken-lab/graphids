"""Pipeline status: aggregated view of training assets backed by the DuckDB catalog.

Topology comes from the planner (``enumerate_assets``), status from
the DuckDB catalog. No dagster dependency.

The public entry point is :func:`show_pipeline_status`; ``_LOG_FILTERS``
is re-exported so the CLI shim can build ``choices=`` from its keys.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from graphids.config.constants import CONFIG_DIR, LAKE_ROOT
from graphids.config.topology import (
    TOPOLOGY,  # noqa: F401 (used in module body)
    catalog_path,
)
from graphids.slurm.env import SLURM_LOG_DIR

RECIPES_DIR = CONFIG_DIR / "recipes"

# ---------------------------------------------------------------------------
# Constants + dataclass
# ---------------------------------------------------------------------------

_FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"})
_RUNNING_STATES = frozenset({"RUNNING", "PENDING"})
_STAGE_ORDER = {s: i for i, s in enumerate(TOPOLOGY.default_stages)}

_CATALOG_STATUS_MAP = {
    "completed": "COMPLETED",
    "failed": "FAILED",
    "started": "RUNNING",
}


@dataclass
class RecipeAssetStatus:
    asset: str
    stage: str
    label: str
    status: str
    train: str
    test: str
    analyze: str
    wall_time: str
    job_id: str
    upstream: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Topology from planner (replaces dagster AssetGraph)
# ---------------------------------------------------------------------------


def _load_topology(recipe_path: str | Path | None = None) -> list:
    """Load asset topology from the planner.

    Returns a list of ``StageConfig`` objects with ``asset_name``,
    ``stage``, ``upstream_asset_names``, ``model_type``, ``scale``,
    and ``model_init_overrides`` — everything needed for the status table.
    """
    from graphids.config.jsonnet import render
    from graphids.orchestrate.planning import enumerate_assets, expand_recipe_configs

    if recipe_path is None:
        from graphids.config.settings import get_settings

        recipe_env = get_settings().recipe
        recipe_path = Path(recipe_env) if recipe_env else RECIPES_DIR / "ablation.jsonnet"

    raw = render(Path(recipe_path))
    expanded = expand_recipe_configs(raw)
    return enumerate_assets(TOPOLOGY.model_dump(), expanded)


def _asset_description(cfg) -> str:
    """Human-readable asset description from a StageConfig."""
    parts = [cfg.model_type, cfg.scale]
    for k, v in sorted(cfg.model_init_overrides.items()):
        parts.append(f"{k}={v}")
    return f"{cfg.stage} ({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Catalog view (the only source)
# ---------------------------------------------------------------------------


def _phase_symbol(phases: dict, key: str) -> str:
    """Phase symbol: checkmark (passed), x (failed), - (unrecorded / in-progress).

    ``None`` is treated as unrecorded because DuckDB's ``read_json_auto``
    with ``union_by_name=true`` inflates an empty ``phases: {}`` dict to a
    struct with NULL fields, which round-trips as JSON ``null`` values.
    """
    val = phases.get(key)
    if val is None:
        return "-"
    return "\u2713" if val else "\u2717"


def _format_wall_time(seconds: float | None) -> str:
    if not seconds:
        return ""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _collect_from_catalog(
    *,
    recipe_path: str | Path | None = None,
    dataset: str | None = None,
    seed: int = 42,
) -> list[RecipeAssetStatus]:
    """Load topology from planner, status from DuckDB catalog.

    The catalog row's ``asset_name`` column (computed as
    ``stage || identity_hash || kd_tag`` in ``rebuild_catalog``) matches the
    planner asset name exactly — single dict lookup, no fuzzy matching.
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
            "wall_time": _format_wall_time(wall_s),
            "job_id": job_id_str or "",
        }

    # Build status list from planner topology (replaces dagster AssetGraph)
    status_map: dict[str, str] = {}
    out: list[RecipeAssetStatus] = []

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
            train = _phase_symbol(phases, "train")
            test = _phase_symbol(phases, "test")
            analyze = _phase_symbol(phases, "analyze")
            wall = hit["wall_time"]
            job_id = hit["job_id"]
        else:
            # No catalog row — infer pending/blocked/waiting from upstream state
            upstream = [status_map.get(p, "WAITING") for p in parents]
            if any(s in _FAILED_STATES for s in upstream):
                status = "BLOCKED"
            elif not upstream or all(s == "COMPLETED" for s in upstream):
                status = "PENDING"
            else:
                status = "WAITING"
            train = test = analyze = "-"
            wall = job_id = ""

        status_map[name] = status
        out.append(
            RecipeAssetStatus(
                asset=name,
                stage=cfg.stage,
                label=_asset_description(cfg),
                status=status,
                train=train,
                test=test,
                analyze=analyze,
                wall_time=wall,
                job_id=job_id,
                upstream=parents,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _progress_summary(rows: list[RecipeAssetStatus]) -> str:
    """One-line progress: '15/32 done, 2 failed, 3 running, 12 pending'."""
    total = len(rows)
    counts: dict[str, int] = {}
    for r in rows:
        if r.status == "COMPLETED":
            bucket = "done"
        elif r.status in _FAILED_STATES:
            bucket = "failed"
        elif r.status in _RUNNING_STATES:
            bucket = "running"
        else:
            bucket = "pending"
        counts[bucket] = counts.get(bucket, 0) + 1

    parts = [f"{counts.get('done', 0)}/{total} done"]
    for key in ("failed", "running", "pending"):
        if counts.get(key, 0) > 0:
            parts.append(f"{counts[key]} {key}")
    return ", ".join(parts)


def _render_grouped_table(
    rows: list[RecipeAssetStatus],
    dataset: str | None,
    seed: int,
) -> None:
    """Rich table grouped by stage with progress header."""
    from itertools import groupby

    from rich.console import Console
    from rich.table import Table

    console = Console()

    ds_label = dataset or "all"
    summary = _progress_summary(rows)
    console.print(f"\n[bold]Pipeline Status[/bold]: {ds_label} seed {seed} -- {summary}\n")

    status_styles = {
        "COMPLETED": "green",
        "SUCCESS": "green",
        "FAILED": "red bold",
        "TIMEOUT": "red bold",
        "OUT_OF_MEMORY": "red bold",
        "CANCELLED": "yellow",
        "RUNNING": "yellow",
        "PENDING": "dim",
        "BLOCKED": "magenta",
        "WAITING": "dim",
    }
    phase_styles = {"\u2713": "green", "\u2717": "red bold", "-": "dim"}

    for stage, group in groupby(rows, key=lambda r: r.stage):
        items = list(group)
        table = Table(
            title=f"{stage.upper()} ({len(items)})",
            title_style="bold cyan",
            show_lines=False,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Asset", style="cyan", no_wrap=True, ratio=3)
        table.add_column("Config", ratio=3)
        table.add_column("Status", no_wrap=True, ratio=1)
        table.add_column("T", justify="center", width=1)
        table.add_column("E", justify="center", width=1)
        table.add_column("A", justify="center", width=1)
        table.add_column("Wall", justify="right", ratio=1)
        table.add_column("Job", justify="right", ratio=1)

        for r in items:
            ss = status_styles.get(r.status, "")
            # Strip stage prefix from label: "curriculum (gat, small)" -> "gat, small"
            label = r.label
            if label.startswith(f"{r.stage} (") and label.endswith(")"):
                label = label[len(r.stage) + 2 : -1]
            table.add_row(
                r.asset,
                label,
                f"[{ss}]{r.status}[/{ss}]" if ss else r.status,
                f"[{phase_styles.get(r.train, '')}]{r.train}[/]",
                f"[{phase_styles.get(r.test, '')}]{r.test}[/]",
                f"[{phase_styles.get(r.analyze, '')}]{r.analyze}[/]",
                r.wall_time or "-",
                r.job_id or "-",
            )

        console.print(table)
        console.print()


# ---------------------------------------------------------------------------
# Orchestrator event log (--log)
# ---------------------------------------------------------------------------


_LOG_FILTERS: dict[str, str | None] = {
    "all": None,
    "failures": "asset_failed",
    "retries": "resource_scaled",
    "completions": "asset_complete",
    "submissions": "submitted",
    "polls": "slurm_poll",
}


def _latest_log() -> Path | None:
    """Find the most recent orchestrator JSONL log file."""
    log_dir = Path(SLURM_LOG_DIR)
    logs = sorted(log_dir.glob("orchestrator_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _print_event(line: str, event_filter: str | None) -> None:
    """Parse one JSONL line and print if it matches the filter."""
    line = line.strip()
    if not line:
        return
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if event_filter and event.get("event") != event_filter:
        return
    print(json.dumps(event, indent=2), flush=True)


def _render_log(log_path: Path, *, event_filter: str | None, follow: bool) -> None:
    """Read orchestrator JSONL and pretty-print, optionally tailing for new events."""
    import time

    with open(log_path) as f:
        for line in f:
            _print_event(line, event_filter)

        if not follow:
            return

        print(f"--- following {log_path.name} (Ctrl-C to stop) ---", file=sys.stderr, flush=True)
        try:
            while True:
                line = f.readline()
                if line:
                    _print_event(line, event_filter)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass


def show_pipeline_status(
    *,
    recipe_path: str | None = None,
    dataset: str | None = None,
    seed: int = 42,
    as_json: bool = False,
    log_filter: str | None = None,
    log_file: Path | None = None,
    follow: bool = False,
) -> None:
    """Render pipeline status or orchestrator event log.

    When ``log_filter`` is set (a key into :data:`_LOG_FILTERS`), this
    reads the orchestrator JSONL at ``log_file`` (or the latest if omitted)
    and pretty-prints matching events, optionally tailing. Otherwise it
    reads asset topology from the planner and joins with the DuckDB catalog
    to render a grouped-by-stage table (or JSON when ``as_json`` is set).
    """
    # --- Log mode ---
    if log_filter is not None:
        log_path = log_file or _latest_log()
        if log_path is None or not log_path.exists():
            print(f"No orchestrator logs found in {SLURM_LOG_DIR}/", file=sys.stderr)
            raise SystemExit(1)
        _render_log(log_path, event_filter=_LOG_FILTERS[log_filter], follow=follow)
        return

    # --- Recipe-aware grouped view ---
    rows = _collect_from_catalog(recipe_path=recipe_path, dataset=dataset, seed=seed)

    if not rows:
        print("No assets defined in current recipe.")
        return

    if as_json:
        summary = _progress_summary(rows)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "summary": summary,
                    "assets": [asdict(r) for r in rows],
                },
                indent=2,
            )
        )
    else:
        _render_grouped_table(rows, dataset, seed)
