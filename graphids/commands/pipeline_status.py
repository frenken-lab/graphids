"""Pipeline status: aggregated view of dagster assets backed by the DuckDB catalog.

Usage:
    python -m graphids pipeline-status                        # grouped recipe view
    python -m graphids pipeline-status --dataset set_01       # filter partition
    python -m graphids pipeline-status --json                 # JSON for scripting
    python -m graphids pipeline-status --log [FILTER]         # orchestrator event log
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from graphids.config import SLURM_LOG_DIR, catalog_path
from graphids.config.runtime import LAKE_ROOT

# ---------------------------------------------------------------------------
# Constants + dataclass
# ---------------------------------------------------------------------------

_FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"})
_RUNNING_STATES = frozenset({"RUNNING", "PENDING"})
_STAGE_ORDER = {"autoencoder": 0, "normal": 1, "curriculum": 2, "fusion": 3}

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
# Catalog view (the only source)
# ---------------------------------------------------------------------------


def _phase_symbol(phases: dict, key: str) -> str:
    """Phase symbol: ✓ (passed), ✗ (failed), - (unrecorded / in-progress).

    ``None`` is treated as unrecorded because DuckDB's ``read_json_auto``
    with ``union_by_name=true`` inflates an empty ``phases: {}`` dict to a
    struct with NULL fields, which round-trips as JSON ``null`` values.
    """
    val = phases.get(key)
    if val is None:
        return "-"
    return "✓" if val else "✗"


def _format_wall_time(seconds: float | None) -> str:
    if not seconds:
        return ""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _collect_from_catalog(
    *, dataset: str | None = None, seed: int = 42,
) -> list[RecipeAssetStatus]:
    """Load topology from dagster AssetGraph, status from DuckDB catalog.

    The catalog row's ``asset_name`` column (computed as
    ``stage || identity_hash || kd_tag`` in ``rebuild_catalog``) matches the
    dagster asset name exactly — single dict lookup, no fuzzy matching.
    """
    import duckdb

    from graphids.orchestrate.definitions import defs

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

    ag = defs.resolve_asset_graph()
    status_map: dict[str, str] = {}
    out: list[RecipeAssetStatus] = []

    all_keys = sorted(
        ag.get_all_asset_keys(),
        key=lambda k: (_STAGE_ORDER.get(ag.get(k).group_name, 99), k.path[0]),
    )

    for key in all_keys:
        name = key.path[0]
        node = ag.get(key)
        parents = sorted(p.path[0] for p in node.parent_keys)

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
        out.append(RecipeAssetStatus(
            asset=name,
            stage=node.group_name,
            label=node.description or f"{node.group_name} ({name})",
            status=status,
            train=train,
            test=test,
            analyze=analyze,
            wall_time=wall,
            job_id=job_id,
            upstream=parents,
        ))

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
    phase_styles = {"✓": "green", "✗": "red bold", "-": "dim"}

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
            # Strip stage prefix from label: "curriculum (gat, small)" → "gat, small"
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

        print(f"--- following {log_path.name} (Ctrl-C to stop) ---",
              file=sys.stderr, flush=True)
        try:
            while True:
                line = f.readline()
                if line:
                    _print_event(line, event_filter)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graphids pipeline-status",
        description="Pipeline status from dagster asset graph + DuckDB catalog",
    )
    parser.add_argument("--log", nargs="?", const="all", metavar="FILTER",
                        choices=list(_LOG_FILTERS),
                        help="Read orchestrator event log. Filters: "
                             + ", ".join(_LOG_FILTERS))
    parser.add_argument("--follow", "-f", action="store_true",
                        help="Follow log output (like tail -f). Use with --log")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Specific log file (default: latest)")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Filter to dataset partition (e.g. set_01)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Seed partition (default: 42)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON instead of table")
    args = parser.parse_args(argv)

    # --- Log mode ---
    if args.log is not None:
        log_path = args.log_file or _latest_log()
        if log_path is None or not log_path.exists():
            print(f"No orchestrator logs found in {SLURM_LOG_DIR}/", file=sys.stderr)
            raise SystemExit(1)
        _render_log(log_path, event_filter=_LOG_FILTERS[args.log], follow=args.follow)
        return

    # --- Recipe-aware grouped view ---
    rows = _collect_from_catalog(dataset=args.dataset, seed=args.seed)

    if not rows:
        print("No assets defined in current recipe.")
        return

    if args.as_json:
        summary = _progress_summary(rows)
        print(json.dumps({
            "dataset": args.dataset,
            "seed": args.seed,
            "summary": summary,
            "assets": [asdict(r) for r in rows],
        }, indent=2))
    else:
        _render_grouped_table(rows, args.dataset, args.seed)
