"""Pipeline status: aggregated view of dagster assets + SLURM phase markers.

Usage:
    python -m graphids pipeline-status                        # grouped recipe view
    python -m graphids pipeline-status --dataset set_01       # filter partition
    python -m graphids pipeline-status --json                 # JSON for scripting
    python -m graphids pipeline-status --dagster              # legacy flat view
    python -m graphids pipeline-status --log [FILTER]         # orchestrator event log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from graphids.config import COMPLETE_MARKER, PHASE_MARKERS, SLURM_LOG_DIR, dataset_names
from graphids.config.runtime import CKPT_SUBPATH, LAKE_ROOT
from graphids.slurm import sacct_by_user

_CKPT_DEPTH = len(Path(CKPT_SUBPATH).parts)

# ---------------------------------------------------------------------------
# Shared helpers (used by both recipe-aware and legacy paths)
# ---------------------------------------------------------------------------


def _parse_sacct() -> dict[str, dict[str, str]]:
    """Query sacct for recent jobs, return {asset_name: {job_id, state, elapsed, dataset, seed}}.

    Most recent job per asset wins (sacct returns chronological order).
    """
    stdout = sacct_by_user()
    if not stdout:
        return {}
    known_ds = frozenset(dataset_names())
    out: dict[str, dict[str, str]] = {}
    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4 or "." in parts[0]:  # skip .batch/.extern steps
            continue
        jid, jname, state, elapsed = parts[:4]
        state = state.split()[0]  # "CANCELLED by 35950" → "CANCELLED"
        # Parse job name: {asset_name}_{dataset}_s{seed}
        for ds in sorted(known_ds, key=len, reverse=True):
            marker = f"_{ds}_s"
            idx = jname.find(marker)
            if idx < 0:
                continue
            seed_str = jname[idx + len(marker) :]
            if seed_str.isdigit():
                out[jname[:idx]] = {
                    "job_id": jid,
                    "state": state,
                    "elapsed": elapsed,
                    "dataset": ds,
                    "seed": seed_str,
                }
            break
    return out


def _find_run_dir_fs(dataset: str, asset_name: str, seed: str) -> Path | None:
    """Find run directory on filesystem by globbing for asset_name suffix."""
    base = Path(LAKE_ROOT) / "dev" / os.environ.get("USER", "unknown") / dataset
    if not base.is_dir():
        return None
    # Dir name is {model_type}_{scale}_{asset_name}, e.g. vgae_small_autoencoder_288aba35
    matches = list(base.glob(f"*_{asset_name}/seed_{seed}"))
    return matches[0] if matches else None


def _phase_status(run_dir: Path | None) -> dict[str, str]:
    """Check phase marker files in a run directory.

    Returns symbol per phase: ✓ (passed), ✗ (failed), - (unknown/legacy).
    If no markers exist at all, the run predates this feature — show all as -.
    """
    no_data = {"train": "-", "test": "-", "analyze": "-"}
    if run_dir is None or not run_dir.exists():
        return no_data
    raw = {phase: (run_dir / marker).exists() for phase, marker in PHASE_MARKERS.items()}
    # No markers at all → legacy run, don't report false failures
    if not any(raw.values()):
        return no_data
    return {phase: ("✓" if ok else "✗") for phase, ok in raw.items()}


# ---------------------------------------------------------------------------
# Recipe-aware view (default) — uses dagster AssetGraph + sacct + filesystem
# ---------------------------------------------------------------------------

_DONE_STATES = frozenset({"COMPLETED", "SUCCESS"})
_FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"})
_RUNNING_STATES = frozenset({"RUNNING", "PENDING"})

# Stage display order for grouped output
_STAGE_ORDER = {"autoencoder": 0, "normal": 1, "curriculum": 2, "fusion": 3, "temporal": 4}


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


def _derive_status(
    sacct_entry: dict[str, str] | None,
    complete_marker: bool,
    upstream_statuses: list[str],
) -> str:
    """Derive display status from sacct + filesystem + upstream state."""
    if sacct_entry:
        state = sacct_entry["state"]
        if state in _RUNNING_STATES:
            return state
        if complete_marker or state in _DONE_STATES:
            return "COMPLETED"
        if state in _FAILED_STATES:
            return state
        # Other sacct states (COMPLETING, REQUEUED, etc.)
        return state
    # No sacct entry — asset was never submitted (or >30 days ago)
    if complete_marker:
        return "COMPLETED"
    if any(s in _FAILED_STATES for s in upstream_statuses):
        return "BLOCKED"
    if upstream_statuses and all(s in _DONE_STATES | {"COMPLETED"} for s in upstream_statuses):
        return "PENDING"
    if not upstream_statuses:
        return "PENDING"  # root asset, never submitted
    return "WAITING"


def _collect_from_graph(
    *, dataset: str | None = None, seed: int = 42,
) -> list[RecipeAssetStatus]:
    """Load dagster AssetGraph for universe + topology, reconcile with sacct + filesystem."""
    from graphids.orchestrate.definitions import defs

    ag = defs.resolve_asset_graph()
    sacct = _parse_sacct()

    # Build rows in topological order (parents before children)
    status_map: dict[str, str] = {}
    rows: list[RecipeAssetStatus] = []

    # Sort by stage order, then asset name for deterministic output
    all_keys = sorted(
        ag.get_all_asset_keys(),
        key=lambda k: (_STAGE_ORDER.get(ag.get(k).group_name, 99), k.path[0]),
    )

    for key in all_keys:
        name = key.path[0]
        node = ag.get(key)
        parents = sorted(p.path[0] for p in node.parent_keys)

        # sacct entry (filter by dataset/seed if specified)
        sr = sacct.get(name)
        if sr and dataset and sr["dataset"] != dataset:
            sr = None
        if sr and str(seed) != sr.get("seed", "42"):
            sr = None

        # Filesystem: run_dir + phase markers + complete marker
        ds = sr["dataset"] if sr else (dataset or "")
        sd = sr["seed"] if sr else str(seed)
        run_dir = _find_run_dir_fs(ds, name, sd) if ds else None
        phases = _phase_status(run_dir)
        complete = bool(run_dir and (run_dir / COMPLETE_MARKER).exists())

        # Derive status from sacct + filesystem + upstream
        upstream_statuses = [status_map.get(p, "WAITING") for p in parents]
        status = _derive_status(sr, complete, upstream_statuses)
        status_map[name] = status

        rows.append(RecipeAssetStatus(
            asset=name,
            stage=node.group_name,
            label=node.description or f"{node.group_name} ({name})",
            status=status,
            train=phases["train"],
            test=phases["test"],
            analyze=phases["analyze"],
            wall_time=sr["elapsed"] if sr else "",
            job_id=sr["job_id"] if sr else "",
            upstream=parents,
        ))

    return rows


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
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Header
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

    # Group by stage
    from itertools import groupby

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
            # Strip stage prefix from label for compactness: "curriculum (gat, small)" → "gat, small"
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
# Legacy flat view (--dagster)
# ---------------------------------------------------------------------------


@dataclass
class AssetStatus:
    asset: str
    run_status: str
    partition: str
    job_id: str
    wall_time: str
    train: str
    test: str
    analyze: str
    run_dir: str


def _run_dir_from_ckpt(ckpt_path: str) -> Path | None:
    """Derive run_dir by stripping CKPT_SUBPATH depth from checkpoint path."""
    p = Path(ckpt_path)
    if not p.name.endswith(".ckpt"):
        return None
    for _ in range(_CKPT_DEPTH):
        p = p.parent
    return p


def _collect(*, limit: int, use_sacct: bool = True) -> list[AssetStatus]:
    """Query DagsterInstance for asset materialization status, reconciled with sacct."""
    from dagster import DagsterInstance

    os.environ.setdefault(
        "DAGSTER_HOME",
        os.environ.get("KD_GAT_DAGSTER_HOME", "/fs/scratch/PAS1266/dagster"),
    )

    sacct = _parse_sacct() if use_sacct else {}

    rows: list[AssetStatus] = []
    with DagsterInstance.get() as inst:
        keys = inst.get_asset_keys()[:limit]
        records = inst.get_asset_records(keys)

        # Batch-fetch runs, deduplicated by run_id
        run_ids = {
            r.asset_entry.last_run_id for r in records if r.asset_entry.last_run_id
        }
        runs = {rid: inst.get_run_by_id(rid) for rid in run_ids}

        for record in records:
            entry = record.asset_entry
            asset_name = entry.asset_key.path[0]
            event = entry.last_materialization

            if not event:
                rows.append(AssetStatus(
                    asset=asset_name, run_status="NEVER_RUN", partition="",
                    job_id="", wall_time="", train="-", test="-",
                    analyze="-", run_dir="",
                ))
                continue

            # Run status from dagster (cached lookup)
            run = runs.get(event.run_id)
            run_status = run.status.value if run else "UNKNOWN"
            partition = run.tags.get("dagster/partition", "") if run else ""

            # Metadata from materialization
            md = event.asset_materialization.metadata if event.asset_materialization else {}
            ckpt_val = md.get("checkpoint_path")
            ckpt_path = ckpt_val.value if ckpt_val else ""
            job_id_val = md.get("job_id")
            job_id = str(job_id_val.value) if job_id_val else ""
            wall_val = md.get("wall_time")
            wall_time = wall_val.value if wall_val else ""

            # Reconcile with sacct — ground truth for SLURM job state
            sr = sacct.get(asset_name)
            if sr:
                if run_status in ("STARTED", "UNKNOWN"):
                    run_status = sr["state"]
                if not job_id:
                    job_id = sr["job_id"]
                if not wall_time:
                    wall_time = sr["elapsed"]
                if not partition:
                    partition = f"{sr['dataset']}|{sr['seed']}"

            # Phase markers: try dagster metadata, then run_dir metadata, then filesystem
            run_dir = _run_dir_from_ckpt(ckpt_path) if ckpt_path else None
            if run_dir is None:
                rd_val = md.get("run_dir")
                if rd_val:
                    run_dir = Path(rd_val.value)
            if run_dir is None and sr:
                run_dir = _find_run_dir_fs(sr["dataset"], asset_name, sr["seed"])
            phases = _phase_status(run_dir)

            rows.append(AssetStatus(
                asset=asset_name,
                run_status=run_status,
                partition=partition,
                job_id=job_id,
                wall_time=wall_time,
                train=phases["train"],
                test=phases["test"],
                analyze=phases["analyze"],
                run_dir=str(run_dir) if run_dir else "",
            ))

    return rows


def _render_table(rows: list[AssetStatus]) -> None:
    """Print a rich table to stdout."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Pipeline Status (legacy)", show_lines=False, expand=True)
    table.add_column("Asset", style="cyan", no_wrap=True, ratio=3)
    table.add_column("Status", no_wrap=True, ratio=1)
    table.add_column("Partition", ratio=1)
    table.add_column("T", justify="center", width=1)
    table.add_column("E", justify="center", width=1)
    table.add_column("A", justify="center", width=1)
    table.add_column("Wall", justify="right", ratio=1)
    table.add_column("Job ID", justify="right", ratio=1)

    status_styles = {
        "SUCCESS": "green",
        "COMPLETED": "green",
        "FAILURE": "red bold",
        "FAILED": "red bold",
        "OUT_OF_MEMORY": "red bold",
        "TIMEOUT": "red bold",
        "CANCELLED": "yellow",
        "STARTED": "yellow",
        "RUNNING": "yellow",
        "PENDING": "dim",
        "NEVER_RUN": "dim",
        "UNKNOWN": "dim",
    }

    phase_styles = {"✓": "green", "✗": "red bold", "-": "dim"}

    for r in rows:
        style = status_styles.get(r.run_status, "")
        table.add_row(
            r.asset,
            f"[{style}]{r.run_status}[/{style}]" if style else r.run_status,
            r.partition,
            f"[{phase_styles.get(r.train, '')}]{r.train}[/]",
            f"[{phase_styles.get(r.test, '')}]{r.test}[/]",
            f"[{phase_styles.get(r.analyze, '')}]{r.analyze}[/]",
            r.wall_time or "-",
            r.job_id or "-",
        )

    Console().print(table)


# ---------------------------------------------------------------------------
# Orchestrator event log (--log)
# ---------------------------------------------------------------------------


def _latest_log() -> Path | None:
    """Find the most recent orchestrator JSONL log file."""
    log_dir = Path(SLURM_LOG_DIR)
    logs = sorted(log_dir.glob("orchestrator_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


_LOG_FILTERS: dict[str, str | None] = {
    "all": None,
    "failures": "asset_failed",
    "retries": "resource_scaled",
    "completions": "asset_complete",
    "submissions": "submitted",
    "polls": "slurm_poll",
}


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
        description="Pipeline status from dagster asset graph + SLURM",
    )
    # Mode selection
    parser.add_argument("--dagster", action="store_true",
                        help="Legacy flat view from dagster instance only")
    parser.add_argument("--log", nargs="?", const="all", metavar="FILTER",
                        choices=list(_LOG_FILTERS),
                        help="Read orchestrator event log. Filters: "
                             + ", ".join(_LOG_FILTERS))
    parser.add_argument("--follow", "-f", action="store_true",
                        help="Follow log output (like tail -f). Use with --log")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Specific log file (default: latest)")

    # Recipe-aware options
    parser.add_argument("--dataset", "-d", default=None,
                        help="Filter to dataset partition (e.g. set_01)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Seed partition (default: 42)")

    # Output
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON instead of table")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max assets for legacy --dagster view")
    parser.add_argument("--no-sacct", dest="use_sacct", action="store_false",
                        default=True, help="Skip sacct reconciliation")
    args = parser.parse_args(argv)

    # --- Log mode ---
    if args.log is not None:
        log_path = args.log_file or _latest_log()
        if log_path is None or not log_path.exists():
            print(f"No orchestrator logs found in {SLURM_LOG_DIR}/", file=sys.stderr)
            raise SystemExit(1)
        _render_log(log_path, event_filter=_LOG_FILTERS[args.log], follow=args.follow)
        return

    # --- Legacy dagster-only view ---
    if args.dagster:
        try:
            rows = _collect(limit=args.limit, use_sacct=args.use_sacct)
        except Exception as exc:
            print(f"Error querying dagster: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if not rows:
            print("No assets found in dagster instance.")
            return
        if args.as_json:
            print(json.dumps([asdict(r) for r in rows], indent=2))
        else:
            _render_table(rows)
        return

    # --- Recipe-aware grouped view (default) ---
    try:
        rows = _collect_from_graph(dataset=args.dataset, seed=args.seed)
    except Exception as exc:
        print(f"Error loading asset graph: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

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
