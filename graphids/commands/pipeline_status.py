"""Pipeline status: aggregated view of dagster assets + SLURM phase markers.

Usage:
    python -m graphids pipeline-status
    python -m graphids pipeline-status --limit 30
    python -m graphids pipeline-status --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from graphids.config import PHASE_MARKERS, dataset_names
from graphids.config.runtime import CKPT_SUBPATH, LAKE_ROOT
from graphids.slurm import sacct_by_user

_CKPT_DEPTH = len(Path(CKPT_SUBPATH).parts)


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
            seed_str = jname[idx + len(marker):]
            if seed_str.isdigit():
                out[jname[:idx]] = {
                    "job_id": jid, "state": state, "elapsed": elapsed,
                    "dataset": ds, "seed": seed_str,
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


def _phase_status(run_dir: Path | None) -> dict[str, str]:
    """Check phase marker files in a run directory.

    Returns symbol per phase: ✓ (passed), ✗ (failed), - (unknown/legacy).
    If no markers exist at all, the run predates this feature — show all as -.
    """
    no_data = {"train": "-", "test": "-", "analyze": "-"}
    if run_dir is None or not run_dir.exists():
        return no_data
    raw = {
        phase: (run_dir / marker).exists()
        for phase, marker in PHASE_MARKERS.items()
    }
    # No markers at all → legacy run, don't report false failures
    if not any(raw.values()):
        return no_data
    return {phase: ("✓" if ok else "✗") for phase, ok in raw.items()}


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
            r.asset_entry.last_run_id
            for r in records
            if r.asset_entry.last_run_id
        }
        runs = {
            rid: inst.get_run_by_id(rid)
            for rid in run_ids
        }

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
            partition = (run.tags.get("dagster/partition", "") if run else "")

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

    table = Table(title="Pipeline Status", show_lines=False, expand=True)
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


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graphids pipeline-status",
        description="Aggregated pipeline status from dagster + SLURM phase markers",
    )
    parser.add_argument("--limit", type=int, default=50,
                        help="Max assets to display")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output as JSON instead of table")
    parser.add_argument("--no-sacct", dest="use_sacct", action="store_false",
                        default=True, help="Skip sacct reconciliation (dagster-only)")
    args = parser.parse_args(argv)

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
