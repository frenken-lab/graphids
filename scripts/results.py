#!/usr/bin/env python3
"""Canonical MLflow results table — view profiles in configs/result_views.yml.

The empirical-notes markdown is selective and goes stale. **MLflow is the
trial-state store** (per .claude/rules/data-layout.md); this script is the
sanctioned way to query it. Adding a new model group means editing
configs/result_views.yml — no Python change.

Usage:

    python scripts/results.py --list-views
    python scripts/results.py --view fusion
    python scripts/results.py --view gat --dataset set_01
    python scripts/results.py --view fusion --variant bandit --plan-id 019e0052
    python scripts/results.py --view any --group gat_loss --variant focal --phase fit
    python scripts/results.py --view fusion --all              # no most-recent dedup
    python scripts/results.py --view fusion --format json      # for tooling

CLI filter flags compose with the view's base filter; CLI wins on conflict.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

VIEWS_PATH = Path(__file__).parent.parent / "configs" / "result_views.yml"

# build_search_filter accepts these kwargs; we expose all of them as CLI flags
# so a view can be overridden along any identity axis without editing YAML.
FILTER_KEYS = (
    "dataset",
    "group",
    "variant",
    "seed",
    "phase",
    "cluster",
    "plan_id",
    "plan_module",
    "git_sha",
    "row_name",
    "status",
)


def _load_views() -> dict:
    if not VIEWS_PATH.exists():
        sys.exit(f"missing config: {VIEWS_PATH}")
    return yaml.safe_load(VIEWS_PATH.read_text()) or {}


def _resolve_filter(view: dict, cli_overrides: dict) -> dict:
    """View base filter merged with CLI overrides (CLI wins). Drops Nones."""
    base = dict(view.get("filter") or {})
    base.update({k: v for k, v in cli_overrides.items() if v is not None})
    # build_search_filter wants seed as int if present
    if base.get("seed") is not None:
        base["seed"] = int(base["seed"])
    return base


def _experiments_for(client, group: str | None, dataset: str | None):
    # Experiment names are graphids/{dataset}/{group}; pattern-match accordingly.
    ds = dataset or "%"
    grp = group or "%"
    return client.search_experiments(filter_string=f"name LIKE 'graphids/{ds}/{grp}'")


def fetch(view_name: str, cli_overrides: dict, *, most_recent_only: bool):
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter, configure_tracking_uri

    views = _load_views()
    if view_name not in views:
        sys.exit(f"unknown view {view_name!r}; available: {sorted(views)}")
    view = views[view_name]

    configure_tracking_uri()
    client = MlflowClient()

    flt = _resolve_filter(view, cli_overrides)
    exps = _experiments_for(client, flt.get("group"), flt.get("dataset"))
    if not exps:
        return [], view
    # plan_id / git_sha are commonly passed as 12-char prefixes, but
    # build_search_filter renders `=`. Strip them from the server filter and
    # apply prefix matching client-side after the search.
    prefix_keys = {k: flt.pop(k) for k in ("plan_id", "git_sha") if flt.get(k)}
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in exps],
        filter_string=build_search_filter(**flt) or None,
        order_by=["attributes.start_time DESC"],
        max_results=2000,
    )
    for k, prefix in prefix_keys.items():
        tag = f"graphids.{k}"
        runs = [r for r in runs if r.data.tags.get(tag, "").startswith(prefix)]
    if not most_recent_only:
        return runs, view

    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for r in runs:
        key = (
            r.data.tags.get("graphids.dataset", "?"),
            r.data.tags.get("graphids.variant", "?"),
            r.data.tags.get("graphids.phase", "?"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped, view


def _row(r, metric_keys: list[str]) -> dict:
    m = r.data.metrics
    out = {
        "dataset": r.data.tags.get("graphids.dataset"),
        "variant": r.data.tags.get("graphids.variant"),
        "phase": r.data.tags.get("graphids.phase"),
        "plan_id": r.data.tags.get("graphids.plan_id", "")[:12],
        "git_sha": r.data.tags.get("graphids.git_sha", "")[:12],
        "started": dt.datetime.fromtimestamp(r.info.start_time / 1000).strftime("%m-%d %H:%M"),
        "run_id": r.info.run_id,
        **{k: m.get(k) for k in metric_keys},
    }
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.4f}"
    return str(v)


def _print_table(rows: list[dict], metric_keys: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    rows = sorted(
        rows, key=lambda r: (r.get("dataset") or "", r.get("variant") or "", r.get("phase") or "")
    )
    headers = ["dataset", "variant", "phase", *metric_keys, "plan_id", "git_sha", "started"]
    widths = {h: max(len(h), max((len(_fmt(r.get(h))) for r in rows), default=0)) for h in headers}
    fmt = "  ".join(f"{{:{widths[h]}s}}" for h in headers)
    print(fmt.format(*headers))
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print(fmt.format(*[_fmt(r.get(h)) for h in headers]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--view", help="View profile name (see configs/result_views.yml)")
    p.add_argument("--list-views", action="store_true", help="List available view names and exit.")
    for k in FILTER_KEYS:
        p.add_argument(
            f"--{k.replace('_', '-')}", dest=k, default=None, help=f"Override filter.{k}"
        )
    p.add_argument(
        "--all",
        dest="most_recent_only",
        action="store_false",
        default=True,
        help="Show all rows (no most-recent dedup per dataset+variant+phase).",
    )
    p.add_argument("--format", choices=("table", "json"), default="table")
    args = p.parse_args(argv)

    if args.list_views:
        for name, v in _load_views().items():
            print(f"{name:14s} {v.get('filter') or {}}")
        return 0
    if not args.view:
        p.error("--view is required (or pass --list-views)")

    cli_overrides = {k: getattr(args, k) for k in FILTER_KEYS}
    runs, view = fetch(args.view, cli_overrides, most_recent_only=args.most_recent_only)
    metric_keys = view.get("metrics") or []
    rows = [_row(r, metric_keys) for r in runs]

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows, metric_keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
