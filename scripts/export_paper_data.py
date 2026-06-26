#!/usr/bin/env python3
"""Export paper metrics CSVs from MLflow — bypasses HuggingFace.

Queries the graphids MLflow database directly and writes leaderboard,
effect_size, expected_max, and tie_candidates CSVs to the kd-gat-paper
data directory. Validates the written files against the paper's schemas.yaml.

Usage:
    python scripts/export_paper_data.py
    python scripts/export_paper_data.py --out ~/kd-gat-paper/data/csv
    python scripts/export_paper_data.py --metric f1_macro --tie-threshold 0.02
    python scripts/export_paper_data.py --dry-run
    python scripts/export_paper_data.py --no-validate
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path.home() / "kd-gat-paper"
DEFAULT_OUT = PAPER_ROOT / "data" / "csv"
DEFAULT_METRIC = "f1_macro"
TIE_THRESHOLD = 0.02

_RESULTS_TABLE_METRICS = ["f1_macro", "auroc_macro", "ap_macro", "mcc", "accuracy"]

_SPLIT_ORDER = [
    "pooled",
    "test",
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
    "test_05_suppress",
    "test_06_masquerade",
]

_SPLIT_LABELS: dict[str, str] = {
    "pooled": "Pooled",
    "test": "Test",
    "test_01_known_vehicle_known_attack": "Known veh. / Known att.",
    "test_02_unknown_vehicle_known_attack": "Unknown veh. / Known att.",
    "test_03_known_vehicle_unknown_attack": "Known veh. / Unknown att.",
    "test_04_unknown_vehicle_unknown_attack": "Unknown veh. / Unknown att.",
    "test_05_suppress": "Suppression",
    "test_06_masquerade": "Masquerade",
}

_VARIANT_LABELS: dict[str, str] = {
    "ce": "Cross-Entropy",
    "focal": "Focal",
    "weighted_ce": "Weighted CE",
    "none": "No sampling",
    "hash": "Hash encoding",
    "lookup": "Lookup encoding",
    "moe": "MoE",
    "moe_noaux": "MoE (no aux)",
    "dqn": "DQN",
    "mlp": "MLP",
    "bandit": "Bandit",
    "weighted_avg": "Weighted avg.",
    "teacher_vgae": "Teacher VGAE",
    "teacher_gat": "Teacher GAT",
}

_DATASET_ORDER = ["hcrl_sa", "set_01", "set_02", "set_03", "set_04"]

DEFAULT_JSON_OUT = (
    PAPER_ROOT / "interactive" / "src" / "figures" / "data" / "results-table" / "data.json"
)

# Matches schemas.yaml input.*.sort_by / descending
_SORT: dict[str, tuple[list[str], list[bool]]] = {
    "leaderboard": (["group", "dataset", "mean"], [False, False, True]),
    "effect_size": (["group", "dataset", "mean_diff"], [False, False, True]),
    "expected_max": (["group", "dataset", "expected_max"], [False, False, True]),
    "tie_candidates": (["group", "dataset", "gap"], [False, False, False]),
}


# ── MLflow helpers ────────────────────────────────────────────────────────────


def _configure_mlflow() -> None:
    import mlflow  # noqa: F401 — side-effect import sets defaults

    from graphids._mlflow import configure_tracking_uri

    os.environ.setdefault("GRAPHIDS_LAKE_ROOT", "/fs/ess/PAS1266/graphids")
    configure_tracking_uri()


def _fetch_test_runs(metric: str) -> list[dict]:
    """Fetch FINISHED test runs; dedup to most-recent per (dataset, group, variant, seed)."""
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter

    client = MlflowClient()
    exps = client.search_experiments(filter_string="name LIKE 'graphids/%/%'")
    if not exps:
        return []

    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in exps],
        filter_string=build_search_filter(phase="test", status="FINISHED"),
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )

    seen: set[tuple] = set()
    records: list[dict] = []
    for r in runs:
        t = r.data.tags
        key = (
            t.get("graphids.dataset", "?"),
            t.get("graphids.group", "?"),
            t.get("graphids.variant", "?"),
            t.get("graphids.seed", "?"),
        )
        if key in seen:
            continue
        seen.add(key)
        val = r.data.metrics.get(metric)
        if val is None:
            continue
        records.append(
            {
                "dataset": key[0],
                "group": key[1],
                "variant": key[2],
                "seed": key[3],
                "value": float(val),
            }
        )
    return records


def _fetch_results_table_runs() -> list[dict]:
    """Fetch per-split metrics for results table. Deduped to most-recent per identity."""
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter

    client = MlflowClient()
    exps = client.search_experiments(filter_string="name LIKE 'graphids/%/%'")
    if not exps:
        return []

    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in exps],
        filter_string=build_search_filter(phase="test", status="FINISHED"),
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )

    seen: set[tuple] = set()
    records: list[dict] = []
    for r in runs:
        t = r.data.tags
        key = (
            t.get("graphids.dataset", "?"),
            t.get("graphids.group", "?"),
            t.get("graphids.variant", "?"),
            t.get("graphids.seed", "?"),
        )
        if key in seen:
            continue
        seen.add(key)
        dataset, group, variant, seed = key
        raw = r.data.metrics
        base = {"dataset": dataset, "group": group, "variant": variant, "seed": seed}

        # Top-level pooled metrics (no split prefix)
        pooled = {m: raw.get(m) for m in _RESULTS_TABLE_METRICS}
        if any(v is not None for v in pooled.values()):
            records.append({**base, "split": "pooled", **pooled})

        # Per-split metrics: keys like "test/{split_name}/{metric}"
        split_names: set[str] = set()
        for mkey in raw:
            parts = mkey.split("/")
            if len(parts) == 3 and parts[0] == "test" and parts[2] in _RESULTS_TABLE_METRICS:
                split_names.add(parts[1])
        for split_name in sorted(split_names):
            sm = {m: raw.get(f"test/{split_name}/{m}") for m in _RESULTS_TABLE_METRICS}
            if any(v is not None for v in sm.values()):
                records.append({**base, "split": split_name, **sm})

    return records


# ── statistics ────────────────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _ci95(vals: list[float]) -> tuple[float, float]:
    n, m = len(vals), _mean(vals)
    if n < 2:
        return m, m
    try:
        from scipy import stats

        t = stats.t.ppf(0.975, df=n - 1)
    except ImportError:
        _T = {
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
        }
        t = _T.get(n, 2.0)
    se = _std(vals) / math.sqrt(n)
    return m - t * se, m + t * se


def _cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = _std(a), _std(b)
    pooled = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float("nan") if pooled == 0 else (_mean(a) - _mean(b)) / pooled


def _diff_ci95(a: list[float], b: list[float]) -> tuple[float, float]:
    na, nb = len(a), len(b)
    diff = _mean(a) - _mean(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    sa, sb = _std(a), _std(b)
    se = math.sqrt(sa**2 / na + sb**2 / nb)
    try:
        from scipy import stats

        # Welch–Satterthwaite degrees of freedom
        num = (sa**2 / na + sb**2 / nb) ** 2
        denom = (sa**2 / na) ** 2 / (na - 1) + (sb**2 / nb) ** 2 / (nb - 1)
        df = num / denom if denom > 0 else na + nb - 2
        t = stats.t.ppf(0.975, df=df)
    except ImportError:
        t = 2.0
    return diff - t * se, diff + t * se


def _expected_max(vals: list[float]) -> float:
    """E[max of n i.i.d. draws] via normal order-statistic approximation (Blom)."""
    n = len(vals)
    if n == 1:
        return vals[0]
    m, s = _mean(vals), _std(vals)
    if math.isnan(s) or s == 0:
        return m
    try:
        from scipy import stats

        ez_max = stats.norm.ppf((n - 0.375) / (n + 0.25))
    except ImportError:
        _EZ = {2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163}
        ez_max = _EZ.get(n, 1.5)
    return m + s * ez_max


def _fmt(v: float) -> str:
    return "" if math.isnan(v) else str(round(v, 6))


# ── builders ──────────────────────────────────────────────────────────────────


def _aggregate(records: list[dict]) -> dict[tuple, list[float]]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        buckets[(r["group"], r["dataset"], r["variant"])].append(r["value"])
    return dict(buckets)


def _by_group_ds(buckets: dict) -> dict[tuple, dict[str, list[float]]]:
    out: dict[tuple, dict[str, list[float]]] = defaultdict(dict)
    for (group, dataset, variant), vals in buckets.items():
        out[(group, dataset)][variant] = vals
    return dict(out)


def _sorted(rows: list[dict], name: str) -> list[dict]:
    keys, descs = _SORT[name]
    return sorted(
        rows,
        key=lambda r: tuple(
            (-r[k] if isinstance(r.get(k), (int, float)) and d else r.get(k, ""))
            for k, d in zip(keys, descs)
        ),
    )


def build_leaderboard(buckets: dict) -> list[dict]:
    rows = []
    for (group, dataset, variant), vals in buckets.items():
        ci_low, ci_high = _ci95(vals)
        rows.append(
            {
                "variant": variant,
                "n_seeds": len(vals),
                "mean": round(_mean(vals), 6),
                "ci_low": round(ci_low, 6),
                "ci_high": round(ci_high, 6),
                "group": group,
                "dataset": dataset,
            }
        )
    return _sorted(rows, "leaderboard")


def build_effect_size(buckets: dict) -> list[dict]:
    rows = []
    for (group, dataset), variants in sorted(_by_group_ds(buckets).items()):
        vlist = sorted(variants.items(), key=lambda x: -_mean(x[1]))
        for i, (va, a) in enumerate(vlist):
            for vb, b in vlist[i + 1 :]:
                diff = _mean(a) - _mean(b)
                d = _cohens_d(a, b)
                ci_low, ci_high = _diff_ci95(a, b)
                rows.append(
                    {
                        "variant_a": va,
                        "variant_b": vb,
                        "mean_diff": round(diff, 6),
                        "cohens_d": _fmt(d),
                        "diff_ci_low": _fmt(ci_low),
                        "diff_ci_high": _fmt(ci_high),
                        "n_seeds_a": len(a),
                        "n_seeds_b": len(b),
                        "group": group,
                        "dataset": dataset,
                    }
                )
    return _sorted(rows, "effect_size")


def build_expected_max(buckets: dict) -> list[dict]:
    rows = []
    for (group, dataset, variant), vals in buckets.items():
        rows.append(
            {
                "variant": variant,
                "n_seeds": len(vals),
                "expected_max": round(_expected_max(vals), 6),
                "group": group,
                "dataset": dataset,
            }
        )
    return _sorted(rows, "expected_max")


def build_tie_candidates(buckets: dict, threshold: float) -> list[dict]:
    rows = []
    for (group, dataset), variants in sorted(_by_group_ds(buckets).items()):
        vlist = sorted(variants.items(), key=lambda x: -_mean(x[1]))
        for i, (va, a) in enumerate(vlist):
            for vb, b in vlist[i + 1 :]:
                gap = abs(_mean(a) - _mean(b))
                if gap < threshold:
                    rows.append(
                        {
                            "variant_a": va,
                            "variant_b": vb,
                            "mean_a": round(_mean(a), 6),
                            "mean_b": round(_mean(b), 6),
                            "gap": round(gap, 6),
                            "n_seeds_a": len(a),
                            "n_seeds_b": len(b),
                            "group": group,
                            "dataset": dataset,
                        }
                    )
    return _sorted(rows, "tie_candidates")


def build_results_table_json(records: list[dict]) -> dict:
    """Aggregate per-split MLflow metrics into results-table data.json format."""
    # Mean-aggregate across seeds per (group, variant, dataset, split)
    agg: dict[tuple, dict[str, list[float]]] = {}
    for r in records:
        k = (r["group"], r["variant"], r["dataset"], r["split"])
        if k not in agg:
            agg[k] = defaultdict(list)
        for m in _RESULTS_TABLE_METRICS:
            if r.get(m) is not None:
                agg[k][m].append(float(r[m]))

    # Collect dataset and per-dataset split inventories
    datasets_seen: list[str] = []
    splits_by_dataset: dict[str, list[str]] = defaultdict(list)
    for _group, _variant, dataset, split in agg:
        if dataset not in datasets_seen:
            datasets_seen.append(dataset)
        if split not in splits_by_dataset[dataset]:
            splits_by_dataset[dataset].append(split)

    datasets = [d for d in _DATASET_ORDER if d in datasets_seen] + [
        d for d in datasets_seen if d not in _DATASET_ORDER
    ]
    for ds in datasets:
        raw_sp = splits_by_dataset[ds]
        splits_by_dataset[ds] = [s for s in _SPLIT_ORDER if s in raw_sp] + [
            s for s in sorted(raw_sp) if s not in _SPLIT_ORDER
        ]

    all_splits = list(dict.fromkeys(s for sl in splits_by_dataset.values() for s in sl))

    rows = []
    for (group, variant, dataset, split), m_vals in sorted(agg.items()):
        row: dict = {
            "variant": variant,
            "group": group,
            "dataset": dataset,
            "split": split,
            "n_seeds": max((len(v) for v in m_vals.values()), default=1),
        }
        for m in _RESULTS_TABLE_METRICS:
            vals = m_vals.get(m, [])
            row[m] = round(_mean(vals), 6) if vals else None
        rows.append(row)

    return {
        "datasets": datasets,
        "splits_by_dataset": dict(splits_by_dataset),
        "split_labels": {s: _SPLIT_LABELS.get(s, s) for s in all_splits},
        "metrics": _RESULTS_TABLE_METRICS,
        "metric_labels": {
            "f1_macro": "F1",
            "auroc_macro": "AUROC",
            "ap_macro": "AP",
            "mcc": "MCC",
            "accuracy": "Accuracy",
        },
        "variant_labels": _VARIANT_LABELS,
        "rows": rows,
    }


# ── I/O ───────────────────────────────────────────────────────────────────────


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict], dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] {len(rows):4d} rows → {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows):4d} rows → {path}")


def _write_json(path: Path, obj: dict, dry_run: bool) -> None:
    import json as _json

    n = len(obj.get("rows", []))
    if dry_run:
        print(f"  [dry-run] {n:4d} rows → {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump(obj, f, indent=2)
    print(f"  {n:4d} rows → {path}")


def _validate(out: Path) -> bool:
    """Validate only the four CSVs we write against the paper's schemas."""
    validator_path = PAPER_ROOT / "tools" / "validate" / "inputs" / "data.py"
    if not validator_path.exists():
        print(f"  [warn] validator not found at {validator_path}; skipping")
        return True

    sys.path.insert(0, str(PAPER_ROOT / "tools" / "validate" / "inputs"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("_data_validator", validator_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    schemas = mod.load_schemas()
    _csv = schemas.get("csv", {})
    targets = {"leaderboard.csv", "effect_size.csv", "expected_max.csv", "tie_candidates.csv"}
    errors: list[str] = []
    for name in targets:
        if name in _csv:
            errors.extend(mod.validate_csv(name, _csv[name]))
    if errors:
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        return False
    print(f"  OK: {len(targets)} CSVs validated")
    return True


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for CSVs")
    p.add_argument("--metric", default=DEFAULT_METRIC, help="MLflow metric key (default: f1_macro)")
    p.add_argument(
        "--tie-threshold",
        type=float,
        default=TIE_THRESHOLD,
        help="Gap threshold for tie_candidates (default: 0.02)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print what would be written without writing files"
    )
    p.add_argument("--no-validate", action="store_true", help="Skip kd-gat-paper schema validator")
    args = p.parse_args(argv)

    out = Path(args.out).expanduser()
    sys.path.insert(0, str(REPO_ROOT))
    _configure_mlflow()

    print(f"Fetching FINISHED test runs (metric={args.metric!r})...")
    records = _fetch_test_runs(args.metric)
    if not records:
        print("No FINISHED test runs with that metric — nothing to write.", file=sys.stderr)
        return 1

    datasets = {r["dataset"] for r in records}
    combos = {(r["group"], r["variant"]) for r in records}
    print(
        f"  {len(records)} seed-runs across {len(datasets)} datasets, "
        f"{len(combos)} group/variant combos"
    )

    buckets = _aggregate(records)

    lb = build_leaderboard(buckets)
    es = build_effect_size(buckets)
    em = build_expected_max(buckets)
    tc = build_tie_candidates(buckets, args.tie_threshold)

    _write_csv(
        out / "leaderboard.csv",
        ["variant", "n_seeds", "mean", "ci_low", "ci_high", "group", "dataset"],
        lb,
        args.dry_run,
    )
    _write_csv(
        out / "effect_size.csv",
        [
            "variant_a",
            "variant_b",
            "mean_diff",
            "cohens_d",
            "diff_ci_low",
            "diff_ci_high",
            "n_seeds_a",
            "n_seeds_b",
            "group",
            "dataset",
        ],
        es,
        args.dry_run,
    )
    _write_csv(
        out / "expected_max.csv",
        ["variant", "n_seeds", "expected_max", "group", "dataset"],
        em,
        args.dry_run,
    )
    _write_csv(
        out / "tie_candidates.csv",
        [
            "variant_a",
            "variant_b",
            "mean_a",
            "mean_b",
            "gap",
            "n_seeds_a",
            "n_seeds_b",
            "group",
            "dataset",
        ],
        tc,
        args.dry_run,
    )

    if not args.dry_run and not args.no_validate:
        print("\nValidating written CSVs against schemas.yaml...")
        if not _validate(out):
            return 1

    print("\nBuilding results-table JSON...")
    rt_records = _fetch_results_table_runs()
    if not rt_records:
        print("  [warn] no results-table records found", file=sys.stderr)
    else:
        rt_json = build_results_table_json(rt_records)
        _write_json(DEFAULT_JSON_OUT, rt_json, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
