"""Export experiment results to static JSON for the Quarto reports site.

Data sources:
  - Datalake: data/datalake/*.parquet (primary — metadata + metrics)
  - Filesystem: experimentruns/{ds}/{run}/ (binary artifacts)
  - Catalog: config/datasets.yaml

Usage:
    python -m graphids.pipeline.export [--output-dir reports/data]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from graphids.config.constants import graph_attack_type, graph_node_attack_type

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("reports/data")
EXPERIMENT_ROOT = Path(os.environ.get("KD_GAT_EXPERIMENT_ROOT", "experimentruns"))
_data_root = os.environ.get("KD_GAT_DATA_ROOT")
_DATALAKE_ROOT = Path(_data_root) / "datalake" if _data_root else Path("data/datalake")


def _versioned_envelope(data: list | dict) -> dict:
    """Wrap export data with schema version and timestamp."""
    return {
        "schema_version": "1.0.0",
        "exported_at": datetime.now(UTC).isoformat(),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Data source: datalake Parquet (primary) with filesystem fallback
# ---------------------------------------------------------------------------


def _scan_runs() -> list[dict]:
    """Load run metadata from datalake Parquet, with filesystem dir paths.

    Falls back to filesystem scan if datalake doesn't exist.
    """
    runs_parquet = _DATALAKE_ROOT / "runs.parquet"
    if runs_parquet.exists():
        return _scan_runs_from_datalake()
    return _scan_runs_from_filesystem()


def _scan_runs_from_datalake() -> list[dict]:
    """Read run metadata from datalake Parquet, attach filesystem paths."""
    import duckdb

    datalake = str(_DATALAKE_ROOT)
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT run_id, dataset, model_type, scale, stage, has_kd, success
        FROM '{datalake}/runs.parquet'
        ORDER BY dataset, run_id
    """).fetchall()
    con.close()

    runs = []
    for run_id, dataset, model_type, scale, stage, has_kd, _success in rows:
        run_dir = EXPERIMENT_ROOT / run_id
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except Exception:
            cfg = {}

        has_metrics = (run_dir / "metrics.json").exists()
        has_checkpoint = (run_dir / "best_model.pt").exists()

        runs.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "model_type": model_type,
                "scale": scale,
                "stage": stage,
                "has_kd": 1 if has_kd else 0,
                "status": "complete" if has_metrics or has_checkpoint else "running",
                "config": cfg,
                "dir": run_dir,
            }
        )
    return runs


def _scan_runs_from_filesystem() -> list[dict]:
    """Legacy filesystem scan (fallback when datalake doesn't exist)."""
    runs = []
    if not EXPERIMENT_ROOT.is_dir():
        return runs

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue

            parts = run_dir.name.split("_")
            model_type = cfg.get("model_type") or (parts[0] if parts else "")
            scale = cfg.get("scale") or (parts[1] if len(parts) > 1 else "")

            _AUX_SUFFIXES = {"kd", "nokd"}
            if cfg.get("stage"):
                stage = cfg["stage"]
            else:
                remaining = parts[2:]
                if remaining and remaining[-1] in _AUX_SUFFIXES:
                    remaining = remaining[:-1]
                stage = "_".join(remaining)

            has_kd = bool(cfg.get("auxiliaries")) or (
                "_kd" in run_dir.name and "nokd" not in run_dir.name
            )

            run_id = f"{ds_dir.name}/{run_dir.name}"
            has_metrics = (run_dir / "metrics.json").exists()
            has_checkpoint = (run_dir / "best_model.pt").exists()

            runs.append(
                {
                    "run_id": run_id,
                    "dataset": ds_dir.name,
                    "model_type": model_type,
                    "scale": scale,
                    "stage": stage,
                    "has_kd": 1 if has_kd else 0,
                    "status": "complete" if has_metrics or has_checkpoint else "running",
                    "config": cfg,
                    "dir": run_dir,
                }
            )
    return runs


def _load_eval_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json from an evaluation run directory."""
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("gat", "vgae", "fusion")


def _extract_core_metrics(metrics: dict, base: dict, target_metrics: set) -> list[dict]:
    """Extract core metrics across model keys, returning rows with base fields merged in."""
    rows = []
    for model_key in _MODEL_KEYS:
        core = metrics.get(model_key, {}).get("core", {})
        for name in target_metrics:
            val = core.get(name)
            if isinstance(val, (int, float)):
                rows.append(
                    {**base, "model": model_key, "metric_name": name, "best_value": round(val, 6)}
                )
    return rows


def export_leaderboard(output_dir: Path) -> Path:
    """Best F1/accuracy per model x dataset x scale from evaluation metrics.json files."""
    target_metrics = {"f1", "accuracy", "precision", "recall", "auc", "mcc"}
    rows = []

    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue

        base = {
            "dataset": run["dataset"],
            "model_type": run["model_type"],
            "scale": run["scale"],
            "has_kd": run["has_kd"],
        }
        rows.extend(_extract_core_metrics(metrics, base, target_metrics))

    out = output_dir / "leaderboard.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d leaderboard entries → %s", len(rows), out)
    return out


def export_runs(output_dir: Path) -> Path:
    """All runs with status."""
    rows = []
    for run in _scan_runs():
        started_at = None
        completed_at = None
        cfg_path = run["dir"] / "config.json"
        if cfg_path.exists():
            started_at = datetime.fromtimestamp(cfg_path.stat().st_mtime, tz=UTC).isoformat()
        for end_file in ("best_model.pt", "metrics.json"):
            p = run["dir"] / end_file
            if p.exists():
                completed_at = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()
                break

        rows.append(
            {
                "run_id": run["run_id"],
                "dataset": run["dataset"],
                "model_type": run["model_type"],
                "scale": run["scale"],
                "stage": run["stage"],
                "has_kd": run["has_kd"],
                "status": run["status"],
                "teacher_run": "",
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )

    out = output_dir / "runs.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d runs → %s", len(rows), out)
    return out


def export_metrics(output_dir: Path) -> Path:
    """Per-run flattened metrics from evaluation metrics.json files."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue

        rows = []
        for model_key in _MODEL_KEYS:
            model_data = metrics.get(model_key, {})
            for scenario_type in ("core", "additional"):
                section = model_data.get(scenario_type, {})
                for metric_name, value in section.items():
                    if isinstance(value, (int, float)):
                        rows.append(
                            {
                                "model": model_key,
                                "scenario": "val",
                                "metric_name": metric_name,
                                "value": value,
                            }
                        )

        fname = run["run_id"].replace("/", "_") + ".json"
        (metrics_dir / fname).write_text(json.dumps(_versioned_envelope(rows), indent=2))
        count += 1

    log.info("Exported metrics for %d runs → %s", count, metrics_dir)
    return metrics_dir


def export_metric_catalog(output_dir: Path) -> Path:
    """Export distinct metric names for dynamic dashboard dropdown."""
    all_names: set[str] = set()

    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue
        for model_key in _MODEL_KEYS:
            model_data = metrics.get(model_key, {})
            for section in ("core", "additional"):
                all_names.update(
                    k for k, v in model_data.get(section, {}).items() if isinstance(v, (int, float))
                )

    catalog = sorted(all_names)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "metric_catalog.json"
    out.write_text(json.dumps(_versioned_envelope(catalog), indent=2))
    log.info("Exported %d metric names → %s", len(catalog), out)
    return out


def export_datasets(output_dir: Path) -> Path:
    """Dataset metadata from config/datasets.yaml catalog."""
    from graphids.config.catalog import load_catalog

    catalog = load_catalog()
    rows = []
    for name, entry in catalog.items():
        rows.append(
            {
                "name": name,
                "domain": getattr(entry, "domain", "automotive"),
                "protocol": getattr(entry, "protocol", "CAN"),
                "source": getattr(entry, "source", ""),
                "description": getattr(entry, "description", ""),
            }
        )

    out = output_dir / "datasets.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d datasets → %s", len(rows), out)
    return out


def export_kd_transfer(output_dir: Path) -> Path:
    """Teacher vs student metric pairs for KD analysis.

    Scans evaluation runs, pairs large (teacher) with small+kd (student)
    on the same dataset.
    """
    target_metrics = {"f1", "accuracy", "auc"}
    rows = []

    eval_runs: dict[str, list[dict]] = {}
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        eval_runs.setdefault(run["dataset"], []).append(run)

    for ds, runs in eval_runs.items():
        teachers = [r for r in runs if r["scale"] == "large" and not r["has_kd"]]
        students = [r for r in runs if r["scale"] == "small" and r["has_kd"]]

        if not teachers or not students:
            continue

        teacher = teachers[0]
        student = students[0]
        t_metrics = _load_eval_metrics(teacher["dir"])
        s_metrics = _load_eval_metrics(student["dir"])
        if not t_metrics or not s_metrics:
            continue

        for model_key in _MODEL_KEYS:
            t_core = t_metrics.get(model_key, {}).get("core", {})
            s_core = s_metrics.get(model_key, {}).get("core", {})
            for mn in target_metrics:
                if mn in t_core and mn in s_core:
                    rows.append(
                        {
                            "student_run": student["run_id"],
                            "dataset": ds,
                            "model_type": teacher["model_type"],
                            "student_scale": "small",
                            "teacher_run": teacher["run_id"],
                            "metric_name": mn,
                            "student_value": round(s_core[mn], 6),
                            "teacher_value": round(t_core[mn], 6),
                        }
                    )

    out = output_dir / "kd_transfer.json"
    out.write_text(json.dumps(_versioned_envelope(rows), indent=2))
    log.info("Exported %d KD transfer pairs → %s", len(rows), out)
    return out


def export_training_curves(output_dir: Path) -> Path:
    """Per-run training curves from Lightning CSV logs."""
    curves_dir = output_dir / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    exported_files: list[str] = []

    if not EXPERIMENT_ROOT.is_dir():
        index_path = curves_dir / "index.json"
        index_path.write_text(json.dumps(_versioned_envelope([]), indent=2))
        return curves_dir

    for ds_dir in sorted(EXPERIMENT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            csv_logs = list(run_dir.glob("csv_logs/*/metrics.csv")) + list(
                run_dir.glob("lightning_logs/*/metrics.csv")
            )
            if not csv_logs:
                continue

            try:
                rows = []
                with open(csv_logs[0]) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epoch = row.get("epoch")
                        if epoch is None:
                            continue
                        for key, val in row.items():
                            if key == "epoch" or key == "step" or val == "":
                                continue
                            try:
                                rows.append(
                                    {
                                        "epoch": int(float(epoch)),
                                        "metric_name": key,
                                        "value": float(val),
                                    }
                                )
                            except (ValueError, TypeError):
                                continue

                if rows:
                    run_id = f"{ds_dir.name}/{run_dir.name}"
                    fname = run_id.replace("/", "_") + ".json"
                    (curves_dir / fname).write_text(json.dumps(_versioned_envelope(rows), indent=2))
                    exported_files.append(fname)
                    count += 1
            except Exception as e:
                log.warning("Failed to parse CSV log in %s: %s", run_dir, e)

    index_path = curves_dir / "index.json"
    index_path.write_text(json.dumps(_versioned_envelope(sorted(exported_files)), indent=2))

    log.info("Exported training curves for %d runs → %s", count, curves_dir)
    return curves_dir


def export_model_sizes(output_dir: Path) -> Path:
    """Export parameter counts per model_type x scale from config resolution."""
    from graphids.config import resolve
    from graphids.config.constants import NODE_FEATURE_COUNT

    sizes: list[dict] = []
    num_ids = 30
    in_ch = NODE_FEATURE_COUNT

    for model_type in ("vgae", "gat", "dqn"):
        for scale in ("large", "small"):
            try:
                cfg = resolve(model_type, scale, dataset="hcrl_sa")
                from graphids.core.models.registry import get as get_model

                entry = get_model(model_type)
                model = entry.factory(cfg, num_ids, in_ch)
                param_count = sum(p.numel() for p in model.parameters())
                sizes.append(
                    {
                        "model_type": model_type,
                        "scale": scale,
                        "param_count": param_count,
                        "param_count_M": round(param_count / 1e6, 3),
                    }
                )
                del model
            except Exception as e:
                log.warning("Could not instantiate %s/%s for param count: %s", model_type, scale, e)

    out = output_dir / "model_sizes.json"
    out.write_text(json.dumps(_versioned_envelope(sizes), indent=2))
    log.info("Exported %d model size entries → %s", len(sizes), out)
    return out


def export_loss_landscape(output_dir: Path) -> Path | None:
    """Copy loss landscape Parquet files to reports/data/ as a single merged file.

    Reads individual per-model Parquet files from datalake and merges into
    a single ``loss_landscape.parquet`` with columns:
    x, y, loss, model_type, scale, dataset, direction_seed.
    """
    landscape_dir = _DATALAKE_ROOT / "loss_landscapes"
    if not landscape_dir.is_dir():
        log.info("No loss landscape data found at %s — skipping", landscape_dir)
        return None

    parquet_files = sorted(landscape_dir.glob("*.parquet"))
    if not parquet_files:
        log.info("No loss landscape Parquet files found — skipping")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for f in parquet_files:
        tables.append(pq.read_table(f))
    merged = pa.concat_tables(tables)

    out = output_dir / "loss_landscape.parquet"
    pq.write_table(merged, out)
    log.info("Exported loss landscape (%d rows, %d files) → %s", merged.num_rows, len(tables), out)
    return out


def export_graph_samples(output_dir: Path) -> Path | None:
    """Export diverse graph samples from cached .pt files for force-directed visualization.

    Samples 3 normal + up to 2 per attack type per dataset. Produces v2 JSON schema
    with attack_type metadata, 26-D node features, 11-D edge features.
    """
    import torch

    from graphids.config.catalog import load_catalog
    from graphids.config.constants import (
        EDGE_FEATURE_NAMES,
        NODE_FEATURE_NAMES,
    )

    # Import attack type name mapping
    try:
        from graphids.core.preprocessing.adapters.can_bus import ATTACK_TYPE_NAMES
    except ImportError:
        ATTACK_TYPE_NAMES = {
            0: "normal",
            1: "dos",
            2: "fuzzing",
            3: "gear_spoofing",
            4: "rpm_spoofing",
            5: "suppress",
            6: "masquerade",
            7: "mixed",
            8: "unknown",
        }

    catalog = load_catalog()
    _data_root_str = os.environ.get("KD_GAT_DATA_ROOT")
    cache_base = Path(_data_root_str) / "cache" if _data_root_str else Path("data/cache")

    samples = []

    for ds_name in sorted(catalog.keys()):
        cache_dir = cache_base / ds_name

        # Collect all .pt files (train + test scenarios)
        pt_files = sorted(cache_dir.glob("*.pt")) if cache_dir.is_dir() else []
        if not pt_files:
            log.info("No cached graphs for %s — skipping", ds_name)
            continue

        all_graphs = []
        for pt_file in pt_files:
            try:
                graphs = torch.load(pt_file, map_location="cpu", weights_only=False)
                if hasattr(graphs, "data_list"):
                    graphs = graphs.data_list
                if not isinstance(graphs, list):
                    graphs = list(graphs)
                all_graphs.extend(graphs)
            except Exception as e:
                log.warning("Failed to load %s: %s", pt_file, e)

        if not all_graphs:
            continue

        # Partition by attack type
        normal_graphs = []
        attack_graphs: dict[int, list] = {}  # attack_type_code -> graphs
        for g in all_graphs:
            label = g.y.item() if hasattr(g, "y") else 0
            at = graph_attack_type(g, default=0 if label == 0 else -1)
            if label == 0 or at == 0:
                normal_graphs.append(g)
            else:
                attack_graphs.setdefault(at, []).append(g)

        # Sample: 3 normal + 2 per attack type
        import random

        rng = random.Random(42)
        selected = rng.sample(normal_graphs, min(3, len(normal_graphs)))
        for at_code, at_graphs in sorted(attack_graphs.items()):
            selected.extend(rng.sample(at_graphs, min(2, len(at_graphs))))

        for g in selected:
            sample = _graph_to_json(
                g, ds_name, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES, ATTACK_TYPE_NAMES
            )
            if sample:
                samples.append(sample)

    if not samples:
        log.warning("No graph samples exported — caches may be empty or missing")
        return None

    out = output_dir / "graph_samples.json"
    envelope = _versioned_envelope(samples)
    envelope["schema_version"] = "2.0.0"
    envelope["feature_names"] = {
        "node": list(NODE_FEATURE_NAMES),
        "edge": list(EDGE_FEATURE_NAMES),
    }
    out.write_text(json.dumps(envelope, indent=2))
    log.info("Exported %d graph samples → %s", len(samples), out)
    return out


def _graph_to_json(
    g,
    dataset_name: str,
    node_feature_names: list[str],
    edge_feature_names: list[str],
    attack_type_names: dict[int, str],
) -> dict | None:
    """Convert a single PyG Data object to JSON-serializable dict."""
    try:
        x = g.x.numpy()
        edge_index = g.edge_index.numpy()
        label = g.y.item() if hasattr(g, "y") else 0

        # Node data
        nodes = []
        for i in range(x.shape[0]):
            node = {"id": i, "features": [round(float(v), 6) for v in x[i]]}
            if hasattr(g, "node_y") and g.node_y is not None:
                node["node_y"] = int(g.node_y[i].item())
            nat = graph_node_attack_type(g, i)
            if nat is not None:
                node["node_attack_type"] = nat
                node["node_attack_type_name"] = attack_type_names.get(nat, "unknown")
            nodes.append(node)

        # Edge data
        edge_attr = (
            g.edge_attr.numpy() if hasattr(g, "edge_attr") and g.edge_attr is not None else None
        )
        links = []
        for j in range(edge_index.shape[1]):
            link = {"source": int(edge_index[0, j]), "target": int(edge_index[1, j])}
            if edge_attr is not None and j < edge_attr.shape[0]:
                link["edge_features"] = [round(float(v), 6) for v in edge_attr[j]]
            links.append(link)

        result = {
            "dataset": dataset_name,
            "label": label,
            "nodes": nodes,
            "links": links,
            "num_nodes": len(nodes),
            "num_edges": len(links),
        }

        # v2 metadata
        at = graph_attack_type(g, default=None)
        if at is not None:
            result["attack_type"] = at
            result["attack_type_name"] = attack_type_names.get(at, "unknown")
        if hasattr(g, "id_entropy") and g.id_entropy is not None:
            result["id_entropy"] = round(float(g.id_entropy.item()), 4)

        return result
    except Exception as e:
        log.warning("Failed to serialize graph: %s", e)
        return None


def export_data_for_reports(reports_data_dir: Path | None = None) -> None:
    """Copy datalake Parquet + artifact Parquet to reports/data/ for Quarto.

    This is the bridge between the pipeline datalake and the Quarto site.
    FileAttachment in OJS cells loads from reports/data/.
    """
    import shutil

    if reports_data_dir is None:
        reports_data_dir = Path("reports/data")
    reports_data_dir.mkdir(parents=True, exist_ok=True)

    # Core datalake files
    for name in ["metrics.parquet", "runs.parquet", "datasets.parquet"]:
        src = _DATALAKE_ROOT / name
        if src.exists():
            shutil.copy2(src, reports_data_dir / name)
            log.info("Copied %s → reports/data/", name)

    # Training curves: merge all into a single file for easy DuckDB-WASM loading
    tc_dir = _DATALAKE_ROOT / "training_curves"
    if tc_dir.is_dir():
        import pyarrow as pa
        import pyarrow.parquet as pq

        tables = []
        for f in sorted(tc_dir.glob("*.parquet")):
            tables.append(pq.read_table(f))
        if tables:
            merged = pa.concat_tables(tables)
            out = reports_data_dir / "training_curves.parquet"
            pq.write_table(merged, out)
            log.info("Merged %d training curve files → %s", len(tables), out)

    # Graph samples for force-directed visualization
    graph_src = reports_data_dir / "graph_samples.json"
    if not graph_src.exists():
        log.info("graph_samples.json already in reports/data/ or not yet exported")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def export_all(output_dir: Path, *, include_reports: bool = False) -> None:
    """Run all exports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lb = export_leaderboard(output_dir)
    runs = export_runs(output_dir)
    export_metrics(output_dir)
    ds = export_datasets(output_dir)
    kd = export_kd_transfer(output_dir)
    export_training_curves(output_dir)
    export_metric_catalog(output_dir)

    for name, path in [
        ("leaderboard", lb),
        ("runs", runs),
        ("datasets", ds),
        ("kd_transfer", kd),
    ]:
        if path.stat().st_size < 10:
            log.warning("EMPTY EXPORT: %s (%s)", name, path)

    try:
        export_model_sizes(output_dir)
    except Exception as e:
        log.warning("Export model_sizes failed (non-fatal): %s", e)

    try:
        export_loss_landscape(output_dir)
    except Exception as e:
        log.warning("Export loss_landscape failed (non-fatal): %s", e)

    try:
        export_graph_samples(output_dir)
    except Exception as e:
        log.warning("Export graph_samples failed (non-fatal): %s", e)

    # Optionally copy everything to reports/data/ for Quarto
    if include_reports:
        export_data_for_reports()

    log.info("All exports complete → %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.export",
        description="Export experiment results to static JSON/Parquet for Quarto site",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Also copy datalake Parquet data to reports/data/ for Quarto site",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    export_all(args.output_dir, include_reports=args.reports)


if __name__ == "__main__":
    main()
