"""Export experiment results to static JSON for the Quarto reports site.

Data sources:
  - MLflow: experiment tracking store (primary — metadata + metrics)
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


def _versioned_envelope(data: list | dict) -> dict:
    """Wrap export data with schema version and timestamp."""
    return {
        "schema_version": "1.0.0",
        "exported_at": datetime.now(UTC).isoformat(),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Data source: MLflow (primary) with filesystem fallback
# ---------------------------------------------------------------------------


def _scan_runs() -> list[dict]:
    """Load run metadata from MLflow, with filesystem dir paths.

    Falls back to filesystem scan if MLflow has no runs.
    """
    runs = _scan_runs_from_mlflow()
    if runs:
        return runs
    return _scan_runs_from_filesystem()


def _scan_runs_from_mlflow() -> list[dict]:
    """Read run metadata from MLflow tracking store."""
    try:
        import mlflow

        from graphids.config import MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        df = mlflow.search_runs(search_all_experiments=True)
    except Exception as e:
        log.info("MLflow scan failed (%s) — falling back to filesystem", e)
        return []

    if df.empty:
        return []

    runs = []
    for _, row in df.iterrows():
        dataset = row.get("tags.dataset", "")
        model_type = row.get("tags.model_type", "")
        scale = row.get("tags.scale", "")
        stage = row.get("tags.stage", "")
        has_kd = row.get("tags.has_kd", "False") == "True"

        run_dir = (
            EXPERIMENT_ROOT / dataset / f"{model_type}_{scale}_{stage}{'_kd' if has_kd else ''}"
        )
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
                "run_id": f"{dataset}/{run_dir.name}",
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
    """Filesystem scan (fallback when MLflow has no runs)."""
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

            model_type = cfg.get("model_type", "")
            scale = cfg.get("scale", "")
            stage = cfg.get("stage", "")
            has_kd = bool(cfg.get("auxiliaries"))

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

    # Also export as Parquet for dashboard specs
    if rows:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), output_dir / "kd_transfer.parquet")
        log.info("Exported kd_transfer.parquet (%d rows)", len(rows))

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

    # Also export as Parquet with pre-computed label for specs
    if sizes:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for s in sizes:
            s["label"] = f"{s['model_type']} ({s['scale']})"
        pq.write_table(pa.Table.from_pylist(sizes), output_dir / "model_sizes.parquet")
        log.info("Exported model_sizes.parquet (%d rows)", len(sizes))

    log.info("Exported %d model size entries → %s", len(sizes), out)
    return out


def export_pareto(output_dir: Path) -> Path | None:
    """Pre-compute Pareto frontier data for the dashboard spec.

    Computes best F1 per model from evaluation metrics, joins with model_sizes,
    then marks Pareto-optimal points. Outputs pareto.parquet.
    """
    model_sizes_parquet = output_dir / "model_sizes.parquet"
    if not model_sizes_parquet.exists():
        log.info("Missing model_sizes Parquet — skipping Pareto export")
        return None

    # Collect best F1 per model from evaluation runs
    best_f1: dict[str, float] = {}
    for run in _scan_runs():
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue
        for model_key in _MODEL_KEYS:
            f1 = metrics.get(model_key, {}).get("core", {}).get("f1")
            if f1 is not None:
                key = f"{model_key}_{run['scale']}"
                best_f1[key] = max(best_f1.get(key, 0.0), f1)

    if not best_f1:
        log.info("No F1 metrics found — skipping Pareto export")
        return None

    import pyarrow.parquet as pq

    sizes_table = pq.read_table(model_sizes_parquet)
    sizes_df = sizes_table.to_pydict()

    rows = []
    for i in range(len(sizes_df["model_type"])):
        mt = sizes_df["model_type"][i]
        sc = sizes_df["scale"][i]
        key = f"{mt}_{sc}"
        f1 = best_f1.get(key)
        if f1 is not None:
            rows.append(
                {
                    "model": mt,
                    "scale": sc,
                    "param_count_M": sizes_df["param_count_M"][i],
                    "f1": round(f1, 6),
                    "label": f"{mt}_{sc}",
                    "is_pareto": False,
                }
            )

    if not rows:
        log.info("No Pareto data computed — skipping")
        return None

    # Mark Pareto-optimal: best F1 for each param count level
    rows.sort(key=lambda r: r["param_count_M"])
    best_right = 0.0
    for r in reversed(rows):
        best_right = max(best_right, r["f1"])
        r["is_pareto"] = r["f1"] >= best_right

    import pyarrow as pa

    out = output_dir / "pareto.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out)
    n_pareto = sum(1 for r in rows if r["is_pareto"])
    log.info("Exported pareto.parquet (%d rows, %d Pareto-optimal)", len(rows), n_pareto)
    return out


def export_loss_landscape(output_dir: Path) -> Path | None:
    """Copy loss landscape Parquet files to reports/data/ as a single merged file.

    Reads individual per-model Parquet files from datalake and merges into
    a single ``loss_landscape.parquet`` with columns:
    x, y, loss, model_type, scale, dataset, direction_seed.
    """
    _data_root_str = os.environ.get("KD_GAT_DATA_ROOT")
    landscape_dir = (
        Path(_data_root_str) / "loss_landscapes" if _data_root_str else Path("data/loss_landscapes")
    )
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


def _compute_graph_stats(g) -> dict:
    """Compute structural statistics for a single PyG graph using NetworkX."""
    import networkx as nx

    edge_index = g.edge_index.numpy()
    num_nodes = g.x.size(0)
    num_edges = edge_index.shape[1]

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for j in range(num_edges):
        G.add_edge(int(edge_index[0, j]), int(edge_index[1, j]))

    G_undirected = G.to_undirected()

    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / max(len(degrees), 1)
    import numpy as np

    degree_std = float(np.std(degrees)) if degrees else 0.0
    clustering_coeff = nx.average_clustering(G_undirected)
    components = nx.number_connected_components(G_undirected)

    # Diameter (largest component only, skip if trivial)
    diameter = 0
    if components > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        if len(largest_cc) > 1:
            subgraph = G_undirected.subgraph(largest_cc)
            try:
                diameter = nx.diameter(subgraph)
            except nx.NetworkXError:
                diameter = 0

    # Degree assortativity
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(G)
    except (ValueError, nx.NetworkXError):
        degree_assortativity = 0.0

    # Betweenness centrality
    bc = nx.betweenness_centrality(G_undirected)
    bc_values = list(bc.values())
    bc_mean = sum(bc_values) / max(len(bc_values), 1)
    bc_max = max(bc_values) if bc_values else 0.0

    # Attack ratios
    attack_node_ratio = 0.0
    if hasattr(g, "node_attack_type") and g.node_attack_type is not None:
        attack_nodes = (g.node_attack_type > 0).sum().item()
        attack_node_ratio = attack_nodes / max(num_nodes, 1)
    elif hasattr(g, "node_y") and g.node_y is not None:
        attack_nodes = (g.node_y > 0).sum().item()
        attack_node_ratio = attack_nodes / max(num_nodes, 1)

    # Attack edge ratio: edges touching at least one attack node
    attack_edge_ratio = 0.0
    if attack_node_ratio > 0:
        if hasattr(g, "node_attack_type") and g.node_attack_type is not None:
            attack_mask = g.node_attack_type > 0
        elif hasattr(g, "node_y") and g.node_y is not None:
            attack_mask = g.node_y > 0
        else:
            attack_mask = None
        if attack_mask is not None:
            either_attack = (attack_mask[edge_index[0]] | attack_mask[edge_index[1]]).sum().item()
            attack_edge_ratio = either_attack / max(num_edges, 1)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 4),
        "degree_std": round(degree_std, 4),
        "clustering_coeff": round(clustering_coeff, 6),
        "num_components": components,
        "diameter": diameter,
        "degree_assortativity": round(degree_assortativity, 6),
        "betweenness_centrality_mean": round(bc_mean, 6),
        "betweenness_centrality_max": round(bc_max, 6),
        "attack_node_ratio": round(attack_node_ratio, 4),
        "attack_edge_ratio": round(attack_edge_ratio, 4),
    }


def _select_representative_normal(graphs: list, n: int, rng) -> list:
    """Select diverse normal graphs spanning density/node-count/clustering range."""
    if len(graphs) <= n:
        return list(graphs)

    import numpy as np

    # Compute lightweight metrics for selection (no NetworkX needed)
    metrics = []
    for g in graphs:
        num_nodes = g.x.size(0)
        num_edges = g.edge_index.size(1)
        density = 2 * num_edges / max(num_nodes * (num_nodes - 1), 1)
        # Approximate clustering from node features (index 22)
        clustering = float(g.x[:, 22].mean().item()) if g.x.size(1) > 22 else 0.0
        metrics.append([num_nodes, density, clustering])
    metrics = np.array(metrics, dtype=np.float64)

    # Normalize to [0,1]
    mins = metrics.min(axis=0)
    maxs = metrics.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (metrics - mins) / ranges

    # Greedy spread-based selection: pick most distant points
    selected_idx = [rng.randrange(len(graphs))]  # random seed point
    for _ in range(n - 1):
        best_dist = -1
        best_idx = 0
        for i in range(len(graphs)):
            if i in selected_idx:
                continue
            min_dist = min(
                float(np.linalg.norm(normalized[i] - normalized[j])) for j in selected_idx
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i
        selected_idx.append(best_idx)

    return [graphs[i] for i in selected_idx]


def _select_representative_attack(graphs: list, n: int) -> list:
    """Select attack graphs: highest/lowest/median attack ratio."""
    if len(graphs) <= n:
        return list(graphs)

    # Compute attack node ratios
    ratios = []
    for g in graphs:
        if hasattr(g, "node_attack_type") and g.node_attack_type is not None:
            ratio = (g.node_attack_type > 0).float().mean().item()
        elif hasattr(g, "node_y") and g.node_y is not None:
            ratio = (g.node_y > 0).float().mean().item()
        else:
            ratio = 1.0  # graph-level attack label only
        ratios.append(ratio)

    indexed = sorted(enumerate(ratios), key=lambda x: x[1])
    selected_idx = []

    # Lowest non-zero attack ratio
    for i, r in indexed:
        if r > 0:
            selected_idx.append(i)
            break
    if not selected_idx:
        selected_idx.append(indexed[0][0])

    # Highest attack ratio
    if len(indexed) > 1:
        selected_idx.append(indexed[-1][0])

    # Median attack ratio
    if n >= 3 and len(indexed) > 2:
        mid = len(indexed) // 2
        mid_idx = indexed[mid][0]
        if mid_idx not in selected_idx:
            selected_idx.append(mid_idx)

    # Fill remaining slots if n > 3
    import random

    rng = random.Random(42)
    remaining = [i for i in range(len(graphs)) if i not in selected_idx]
    while len(selected_idx) < n and remaining:
        pick = rng.choice(remaining)
        remaining.remove(pick)
        selected_idx.append(pick)

    return [graphs[i] for i in selected_idx[:n]]


def _load_dataset_graphs(ds_name: str, cache_base: Path) -> list:
    """Load cached graphs for a single dataset. Caller should free after use."""
    import torch

    cache_dir = cache_base / ds_name
    pt_files = sorted(cache_dir.glob("*.pt")) if cache_dir.is_dir() else []
    if not pt_files:
        log.info("No cached graphs for %s — skipping", ds_name)
        return []

    graphs = []
    for pt_file in pt_files:
        try:
            loaded = torch.load(pt_file, map_location="cpu", weights_only=False)
            if hasattr(loaded, "data_list"):
                loaded = loaded.data_list
            if not isinstance(loaded, list):
                loaded = list(loaded)
            graphs.extend(loaded)
        except Exception as e:
            log.warning("Failed to load %s: %s", pt_file, e)

    return graphs


def export_graph_samples(
    output_dir: Path,
    *,
    attack_type_filter: str | None = None,
    num_normal: int = 3,
    num_per_attack: int = 3,
) -> Path | None:
    """Export representative graph samples from cached .pt files for force-directed visualization.

    Uses spread-based selection for normal graphs (diversity across density/nodes/clustering)
    and attack-ratio-based selection for attack graphs (highest/lowest/median attack ratio).
    Computes per-graph structural statistics embedded in JSON output.
    """
    from graphids.config.catalog import load_catalog
    from graphids.config.constants import (
        EDGE_FEATURE_NAMES,
        NODE_FEATURE_NAMES,
    )

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

    import gc
    import random

    rng = random.Random(42)
    samples = []

    for ds_name in sorted(catalog.keys()):
        all_graphs = _load_dataset_graphs(ds_name, cache_base)
        if not all_graphs:
            continue
        # Partition by attack type
        normal_graphs = []
        attack_graphs: dict[int, list] = {}
        for g in all_graphs:
            label = g.y.item() if hasattr(g, "y") else 0
            at = graph_attack_type(g, default=0 if label == 0 else -1)
            if label == 0 or at == 0:
                normal_graphs.append(g)
            else:
                attack_graphs.setdefault(at, []).append(g)

        # Apply attack type filter if specified
        if attack_type_filter:
            filter_code = None
            for code, name in ATTACK_TYPE_NAMES.items():
                if name == attack_type_filter:
                    filter_code = code
                    break
            if filter_code is not None:
                attack_graphs = {k: v for k, v in attack_graphs.items() if k == filter_code}

        # Representative selection for normal graphs
        selected = _select_representative_normal(normal_graphs, num_normal, rng)

        # Representative selection for each attack type
        for _at_code, at_graphs in sorted(attack_graphs.items()):
            selected.extend(_select_representative_attack(at_graphs, num_per_attack))

        for g in selected:
            sample = _graph_to_json(
                g, ds_name, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES, ATTACK_TYPE_NAMES
            )
            if sample:
                # Compute and embed per-graph statistics (Task 1.2)
                try:
                    sample["stats"] = _compute_graph_stats(g)
                except Exception as e:
                    log.warning("Failed to compute stats for graph in %s: %s", ds_name, e)
                samples.append(sample)

        # Free memory before loading next dataset
        del all_graphs, normal_graphs, attack_graphs, selected
        gc.collect()
        log.info("Processed %s — %d samples so far", ds_name, len(samples))

    if not samples:
        log.warning("No graph samples exported — caches may be empty or missing")
        return None

    out = output_dir / "graph_samples.json"
    envelope = _versioned_envelope(samples)
    envelope["schema_version"] = "3.0.0"
    envelope["feature_names"] = {
        "node": list(NODE_FEATURE_NAMES),
        "edge": list(EDGE_FEATURE_NAMES),
    }
    out.write_text(json.dumps(envelope, indent=2))
    log.info("Exported %d graph samples → %s", len(samples), out)
    return out


def export_graph_statistics(output_dir: Path, *, max_per_dataset: int = 500) -> Path | None:
    """Export dataset-level graph statistics to Parquet for dashboard visualizations.

    Computes structural metrics across ALL graphs per dataset (stratified sample
    up to max_per_dataset), outputting to graph_statistics.parquet.
    """
    import random

    from graphids.config.catalog import load_catalog

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

    import gc

    rng = random.Random(42)
    rows = []

    for ds_name in sorted(catalog.keys()):
        all_graphs = _load_dataset_graphs(ds_name, cache_base)
        if not all_graphs:
            continue

        # Stratified sampling: group by label, sample proportionally
        by_label: dict[int, list] = {}
        for i, g in enumerate(all_graphs):
            label = g.y.item() if hasattr(g, "y") else 0
            by_label.setdefault(label, []).append(i)

        sampled_indices = []
        total = len(all_graphs)
        for _label, indices in by_label.items():
            n_sample = max(1, int(max_per_dataset * len(indices) / total))
            n_sample = min(n_sample, len(indices))
            sampled_indices.extend(rng.sample(indices, n_sample))

        log.info("Computing stats for %d/%d graphs in %s", len(sampled_indices), total, ds_name)

        for idx in sampled_indices:
            g = all_graphs[idx]
            label = g.y.item() if hasattr(g, "y") else 0
            at = graph_attack_type(g, default=0 if label == 0 else -1)
            at_name = ATTACK_TYPE_NAMES.get(at, "unknown") if at is not None else "unknown"

            try:
                stats = _compute_graph_stats(g)
                rows.append(
                    {
                        "dataset": ds_name,
                        "graph_idx": idx,
                        "label": label,
                        "attack_type": at if at is not None else -1,
                        "attack_type_name": at_name,
                        **stats,
                    }
                )
            except Exception as e:
                log.warning("Failed to compute stats for graph %d in %s: %s", idx, ds_name, e)

        # Free memory before loading next dataset
        del all_graphs
        gc.collect()
        log.info("Processed %s — %d stats rows so far", ds_name, len(rows))

    if not rows:
        log.warning("No graph statistics computed")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    out = output_dir / "graph_statistics.parquet"
    pq.write_table(table, out)
    log.info("Exported %d graph statistics → %s", len(rows), out)
    return out


def export_graph_layout(output_dir: Path) -> tuple[Path, Path] | None:
    """Export pre-computed graph layouts as Parquet for spec-based visualization.

    Creates graph_nodes.parquet and graph_edges.parquet with spring_layout positions.
    Replaces the D3 force-directed simulation with pre-computed coordinates.
    """
    graph_samples_path = output_dir / "graph_samples.json"
    if not graph_samples_path.exists():
        log.info("No graph_samples.json — skipping layout export")
        return None

    import networkx as nx
    import numpy as np

    samples = json.loads(graph_samples_path.read_text())["data"]
    node_rows: list[dict] = []
    edge_rows: list[dict] = []

    for i, sample in enumerate(samples):
        graph_id = f"{sample['dataset']}_{i}"
        nodes = sample["nodes"]
        links = sample["links"]

        # Build networkx graph and compute layout
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        for link in links:
            G.add_edge(link["source"], link["target"])

        pos = nx.spring_layout(G, seed=42, k=1.5 / max(np.sqrt(len(nodes)), 1))

        # Node rows
        for node in nodes:
            nid = node["id"]
            x, y = pos.get(nid, (0, 0))
            degree = G.degree(nid) if nid in G else 0
            can_id = int(node["features"][0]) if node.get("features") else 0
            node_row = {
                "graph_id": graph_id,
                "dataset": sample["dataset"],
                "node_id": nid,
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "can_id": can_id,
                "degree": degree,
                "label": sample.get("label", 0),
                "attack_type_name": sample.get("attack_type_name", "normal"),
            }
            node_rows.append(node_row)

        # Edge rows with pre-computed endpoint positions
        for link in links:
            src, tgt = link["source"], link["target"]
            sx, sy = pos.get(src, (0, 0))
            tx, ty = pos.get(tgt, (0, 0))
            edge_rows.append(
                {
                    "graph_id": graph_id,
                    "dataset": sample["dataset"],
                    "source_x": round(float(sx), 6),
                    "source_y": round(float(sy), 6),
                    "target_x": round(float(tx), 6),
                    "target_y": round(float(ty), 6),
                }
            )

    if not node_rows:
        log.warning("No graph layout data to export")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    nodes_out = output_dir / "graph_nodes.parquet"
    edges_out = output_dir / "graph_edges.parquet"
    pq.write_table(pa.Table.from_pylist(node_rows), nodes_out)
    pq.write_table(pa.Table.from_pylist(edge_rows), edges_out)
    log.info(
        "Exported graph layout: %d nodes, %d edges → %s, %s",
        len(node_rows),
        len(edge_rows),
        nodes_out,
        edges_out,
    )
    return nodes_out, edges_out


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


def export_summary(output_dir: Path) -> Path:
    """Pre-compute dashboard summary stats to avoid runtime SQL in OJS.

    Outputs summary.json with: total_runs, best_f1, best_model, kd_gap, n_datasets.
    """
    runs = _scan_runs()
    total_runs = len(runs)
    n_datasets = len({r["dataset"] for r in runs})

    # Best F1 across all evaluation metrics
    best_f1 = None
    best_model = None
    for run in runs:
        if run["stage"] != "evaluation":
            continue
        metrics = _load_eval_metrics(run["dir"])
        if not metrics:
            continue
        for model_key in _MODEL_KEYS:
            f1 = metrics.get(model_key, {}).get("core", {}).get("f1")
            if f1 is not None and (best_f1 is None or f1 > best_f1):
                best_f1 = round(f1, 6)
                best_model = model_key

    # KD gap: average (teacher_f1 - student_f1) from kd_transfer data
    kd_transfer_path = output_dir / "kd_transfer.json"
    kd_gap = None
    if kd_transfer_path.exists():
        kd_data = json.loads(kd_transfer_path.read_text()).get("data", [])
        f1_pairs = [d for d in kd_data if d.get("metric_name") == "f1"]
        if f1_pairs:
            kd_gap = round(
                sum(d["teacher_value"] - d["student_value"] for d in f1_pairs) / len(f1_pairs),
                6,
            )

    summary = {
        "total_runs": total_runs,
        "best_f1": best_f1,
        "best_model": best_model,
        "kd_gap": kd_gap,
        "n_datasets": n_datasets,
    }

    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("Exported summary stats → %s", out)
    return out


def export_data_for_reports(reports_data_dir: Path | None = None) -> None:
    """Ensure reports/data/ has all needed files for Quarto.

    All exports already write directly to reports/data/ (the output_dir).
    This function is kept for the --reports flag compatibility.
    """
    if reports_data_dir is None:
        reports_data_dir = Path("reports/data")
    reports_data_dir.mkdir(parents=True, exist_ok=True)
    log.info("Reports data directory ready: %s", reports_data_dir)


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
        export_summary(output_dir)
    except Exception as e:
        log.warning("Export summary failed (non-fatal): %s", e)

    try:
        export_model_sizes(output_dir)
    except Exception as e:
        log.warning("Export model_sizes failed (non-fatal): %s", e)

    try:
        export_pareto(output_dir)
    except Exception as e:
        log.warning("Export pareto failed (non-fatal): %s", e)

    try:
        export_loss_landscape(output_dir)
    except Exception as e:
        log.warning("Export loss_landscape failed (non-fatal): %s", e)

    try:
        export_graph_samples(output_dir)
    except Exception as e:
        log.warning("Export graph_samples failed (non-fatal): %s", e)

    try:
        export_graph_statistics(output_dir)
    except Exception as e:
        log.warning("Export graph_statistics failed (non-fatal): %s", e)

    try:
        export_graph_layout(output_dir)
    except Exception as e:
        log.warning("Export graph_layout failed (non-fatal): %s", e)

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
        help="Ensure reports/data/ is ready for Quarto site",
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Only export graph samples and statistics (skip other exports)",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default=None,
        help="Filter graph export to a specific attack type (e.g., 'dos', 'fuzzing')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples per category (default: 3 normal + 3 per attack type)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    if args.graphs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        n = args.num_samples or 3
        export_graph_samples(
            args.output_dir,
            attack_type_filter=args.attack_type,
            num_normal=n,
            num_per_attack=n,
        )
        export_graph_statistics(args.output_dir)
    else:
        export_all(args.output_dir, include_reports=args.reports)


if __name__ == "__main__":
    main()
