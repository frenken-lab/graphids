#!/usr/bin/env python3
"""Run PyGOD baselines (DOMINANT, OCGNN) on cached graph data.

Standalone script — not part of the main pipeline. Requires the `baselines`
optional dependency group: ``uv pip install -e ".[baselines]"``

Usage:
    python scripts/profiling/run_pygod_baselines.py --dataset hcrl_sa
    python scripts/profiling/run_pygod_baselines.py --dataset hcrl_sa --models dominant,ocgnn
    python scripts/profiling/run_pygod_baselines.py --dataset hcrl_sa --mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphids.config import cache_dir, data_dir, resolve

log = logging.getLogger(__name__)

SUPPORTED_MODELS = ("dominant", "ocgnn")


def _load_graphs(dataset: str):
    """Load cached graph data via the project's data loading utils."""
    cfg = resolve("vgae", "large", dataset=dataset)
    from graphids.core.training.datamodules import load_dataset

    train_data, val_data, num_ids = load_dataset(
        cfg.dataset,
        dataset_path=data_dir(cfg),
        cache_dir_path=cache_dir(cfg),
        seed=cfg.seed,
    )
    return train_data, val_data, num_ids


def _graph_label(g) -> int:
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


def _run_model(model_name: str, train_data, val_data, device: str) -> dict:
    """Train and evaluate a single PyGOD model."""
    try:
        from pygod.detector import DOMINANT, OCGNN
    except ImportError:
        log.error("pygod not installed. Install with: uv pip install -e '.[baselines]'")
        raise

    model_cls = {"dominant": DOMINANT, "ocgnn": OCGNN}[model_name]

    # Merge train + val for node-level anomaly detection
    # PyGOD expects a single PyG Data object

    all_graphs = list(train_data) + list(val_data)
    labels = np.array([_graph_label(g) for g in all_graphs])

    # Create per-graph anomaly scores by running PyGOD on each graph
    scores = np.zeros(len(all_graphs))
    t0 = time.time()

    for i, g in enumerate(all_graphs):
        g = g.clone()
        # PyGOD needs node-level labels; we use graph labels as proxy
        n_nodes = g.x.size(0)
        # Mark all nodes in attack graphs as anomalous
        g.y = torch.full((n_nodes,), _graph_label(g), dtype=torch.long)

        try:
            detector = model_cls(gpu=0 if device == "cuda" and torch.cuda.is_available() else -1)
            detector.fit(g)
            node_scores = detector.decision_score_.numpy()
            # Graph-level score = mean node anomaly score
            scores[i] = float(node_scores.mean())
        except Exception as e:
            log.warning("Failed on graph %d: %s", i, e)
            scores[i] = 0.0

        if (i + 1) % 500 == 0:
            log.info("%s: processed %d/%d graphs", model_name, i + 1, len(all_graphs))

    elapsed = time.time() - t0
    log.info("%s: finished %d graphs in %.1fs", model_name, len(all_graphs), elapsed)

    # Threshold via Youden's J
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = (
        float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(scores))
    )
    preds = (scores > best_thresh).astype(int)

    metrics = {
        "model": model_name,
        "n_graphs": len(all_graphs),
        "elapsed_seconds": round(elapsed, 1),
        "threshold": best_thresh,
        "auc_roc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run PyGOD baselines")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. hcrl_sa)")
    parser.add_argument(
        "--models",
        default="dominant,ocgnn",
        help="Comma-separated model names (default: dominant,ocgnn)",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: experimentruns/baselines/)"
    )
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in SUPPORTED_MODELS:
            log.error("Unsupported model: %s (choose from %s)", m, SUPPORTED_MODELS)
            sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir else _ROOT / "experimentruns" / "baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset: %s", args.dataset)
    train_data, val_data, num_ids = _load_graphs(args.dataset)
    log.info("Loaded %d train + %d val graphs, %d CAN IDs", len(train_data), len(val_data), num_ids)

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow

            from graphids.config import MLFLOW_TRACKING_URI

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("kd-gat-baselines")
            mlflow_run = mlflow.start_run(
                run_name=f"pygod-baselines-{args.dataset}",
                tags={"type": "baseline", "framework": "pygod", "dataset": args.dataset},
            )
        except ImportError:
            log.warning("mlflow not installed, skipping MLflow logging")

    all_results = {}
    for model_name in models:
        log.info("Running %s on %s...", model_name, args.dataset)
        metrics = _run_model(model_name, train_data, val_data, args.device)

        out_path = output_dir / f"pygod_{model_name}_{args.dataset}.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        log.info("Saved results to %s", out_path)
        log.info(
            "  AUC-ROC: %.4f  F1: %.4f  Precision: %.4f  Recall: %.4f",
            metrics["auc_roc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

        all_results[model_name] = metrics

        if mlflow_run is not None:
            import mlflow

            mlflow.log_metrics(
                {f"{model_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            )

    if mlflow_run is not None:
        import mlflow

        mlflow.end_run()

    # Print summary table
    print("\n" + "=" * 60)
    print(f"PyGOD Baselines — {args.dataset}")
    print("=" * 60)
    print(f"{'Model':<12} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 60)
    for name, m in all_results.items():
        print(
            f"{name:<12} {m['auc_roc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
