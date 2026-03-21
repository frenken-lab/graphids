#!/usr/bin/env python3
"""Export paper-ready data from KD-GAT evaluation artifacts to ESS.

Output: $KD_GAT_LAKE_ROOT/exports/paper/{csv,figures,metadata}/

Usage:
    python scripts/data/export_paper_data.py --dataset hcrl_sa
    python scripts/data/export_paper_data.py --dataset hcrl_sa --figures-only
    python scripts/data/export_paper_data.py --dataset hcrl_sa --tables-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import structlog
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graphids.core.preprocessing import ATTACK_TYPE_NAMES
import os
from pathlib import Path

log = structlog.get_logger()


def lake_run_dir(lake_root, dataset, model_type, scale, stage, seed=42, production=False):
    """Reconstruct run directory path for existing runs on disk."""
    tier = "production" if production else f"dev/{os.environ.get('USER', 'unknown')}"
    return Path(lake_root) / tier / dataset / f"{model_type}_{scale}_{stage}" / f"seed_{seed}"


SAMPLE_FRAC = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        import subprocess

        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _atk_name(code: int) -> str:
    return ATTACK_TYPE_NAMES.get(code, f"unknown_{code}").replace("_", " ").title()


def _sample_idx(n: int, frac: float = SAMPLE_FRAC, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = max(100, int(n * frac))
    idx = rng.choice(n, size=min(k, n), replace=False)
    idx.sort()
    return idx


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s (%d rows)", path, len(rows))


def _write_json(data, path: Path, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(data, separators=(",", ":")))
    else:
        path.write_text(json.dumps(data, indent=2))
    n = len(data) if isinstance(data, list) else "object"
    log.info("Wrote %s (%s)", path, n)


# ---------------------------------------------------------------------------
# Table exports
# ---------------------------------------------------------------------------


def export_tables(metrics: dict, csv_dir: Path) -> None:
    """Export all CSV tables from metrics.json."""
    # Main results
    rows = []
    for model in ("gat", "vgae", "fusion"):
        c = metrics.get(model, {}).get("core", {})
        if c:
            rows.append(
                {
                    k: c.get(k, 0)
                    for k in ["accuracy", "precision", "recall", "f1", "auc", "specificity", "mcc"]
                }
                | {"model": model.upper()}
            )
    if rows:
        _write_csv(rows, csv_dir / "main_results.csv")

    # Test results
    rows = []
    for model, scenarios in metrics.get("test", {}).items():
        for scenario, m in scenarios.items():
            c = m["core"]
            rows.append(
                {
                    "model": model.upper(),
                    "scenario": scenario,
                    "accuracy": c.get("accuracy", 0),
                    "f1": c.get("f1", 0),
                    "precision": c.get("precision", 0),
                    "recall": c.get("recall", 0),
                }
            )
    if rows:
        _write_csv(rows, csv_dir / "test_results.csv")

    # VGAE threshold
    vgae = metrics.get("vgae", {}).get("core", {})
    if vgae:
        _write_csv(
            [
                {"metric": k, "value": vgae.get(k, 0)}
                for k in ("optimal_threshold", "youden_j", "f1", "auc")
            ],
            csv_dir / "vgae_threshold.csv",
        )


# ---------------------------------------------------------------------------
# Figure exports
# ---------------------------------------------------------------------------


def export_umap(npz, fig_dir: Path) -> None:
    if "gat_emb" not in npz:
        log.warning("Skipping umap — no gat_emb")
        return
    emb, labels, at = (
        npz["gat_emb"],
        npz.get("gat_labels", np.zeros(len(npz["gat_emb"]))),
        npz.get("gat_attack_types", np.zeros(len(npz["gat_emb"]))),
    )
    idx = _sample_idx(len(emb))

    try:
        from umap import UMAP

        coords = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(
            emb[idx]
        )
    except ImportError:
        log.error("umap-learn not installed")
        return

    records = [
        {
            "x": round(float(coords[i, 0]), 3),
            "y": round(float(coords[i, 1]), 3),
            "label": int(labels[idx[i]]),
            "attack_type": _atk_name(int(at[idx[i]])),
            "confidence": 0.0,
            "graph_idx": int(idx[i]),
        }
        for i in range(len(coords))
    ]
    _write_json(records, fig_dir / "umap" / "data.json")


def export_reconstruction(npz, fig_dir: Path) -> None:
    """Export plot-ready reconstruction data: {kde, heatmap, roc} panels."""
    required = ["vgae_errors", "vgae_labels"]
    if not all(k in npz for k in required):
        log.warning("Skipping reconstruction — missing keys")
        return
    errors, labels = npz["vgae_errors"], npz["vgae_labels"]
    has_at = "vgae_attack_types" in npz
    has_comp = all(f"vgae_error_{c}" in npz for c in ("recon", "canid", "nbr", "kl"))

    if not has_comp:
        log.warning("Skipping reconstruction — no per-component errors")
        return

    comp_names = [("recon", "Node Recon"), ("canid", "CAN ID"), ("nbr", "Neighbor"), ("kl", "KL")]
    idx = _sample_idx(len(errors))

    # Panel 1: KDE — flatten to {value, component, class}
    kde = []
    for i in idx:
        cls = "Normal" if labels[i] == 0 else "Attack"
        for key, label in comp_names:
            kde.append(
                {
                    "value": round(float(npz[f"vgae_error_{key}"][i]), 6),
                    "component": label,
                    "class": cls,
                }
            )

    # Panel 2: Heatmap — sort by composite, sample 200, flatten to {row, component, value}
    order = np.argsort(errors)
    step = max(1, len(order) // 200)
    heatmap = []
    for rank, i in enumerate(order[::step]):
        for key, label in comp_names:
            heatmap.append(
                {
                    "row": str(rank),
                    "component": label,
                    "value": round(float(npz[f"vgae_error_{key}"][i]), 6),
                }
            )

    # Panel 3: ROC — pre-compute per component
    from sklearn.metrics import roc_curve as _roc_curve

    roc = []
    for key, label in comp_names:
        scores = npz[f"vgae_error_{key}"]
        if len(set(labels)) < 2:
            continue
        fpr, tpr, _ = _roc_curve(labels, scores)
        # Sample ~100 points
        step = max(1, len(fpr) // 100)
        for j in range(0, len(fpr), step):
            roc.append(
                {"fpr": round(float(fpr[j]), 4), "tpr": round(float(tpr[j]), 4), "component": label}
            )
        roc.append({"fpr": 1.0, "tpr": 1.0, "component": label})

    _write_json(
        {"kde": kde, "heatmap": heatmap, "roc": roc}, fig_dir / "reconstruction" / "data.json"
    )


def export_fusion(policy_path: Path, npz, fig_dir: Path) -> None:
    policy = json.loads(policy_path.read_text())
    alphas, labels = policy.get("alphas", []), policy.get("labels", [])
    at = npz.get("vgae_attack_types") if npz else None

    records = [
        {
            "alpha": round(float(alphas[i]), 4),
            "label": int(labels[i]),
            "attack_type": _atk_name(int(at[i]))
            if at is not None and i < len(at)
            else ("Normal" if labels[i] == 0 else "Attack"),
        }
        for i in range(len(alphas))
    ]
    _write_json(records, fig_dir / "fusion" / "data.json")


def export_cka(cka_path: Path, fig_dir: Path) -> None:
    out = fig_dir / "cka" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cka_path.read_text())
    log.info("Wrote %s", out)


def export_attention(attn_path: Path, fig_dir: Path) -> None:
    """Export attention with pre-computed node positions (spring layout)."""
    import networkx as nx

    data = np.load(attn_path, allow_pickle=True)
    n = int(data.get("n_samples", 0))
    if n == 0:
        return

    samples = []
    for i in range(min(n, 10)):
        pfx = f"sample_{i}"
        edge_index = data[f"{pfx}_edge_index"]
        nf = data[f"{pfx}_node_features"]
        layer_keys = sorted(
            k for k in data.files if k.startswith(f"{pfx}_layer_") and k.endswith("_alpha")
        )

        # Build networkx graph for layout
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nf)))
        for e in range(edge_index.shape[1]):
            G.add_edge(int(edge_index[0, e]), int(edge_index[1, e]))
        pos = nx.spring_layout(G, seed=42, scale=150)

        nodes = [
            {"id": j, "can_id": hex(int(nf[j])), "x": round(pos[j][0], 1), "y": round(pos[j][1], 1)}
            for j in range(len(nf))
        ]

        edges = []
        for e in range(edge_index.shape[1]):
            rec = {"source": int(edge_index[0, e]), "target": int(edge_index[1, e])}
            for lk in layer_keys:
                ln = int(lk.split("_layer_")[1].split("_alpha")[0])
                attn = data[lk]
                if e < attn.shape[0]:
                    val = float(attn[e].mean()) if attn.ndim > 1 else float(attn[e])
                    rec[f"layer_{ln}_attention"] = round(val, 4)
            edges.append(rec)

        samples.append(
            {
                "graph_idx": int(data[f"{pfx}_graph_idx"]),
                "label": int(data[f"{pfx}_label"]),
                "attack_type": "Normal" if int(data[f"{pfx}_label"]) == 0 else "Attack",
                "nodes": nodes,
                "edges": edges,
            }
        )
    _write_json(samples, fig_dir / "attention" / "data.json", compact=False)


# ---------------------------------------------------------------------------
# Manifest + Provenance
# ---------------------------------------------------------------------------


def write_manifest(paper_dir: Path) -> None:
    entries = [
        {
            "name": str(p.relative_to(paper_dir)),
            "size_bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }
        for p in sorted(paper_dir.rglob("*"))
        if p.is_file() and p.name not in ("_manifest.json", "_provenance.json")
    ]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "artifacts": entries,
    }
    (paper_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest (%d artifacts)", len(entries))


def write_provenance(paper_dir: Path, dataset: str, eval_dir: Path) -> None:
    prov = {
        "exported_at": datetime.now(UTC).isoformat(),
        "kd_gat_commit": _git_sha(),
        "dataset": dataset,
        "eval_dir": str(eval_dir),
    }
    (paper_dir / "_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info("Wrote provenance")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Export paper data from KD-GAT eval artifacts")
    p.add_argument("--dataset", default="hcrl_sa")
    p.add_argument("--scale", default="large")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args()

    lake_root = os.environ.get("KD_GAT_LAKE_ROOT")
    if not lake_root:
        log.error("KD_GAT_LAKE_ROOT not set")
        sys.exit(1)

    paper_dir = Path(lake_root) / "exports" / "paper"
    csv_dir, fig_dir = paper_dir / "csv", paper_dir / "figures"

    # Read from ESS production — the data lake is the single source of truth
    eval_dir = lake_run_dir(
        lake_root, args.dataset, "gat", args.scale, "evaluation", seed=args.seed, production=True
    )
    if not eval_dir.exists():
        log.error("No eval artifacts on ESS: %s", eval_dir)
        log.error("Run evaluation first, then migrate to ESS.")
        sys.exit(1)

    # Load artifacts
    metrics_path = eval_dir / "metrics.json"
    npz_path = eval_dir / "embeddings.npz"
    npz = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None

    if not metrics_path.exists() and npz is None:
        log.error("No metrics.json or embeddings.npz in %s", eval_dir)
        sys.exit(1)
    log.info("Reading from %s", eval_dir)

    # Report missing artifacts that need re-run
    missing = []
    if "vgae_error_recon" not in (npz or {}):
        missing.append("per-component VGAE errors (re-run eval with updated evaluation.py)")
    if not (eval_dir / "cka_matrix.json").exists():
        missing.append("CKA matrix (run eval on KD variant: --auxiliaries kd_standard)")
    if "vgae_attack_types" not in (npz or {}):
        missing.append("attack_type arrays (re-run eval with updated evaluation.py)")
    if missing:
        log.warning("Missing artifacts (figures will have partial data):")
        for m in missing:
            log.warning("  - %s", m)

    if not args.figures_only and metrics_path.exists():
        export_tables(json.loads(metrics_path.read_text()), csv_dir)

    if not args.tables_only:
        if npz is not None:
            export_umap(npz, fig_dir)
            export_reconstruction(npz, fig_dir)
        policy_path = eval_dir / "dqn_policy.json"
        if policy_path.exists():
            export_fusion(policy_path, npz, fig_dir)
        cka_path = eval_dir / "cka_matrix.json"
        if cka_path.exists():
            export_cka(cka_path, fig_dir)
        attn_path = eval_dir / "attention_weights.npz"
        if attn_path.exists():
            export_attention(attn_path, fig_dir)

    write_provenance(paper_dir, args.dataset, eval_dir)
    write_manifest(paper_dir)
    log.info("Export complete")


if __name__ == "__main__":
    from graphids.logging import configure_logging
    configure_logging()
    main()
