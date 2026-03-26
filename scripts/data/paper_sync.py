#!/usr/bin/env python3
"""Paper data sync: push eval artifacts to ESS, pull to paper repo.

Subcommands:
    push      Transform eval artifacts -> paper-ready format on ESS
    pull      Copy from ESS -> paper repo + validate
    validate  Schema validation only (no copy)
    status    Show ESS artifacts, checksums, freshness
    diff      Compare ESS manifest with local files

Usage:
    python scripts/data/paper_sync.py push --dataset hcrl_sa
    python scripts/data/paper_sync.py push --dataset hcrl_sa --only umap
    python ~/KD-GAT/scripts/data/paper_sync.py pull --schema data/schemas.yaml
    python ~/KD-GAT/scripts/data/paper_sync.py validate --schema data/schemas.yaml
    python scripts/data/paper_sync.py status
    python scripts/data/paper_sync.py diff --schema ~/kd-gat-paper/data/schemas.yaml
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESS_PAPER_REL = "exports/paper"
SAMPLE_FRAC = 0.10
ARTIFACT_KINDS = ("tables", "umap", "reconstruction", "fusion", "cka", "attention", "metadata")

log = logging.getLogger("paper_sync")

app = typer.Typer(help="Paper data sync: push/pull/validate between KD-GAT and paper repo via ESS.")

# ---------------------------------------------------------------------------
# Shared utilities
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


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %s (%d rows)", path.name, len(rows))


def _write_json(data, path: Path, compact: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(data, separators=(",", ":")))
    else:
        path.write_text(json.dumps(data, indent=2))
    n = len(data) if isinstance(data, list) else "object"
    log.info("Wrote %s (%s)", path.name, n)


def lake_run_dir(
    lake_root, dataset, model_type, scale, stage, seed=42, production=False,
    identity_hash="",
) -> Path:
    """Reconstruct run directory path for existing runs on disk.

    identity_hash: 8-char hash from identity_keys (new path format), or "" for legacy paths.
    """
    tier = "production" if production else f"dev/{os.environ.get('USER', 'unknown')}"
    suffix = f"_{identity_hash}" if identity_hash else ""
    return Path(lake_root) / tier / dataset / f"{model_type}_{scale}_{stage}{suffix}" / f"seed_{seed}"


def _resolve_ess(ess_root: Optional[Path]) -> Path:
    root = ess_root or os.environ.get("KD_GAT_LAKE_ROOT")
    if not root:
        log.error("KD_GAT_LAKE_ROOT not set and --ess-root not provided")
        raise SystemExit(1)
    return Path(root) / ESS_PAPER_REL


# ---------------------------------------------------------------------------
# Push-side helpers (lazy graphids/numpy imports)
# ---------------------------------------------------------------------------

_ATTACK_NAMES: dict[int, str] | None = None
_ATTACK_CODES: dict[int, str] | None = None


def _load_attack_maps():
    global _ATTACK_NAMES, _ATTACK_CODES
    if _ATTACK_NAMES is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from graphids.core.preprocessing import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES

        _ATTACK_NAMES = ATTACK_TYPE_NAMES
        _ATTACK_CODES = ATTACK_TYPE_CODES


def _atk_name(code: int) -> str:
    _load_attack_maps()
    return _ATTACK_NAMES.get(code, f"unknown_{code}").replace("_", " ").title()


def _sample_idx(n: int, frac: float = SAMPLE_FRAC, seed: int = 42):
    import numpy as np

    rng = np.random.default_rng(seed)
    k = max(100, int(n * frac))
    idx = rng.choice(n, size=min(k, n), replace=False)
    idx.sort()
    return idx


# ---------------------------------------------------------------------------
# Single export function
# ---------------------------------------------------------------------------


def export(kind: str, eval_dir: Path, output_dir: Path) -> bool:
    """Export a single artifact kind from eval_dir to paper-ready format.

    Returns True if exported, False if source data missing.
    """
    import numpy as np

    if kind == "tables":
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            return False
        metrics = json.loads(metrics_path.read_text())
        csv_dir = output_dir / "csv"

        # Main results
        rows = []
        for model in ("gat", "vgae", "fusion"):
            c = metrics.get(model, {}).get("core", {})
            if c:
                rows.append(
                    {k: c.get(k, 0) for k in ["accuracy", "precision", "recall", "f1", "auc", "specificity", "mcc"]}
                    | {"model": model.upper()}
                )
        if rows:
            _write_csv(rows, csv_dir / "main_results.csv")

        # Test results
        rows = []
        for model, scenarios in metrics.get("test", {}).items():
            for scenario, m in scenarios.items():
                c = m["core"]
                rows.append({
                    "model": model.upper(), "scenario": scenario,
                    "accuracy": c.get("accuracy", 0), "f1": c.get("f1", 0),
                    "precision": c.get("precision", 0), "recall": c.get("recall", 0),
                })
        if rows:
            _write_csv(rows, csv_dir / "test_results.csv")

        # VGAE threshold
        vgae = metrics.get("vgae", {}).get("core", {})
        if vgae:
            _write_csv(
                [{"metric": k, "value": vgae.get(k, 0)} for k in ("optimal_threshold", "youden_j", "f1", "auc")],
                csv_dir / "vgae_threshold.csv",
            )
        return True

    elif kind == "umap":
        npz_path = eval_dir / "embeddings.npz"
        if not npz_path.exists() or "gat_emb" not in (npz := np.load(npz_path, allow_pickle=True)):
            return False
        emb = npz["gat_emb"]
        labels = npz.get("gat_labels", np.zeros(len(emb)))
        at = npz.get("gat_attack_types", np.zeros(len(emb)))
        idx = _sample_idx(len(emb))

        try:
            from umap import UMAP
            coords = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(emb[idx])
        except ImportError:
            log.error("umap-learn not installed")
            return False

        records = [
            {"x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3),
             "label": int(labels[idx[i]]), "attack_type": _atk_name(int(at[idx[i]])),
             "confidence": 0.0, "graph_idx": int(idx[i])}
            for i in range(len(coords))
        ]
        _write_json(records, output_dir / "figures" / "umap" / "data.json")
        return True

    elif kind == "reconstruction":
        npz_path = eval_dir / "embeddings.npz"
        if not npz_path.exists():
            return False
        npz = np.load(npz_path, allow_pickle=True)
        if not all(k in npz for k in ["vgae_errors", "vgae_labels"]):
            return False
        if not all(f"vgae_error_{c}" in npz for c in ("recon", "canid", "nbr", "kl")):
            return False

        errors, labels = npz["vgae_errors"], npz["vgae_labels"]
        comp_names = [("recon", "Node Recon"), ("canid", "CAN ID"), ("nbr", "Neighbor"), ("kl", "KL")]
        idx = _sample_idx(len(errors))

        # KDE panel
        kde = []
        for i in idx:
            cls = "Normal" if labels[i] == 0 else "Attack"
            for key, label in comp_names:
                kde.append({"value": round(float(npz[f"vgae_error_{key}"][i]), 6), "component": label, "class": cls})

        # Heatmap panel
        order = np.argsort(errors)
        step = max(1, len(order) // 200)
        heatmap = []
        for rank, i in enumerate(order[::step]):
            for key, label in comp_names:
                heatmap.append({"row": str(rank), "component": label, "value": round(float(npz[f"vgae_error_{key}"][i]), 6)})

        # ROC panel
        from sklearn.metrics import roc_curve as _roc_curve
        roc = []
        for key, label in comp_names:
            scores = npz[f"vgae_error_{key}"]
            if len(set(labels)) < 2:
                continue
            fpr, tpr, _ = _roc_curve(labels, scores)
            step = max(1, len(fpr) // 100)
            for j in range(0, len(fpr), step):
                roc.append({"fpr": round(float(fpr[j]), 4), "tpr": round(float(tpr[j]), 4), "component": label})
            roc.append({"fpr": 1.0, "tpr": 1.0, "component": label})

        _write_json({"kde": kde, "heatmap": heatmap, "roc": roc}, output_dir / "figures" / "reconstruction" / "data.json")
        return True

    elif kind == "fusion":
        policy_path = eval_dir / "dqn_policy.json"
        if not policy_path.exists():
            return False
        policy = json.loads(policy_path.read_text())
        alphas, labels = policy.get("alphas", []), policy.get("labels", [])

        npz_path = eval_dir / "embeddings.npz"
        npz = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None
        at = npz.get("vgae_attack_types") if npz is not None else None

        records = [
            {"alpha": round(float(alphas[i]), 4), "label": int(labels[i]),
             "attack_type": _atk_name(int(at[i])) if at is not None and i < len(at) else ("Normal" if labels[i] == 0 else "Attack")}
            for i in range(len(alphas))
        ]
        _write_json(records, output_dir / "figures" / "fusion" / "data.json")
        return True

    elif kind == "cka":
        cka_path = eval_dir / "cka_matrix.json"
        if not cka_path.exists():
            return False
        out = output_dir / "figures" / "cka" / "data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(cka_path.read_text())
        log.info("Wrote %s", out.name)
        return True

    elif kind == "attention":
        import networkx as nx

        attn_path = eval_dir / "attention_weights.npz"
        if not attn_path.exists():
            return False
        data = np.load(attn_path, allow_pickle=True)
        n = int(data.get("n_samples", 0))
        if n == 0:
            return False

        samples = []
        for i in range(min(n, 10)):
            pfx = f"sample_{i}"
            edge_index = data[f"{pfx}_edge_index"]
            nf = data[f"{pfx}_node_features"]
            layer_keys = sorted(k for k in data.files if k.startswith(f"{pfx}_layer_") and k.endswith("_alpha"))

            G = nx.DiGraph()
            G.add_nodes_from(range(len(nf)))
            for e in range(edge_index.shape[1]):
                G.add_edge(int(edge_index[0, e]), int(edge_index[1, e]))
            pos = nx.spring_layout(G, seed=42, scale=150)

            nodes = [{"id": j, "can_id": hex(int(nf[j])), "x": round(pos[j][0], 1), "y": round(pos[j][1], 1)} for j in range(len(nf))]
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

            samples.append({
                "graph_idx": int(data[f"{pfx}_graph_idx"]), "label": int(data[f"{pfx}_label"]),
                "attack_type": "Normal" if int(data[f"{pfx}_label"]) == 0 else "Attack",
                "nodes": nodes, "edges": edges,
            })
        _write_json(samples, output_dir / "figures" / "attention" / "data.json", compact=False)
        return True

    elif kind == "metadata":
        _load_attack_maps()
        meta_dir = output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        mapping = {str(code): name for code, name in _ATTACK_CODES.items()}
        mapping["names"] = {str(code): name for code, name in _ATTACK_NAMES.items()}
        _write_json(mapping, meta_dir / "attack_type_mapping.json")
        return True

    else:
        log.error("Unknown artifact kind: %s", kind)
        return False


# ---------------------------------------------------------------------------
# Manifest + provenance
# ---------------------------------------------------------------------------


def write_manifest(paper_dir: Path) -> None:
    entries = [
        {"name": str(p.relative_to(paper_dir)), "size_bytes": p.stat().st_size, "sha256": _sha256(p)}
        for p in sorted(paper_dir.rglob("*"))
        if p.is_file() and p.name not in ("_manifest.json", "_provenance.json")
    ]
    manifest = {"created_at": datetime.now(UTC).isoformat(), "git_sha": _git_sha(), "artifacts": entries}
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
# Pull + validate
# ---------------------------------------------------------------------------


def load_schema(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        log.error("pyyaml required for pull/validate: pip install pyyaml")
        raise SystemExit(1)
    return yaml.safe_load(path.read_text())


def pull(ess_paper_dir: Path, schema: dict, repo_root: Path) -> list[str]:
    """Copy files from ESS to repo per file_map. Returns errors."""
    if not ess_paper_dir.exists():
        return [f"ESS paper dir not found: {ess_paper_dir}"]

    errors, copied = [], 0
    for ess_rel, repo_rel in schema["file_map"].items():
        src, dst = ess_paper_dir / ess_rel, repo_root / repo_rel
        if not src.exists():
            errors.append(f"Missing on ESS: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    manifest_path = ess_paper_dir / "_manifest.json"
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text()).get("artifacts", []):
            src = ess_paper_dir / entry["name"]
            if src.exists() and _sha256(src) != entry["sha256"]:
                errors.append(f"Checksum mismatch: {entry['name']}")
    else:
        errors.append("No _manifest.json — checksums not verified")

    log.info("Copied %d files from ESS", copied)
    return errors


def validate(schema: dict, repo_root: Path) -> list[str]:
    """Validate local data against schema contracts. Returns errors."""
    errors = []

    for name, spec in schema.get("csv", {}).items():
        path = repo_root / "data" / "csv" / name
        if not path.exists():
            errors.append(f"Missing: data/csv/{name}")
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            errors.append(f"{name}: empty")
            continue
        for col in spec["columns"]:
            if col not in rows[0]:
                errors.append(f"{name}: missing column '{col}'")
        if len(rows) < spec.get("min_rows", 0):
            errors.append(f"{name}: {len(rows)} rows < {spec['min_rows']}")

    for name, spec in schema.get("json", {}).items():
        path = repo_root / "interactive" / "src" / name / "data.json"
        if not path.exists():
            errors.append(f"Missing: interactive/src/{name}/data.json")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{name}: invalid JSON — {e}")
            continue

        if spec["type"] == "array":
            if not isinstance(data, list):
                errors.append(f"{name}: expected array")
            elif len(data) < spec.get("min_items", 0):
                errors.append(f"{name}: {len(data)} items < {spec['min_items']}")
            elif data and any(k not in data[0] for k in spec.get("item_keys", [])):
                missing = [k for k in spec["item_keys"] if k not in data[0]]
                errors.append(f"{name}: missing keys {missing}")
        elif spec["type"] == "object":
            if not isinstance(data, dict):
                errors.append(f"{name}: expected object")
            else:
                missing = [k for k in spec.get("required_keys", []) if k not in data]
                if missing:
                    errors.append(f"{name}: missing keys {missing}")

    return errors


# ---------------------------------------------------------------------------
# Status + diff
# ---------------------------------------------------------------------------


def show_status(ess_paper_dir: Path) -> None:
    if not ess_paper_dir.exists():
        log.error("ESS paper dir not found: %s", ess_paper_dir)
        raise SystemExit(1)

    manifest_path = ess_paper_dir / "_manifest.json"
    if not manifest_path.exists():
        print(f"No _manifest.json at {ess_paper_dir}")
        for p in sorted(ess_paper_dir.rglob("*")):
            if p.is_file():
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
                print(f"  {p.relative_to(ess_paper_dir):<50s} {p.stat().st_size:>10,}B  {mtime:%Y-%m-%d %H:%M}")
        return

    manifest = json.loads(manifest_path.read_text())
    print(f"Manifest created: {manifest['created_at']}")
    print(f"Source commit:    {manifest.get('git_sha', 'unknown')}")
    print(f"Artifacts:        {len(manifest['artifacts'])}")
    print()
    for entry in manifest["artifacts"]:
        exists = (ess_paper_dir / entry["name"]).exists()
        marker = "OK" if exists else "MISSING"
        print(f"  [{marker:>7s}] {entry['name']:<50s} {entry['size_bytes']:>10,}B  sha256:{entry['sha256'][:12]}...")


def show_diff(ess_paper_dir: Path, schema: dict, repo_root: Path) -> None:
    manifest_path = ess_paper_dir / "_manifest.json"
    if not manifest_path.exists():
        log.error("No _manifest.json at %s — run push first", ess_paper_dir)
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text())
    ess_by_name = {e["name"]: e for e in manifest["artifacts"]}

    for ess_rel, repo_rel in schema.get("file_map", {}).items():
        local = repo_root / repo_rel
        ess_entry = ess_by_name.get(ess_rel)

        if not ess_entry:
            print(f"  [NOT ON ESS]  {ess_rel}")
        elif not local.exists():
            print(f"  [LOCAL MISS]  {repo_rel}")
        else:
            local_sha = _sha256(local)
            if local_sha == ess_entry["sha256"]:
                print(f"  [UP TO DATE]  {repo_rel}")
            else:
                print(f"  [  CHANGED ]  {repo_rel}  (local:{local_sha[:12]}.. ess:{ess_entry['sha256'][:12]}..)")


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------


@app.command()
def push(
    dataset: str = typer.Option("hcrl_sa", help="Dataset name"),
    scale: str = typer.Option("large", help="Model scale"),
    seed: int = typer.Option(42, help="Random seed"),
    only: Optional[str] = typer.Option(None, help=f"Export only one kind: {', '.join(ARTIFACT_KINDS)}"),
    figures_only: bool = typer.Option(False, "--figures-only", help="Skip table export"),
    tables_only: bool = typer.Option(False, "--tables-only", help="Skip figure export"),
    ess_root: Optional[Path] = typer.Option(None, "--ess-root", help="Override KD_GAT_LAKE_ROOT"),
):
    """Transform eval artifacts into paper-ready format on ESS."""
    import structlog
    structlog.configure(processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ])

    lake_root = ess_root or os.environ.get("KD_GAT_LAKE_ROOT")
    if not lake_root:
        log.error("KD_GAT_LAKE_ROOT not set")
        raise SystemExit(1)

    output_dir = Path(lake_root) / ESS_PAPER_REL
    eval_dir = lake_run_dir(lake_root, dataset, "gat", scale, "evaluation", seed=seed, production=True)

    if not eval_dir.exists():
        log.error("No eval artifacts: %s", eval_dir)
        raise SystemExit(1)
    log.info("Reading from %s", eval_dir)

    if only:
        if only not in ARTIFACT_KINDS:
            log.error("Unknown kind '%s'. Valid: %s", only, ", ".join(ARTIFACT_KINDS))
            raise SystemExit(1)
        ok = export(only, eval_dir, output_dir)
        if not ok:
            log.warning("Source data missing for '%s'", only)
    else:
        kinds = list(ARTIFACT_KINDS)
        if figures_only:
            kinds = [k for k in kinds if k != "tables"]
        elif tables_only:
            kinds = ["tables", "metadata"]
        for kind in kinds:
            ok = export(kind, eval_dir, output_dir)
            if not ok:
                log.warning("Skipped '%s' — source data missing", kind)

    write_provenance(output_dir, dataset, eval_dir)
    write_manifest(output_dir)
    log.info("Push complete")


@app.command("pull")
def pull_cmd(
    schema: Path = typer.Option(..., "--schema", help="Path to schemas.yaml"),
    ess_root: Optional[Path] = typer.Option(None, "--ess-root", help="Override KD_GAT_LAKE_ROOT"),
    repo_root: Optional[Path] = typer.Option(None, "--repo-root", help="Paper repo root (default: cwd)"),
):
    """Copy from ESS to paper repo + validate."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    s = load_schema(schema)
    root = repo_root or Path.cwd()
    ess = _resolve_ess(ess_root)

    errors = pull(ess, s, root)
    errors.extend(validate(s, root))

    if errors:
        log.error("%d error(s):", len(errors))
        for e in errors:
            log.error("  - %s", e)
        raise SystemExit(1)
    log.info("Pull + validate complete")


@app.command("validate")
def validate_cmd(
    schema: Path = typer.Option(..., "--schema", help="Path to schemas.yaml"),
    repo_root: Optional[Path] = typer.Option(None, "--repo-root", help="Paper repo root (default: cwd)"),
):
    """Validate local files against schemas.yaml (no copy)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    s = load_schema(schema)
    root = repo_root or Path.cwd()

    errors = validate(s, root)
    if errors:
        log.error("%d error(s):", len(errors))
        for e in errors:
            log.error("  - %s", e)
        raise SystemExit(1)
    log.info("All validations passed")


@app.command()
def status(
    ess_root: Optional[Path] = typer.Option(None, "--ess-root", help="Override KD_GAT_LAKE_ROOT"),
):
    """Show ESS artifacts, checksums, freshness."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    show_status(_resolve_ess(ess_root))


@app.command()
def diff(
    schema: Path = typer.Option(..., "--schema", help="Path to schemas.yaml"),
    ess_root: Optional[Path] = typer.Option(None, "--ess-root", help="Override KD_GAT_LAKE_ROOT"),
    repo_root: Optional[Path] = typer.Option(None, "--repo-root", help="Paper repo root (default: cwd)"),
):
    """Compare ESS manifest with local files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    s = load_schema(schema)
    root = repo_root or Path.cwd()
    show_diff(_resolve_ess(ess_root), s, root)


if __name__ == "__main__":
    app()
