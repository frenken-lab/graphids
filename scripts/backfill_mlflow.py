"""One-shot MLflow backfill for a run that completed without opening a row.

Context: the first-ever concurrent cold-start of ``mlflow.db`` raced two
jobs on SQLite DDL init; the loser (VGAE 8588938) trained to completion
but ``start_training_run`` swallowed the DDL conflict. Authoritative
run state is recoverable from ``resolved.json`` + ``checkpoints/*.ckpt``
+ ``checkpoints/best_model.ckpt.sha256`` + the SLURM sacct window.

Usage:
    python scripts/backfill_mlflow.py \\
        --run-dir /fs/.../vgae/seed_42 \\
        --job-id 8588938 \\
        --cluster cardinal \\
        --start 2026-04-16T20:15:49 \\
        --end   2026-04-16T21:02:10
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from graphids._mlflow import (
    _cache_digest_tags,
    _flatten_params,
    _git_sha_tag,
    _identity_tags,
    ensure_tracking_uri,
    parse_run_dir,
    run_name_for,
)


def _ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--cluster", default="cardinal")
    ap.add_argument("--start", required=True, help="ISO-8601 UTC, e.g. 2026-04-16T20:15:49")
    ap.add_argument("--end", required=True, help="ISO-8601 UTC, e.g. 2026-04-16T21:02:10")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    identity = parse_run_dir(run_dir)
    if identity is None:
        raise SystemExit(f"run_dir off-tree: {run_dir}")

    resolved = json.loads((run_dir / "resolved.json").read_text())
    best = torch.load(
        run_dir / "checkpoints" / "best_model.ckpt", map_location="cpu", weights_only=False
    )
    last = torch.load(run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    sha_path = run_dir / "checkpoints" / "best_model.ckpt.sha256"
    ckpt_sha = sha_path.read_text().strip().split()[0] if sha_path.exists() else ""

    best_metrics = {
        k: float(v.item() if hasattr(v, "item") else v) for k, v in best["metrics"].items()
    }
    last_metrics = {
        k: float(v.item() if hasattr(v, "item") else v) for k, v in last["metrics"].items()
    }
    best_epoch = int(best["epoch"])
    epochs_run = int(last["epoch"]) + 1

    import mlflow
    from mlflow.tracking import MlflowClient

    uri = ensure_tracking_uri()
    if not uri:
        raise SystemExit("MLFLOW_TRACKING_URI not resolvable — set LAKE_ROOT")
    mlflow.set_tracking_uri(uri)

    experiment = f"graphids/{identity.group}/{identity.variant}"
    client = MlflowClient(tracking_uri=uri)
    if client.get_experiment_by_name(experiment) is None:
        client.create_experiment(experiment)
    mlflow.set_experiment(experiment)

    run_name = run_name_for(identity, cluster=args.cluster)
    end_ms = _ms(args.end)

    tags = {
        **_identity_tags(identity, run_dir, args.cluster),
        **_cache_digest_tags(resolved),
        **_git_sha_tag(),
        "graphids.phase": "fit",
        "graphids.backfilled": "true",
        "graphids.backfill_reason": "mlflow_schema_race_cold_start",
        "slurm.slurm_job_id": str(args.job_id),
        "slurm.slurm_cluster_name": args.cluster,
        "mlflow.note.content": (
            "Backfilled from run_dir sidecars. No per-epoch metric history — only "
            "final best/last metrics + params + identity tags."
        ),
    }
    if ckpt_sha:
        tags["graphids.ckpt_sha256"] = ckpt_sha
        best_path = str(run_dir / "checkpoints" / "best_model.ckpt")
        tags["graphids.best_ckpt_path"] = best_path

    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(_flatten_params(resolved))
        mlflow.log_metrics(
            {
                "val_loss": best_metrics["val_loss"],
                "train_loss": best_metrics["train_loss"],
            },
            step=best_epoch,
        )
        mlflow.log_metrics(
            {
                "val_loss": last_metrics["val_loss"],
                "train_loss": last_metrics["train_loss"],
            },
            step=int(last["epoch"]),
        )
        mlflow.log_metrics(
            {
                "epochs_run": float(epochs_run),
                "best_epoch": float(best_epoch),
            }
        )
        run = mlflow.active_run()
        client.set_terminated(run.info.run_id, status="FINISHED", end_time=end_ms)

    print(f"backfilled: run_name={run_name} experiment={experiment}")
    print(
        f"  best_epoch={best_epoch} val_loss={best_metrics['val_loss']:.4f} train_loss={best_metrics['train_loss']:.4f}"
    )
    print(
        f"  last_epoch={last['epoch']} val_loss={last_metrics['val_loss']:.4f} train_loss={last_metrics['train_loss']:.4f}"
    )
    print(f"  epochs_run={epochs_run} ckpt_sha={ckpt_sha[:12]}")


if __name__ == "__main__":
    main()
