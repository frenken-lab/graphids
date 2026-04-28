"""Export metrics + artifacts + provenance to a HuggingFace dataset repo.

Implements `~/plans/hf-dataset-design.md`: paper read-surface bundle of
metrics parquets, best-only checkpoints, test-set predictions, analysis
artifacts, traces, and provenance. Pattern lifted from
`~/osc-usage/collect.py::push_to_hf` — stage to tmpdir,
`HfApi.upload_folder`, optionally `create_tag`. Login-node only; no GPU,
no SLURM dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app

_DEFAULT_REPO = "buckeyeguy/graphids-kd-gat"
_DEFAULT_METRIC = "f1_macro"


def _stage_run_artifacts(run_dir: Path, key: str, dest_root: Path) -> dict[str, int]:
    """Copy paper-relevant artifacts out of one run_dir into the staging tree.

    `key` is ``f"{group}_{variant}_{seed}"`` — the per-run namespace used for
    every bucket. Returns a count of files copied per bucket so the caller
    can report a summary.
    """
    counts: dict[str, int] = {
        "checkpoints": 0,
        "predictions": 0,
        "analysis": 0,
        "traces": 0,
        "provenance": 0,
    }

    best = run_dir / "checkpoints" / "best_model.ckpt"
    if best.exists():
        ck_dest = dest_root / "artifacts" / "checkpoints"
        ck_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, ck_dest / f"{key}.ckpt")
        counts["checkpoints"] += 1
        sha = best.with_suffix(best.suffix + ".sha256")
        if sha.exists():
            shutil.copy2(sha, ck_dest / f"{key}.ckpt.sha256")
            counts["checkpoints"] += 1

    test_preds = run_dir / "predictions" / "test"
    if test_preds.is_dir():
        pd_dest = dest_root / "artifacts" / "predictions" / key
        pd_dest.mkdir(parents=True, exist_ok=True)
        for p in test_preds.iterdir():
            if p.is_file():
                shutil.copy2(p, pd_dest / p.name)
                counts["predictions"] += 1

    analysis_src = run_dir / "artifacts"
    if analysis_src.is_dir() and any(analysis_src.iterdir()):
        an_dest = dest_root / "artifacts" / "analysis" / key
        shutil.copytree(analysis_src, an_dest, dirs_exist_ok=True)
        counts["analysis"] += sum(1 for _ in an_dest.rglob("*") if _.is_file())

    traces_src = run_dir / "traces.jsonl"
    if traces_src.exists() and traces_src.stat().st_size > 0:
        tr_dest = dest_root / "logs" / "traces"
        tr_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(traces_src, tr_dest / f"{key}.jsonl")
        counts["traces"] += 1

    pr_dest = dest_root / "provenance" / "runs" / key
    pr_dest.mkdir(parents=True, exist_ok=True)
    for fname in ("resolved.json", "overrides.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, pr_dest / fname)
            counts["provenance"] += 1

    return counts


@app.command("push-hf")
def push_hf(
    ablation_set: Annotated[
        str,
        typer.Option(
            "--ablation-set", help="Named snapshot tag (e.g. 'set_01_v1'). Becomes the HF git tag."
        ),
    ],
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            help="Filter to specific datasets (repeatable). Default: auto-discover from MLflow.",
        ),
    ] = None,
    repo_id: Annotated[str, typer.Option("--repo-id")] = _DEFAULT_REPO,
    metric: Annotated[
        str, typer.Option("--metric", help="Test-phase metric name to aggregate.")
    ] = _DEFAULT_METRIC,
    tag: Annotated[
        bool, typer.Option("--tag/--no-tag", help="Create HF git tag named --ablation-set.")
    ] = True,
    metrics_only: Annotated[
        bool,
        typer.Option(
            "--metrics-only",
            help="Skip artifacts/logs/provenance buckets — push only the metrics parquets + metadata.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Stage to a tmpdir and print what would be uploaded; do not push.",
        ),
    ] = False,
) -> None:
    """Push metrics + provenance bundle to the configured HF dataset repo.

    Discovers (group, dataset) pairs from FINISHED test-phase MLflow rows,
    materializes one parquet per analysis function (concatenated across
    pairs with `group`/`dataset` columns added), writes provenance, and
    uploads via `HfApi.upload_folder`. Idempotent — re-pushing the same
    state produces an identical commit.
    """
    import mlflow
    import pandas as pd

    from graphids._mlflow import ensure_tracking_uri
    from graphids.analysis.compare import effect_size, expected_max, leaderboard, tie_candidates

    token = os.environ.get("HF_TOKEN")
    if not token and not dry_run:
        sys.exit(
            "HF_TOKEN not set in env (~/.env.local). Use --dry-run to materialize without pushing."
        )

    uri = ensure_tracking_uri()
    if not uri:
        sys.exit("MLflow tracking URI not set (GRAPHIDS_LAKE_ROOT or MLFLOW_TRACKING_URI).")
    mlflow.set_tracking_uri(uri)

    df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="tags.\"graphids.phase\" = 'test' and attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    if df.empty:
        sys.exit("No FINISHED test-phase runs found in MLflow.")

    pairs = (
        df[["tags.graphids.group", "tags.graphids.dataset"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"tags.graphids.group": "group", "tags.graphids.dataset": "dataset"})
        .sort_values(["dataset", "group"])
        .reset_index(drop=True)
    )
    if datasets:
        pairs = pairs[pairs["dataset"].isin(datasets)].reset_index(drop=True)
    if pairs.empty:
        sys.exit(f"No (group, dataset) pairs match filter datasets={datasets}.")

    typer.echo(f"Discovered {len(pairs)} (group, dataset) pair(s) from MLflow.")

    sha = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )

    fns = {
        "leaderboard": leaderboard,
        "effect_size": effect_size,
        "expected_max": expected_max,
        "tie_candidates": tie_candidates,
    }

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        metrics_dir = tmp / "metrics"
        metrics_dir.mkdir()

        bundles: dict[str, list[pd.DataFrame]] = {name: [] for name in fns}
        for _, row in pairs.iterrows():
            grp, ds = row["group"], row["dataset"]
            typer.echo(f"  {grp}/{ds}", nl=False)
            for name, fn in fns.items():
                try:
                    df_ = fn(grp, ds, metric=metric)
                except Exception as exc:
                    typer.echo(f"  [WARN {name}: {exc}]", nl=False)
                    continue
                if df_.empty:
                    continue
                df_ = df_.copy()
                df_["group"] = grp
                df_["dataset"] = ds
                bundles[name].append(df_)
            typer.echo("")

        for name, parts in bundles.items():
            if not parts:
                typer.echo(f"  {name}: no rows, skipped")
                continue
            out = metrics_dir / f"{name}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
            typer.echo(f"  wrote {out.relative_to(tmp)} ({sum(len(p) for p in parts)} rows)")

        artifact_summary: dict[str, int] = {
            "checkpoints": 0,
            "predictions": 0,
            "analysis": 0,
            "traces": 0,
            "provenance": 0,
        }
        n_runs_staged = 0
        if not metrics_only:
            target_datasets = set(pairs["dataset"].unique())
            fit_df = mlflow.search_runs(
                search_all_experiments=True,
                filter_string=(
                    "tags.\"graphids.phase\" = 'fit' and attributes.status = 'FINISHED'"
                ),
                output_format="pandas",
            )
            if not fit_df.empty:
                fit_df = fit_df[fit_df["tags.graphids.dataset"].isin(target_datasets)]
            if fit_df.empty:
                typer.echo("  WARN: no FINISHED fit-phase runs found; skipping artifact staging")
            else:
                # Latest FINISHED fit per (group, variant, dataset, seed).
                fit_df = fit_df.sort_values("start_time", ascending=False).drop_duplicates(
                    subset=[
                        "tags.graphids.group",
                        "tags.graphids.variant",
                        "tags.graphids.dataset",
                        "tags.graphids.seed",
                    ],
                    keep="first",
                )
                typer.echo(f"\nStaging artifacts for {len(fit_df)} fit run(s)...")
                missing_run_dirs = 0
                for _, frow in fit_df.iterrows():
                    grp = frow.get("tags.graphids.group")
                    var = frow.get("tags.graphids.variant")
                    seed = frow.get("tags.graphids.seed")
                    rd = frow.get("tags.graphids.run_dir")
                    if not (grp and var and seed and rd):
                        continue
                    run_dir = Path(rd)
                    if not run_dir.exists():
                        missing_run_dirs += 1
                        continue
                    key = f"{grp}_{var}_seed{seed}"
                    counts = _stage_run_artifacts(run_dir, key, tmp)
                    for k, v in counts.items():
                        artifact_summary[k] += v
                    n_runs_staged += 1
                if missing_run_dirs:
                    typer.echo(f"  WARN: {missing_run_dirs} run_dir(s) missing on disk")
                typer.echo(
                    f"  staged: {artifact_summary['checkpoints']} ckpt files, "
                    f"{artifact_summary['predictions']} pred files, "
                    f"{artifact_summary['analysis']} analysis files, "
                    f"{artifact_summary['traces']} traces, "
                    f"{artifact_summary['provenance']} provenance files"
                )

        metadata = {
            "ablation_set": ablation_set,
            "graphids_sha": sha,
            "metric": metric,
            "pushed_at": datetime.now(UTC).isoformat(),
            "groups": sorted(pairs["group"].unique().tolist()),
            "datasets": sorted(pairs["dataset"].unique().tolist()),
            "n_pairs": int(len(pairs)),
            "metrics_only": metrics_only,
            "n_runs_staged": n_runs_staged,
            "artifact_counts": artifact_summary,
        }
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2))

        if dry_run:
            typer.echo(
                f"\n--dry-run: would upload to {repo_id}, tag={ablation_set if tag else '(none)'}"
            )
            typer.echo("  contents:")
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    typer.echo(f"    {p.relative_to(tmp)} ({p.stat().st_size} bytes)")
            return

        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        api = HfApi(token=token)
        try:
            api.dataset_info(repo_id)
        except RepositoryNotFoundError:
            typer.echo(f"Creating private dataset repo {repo_id}")
            api.create_repo(repo_id, repo_type="dataset", private=True)
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"metrics push: {ablation_set} (graphids@{sha[:7]})",
        )
        if tag:
            api.create_tag(
                repo_id,
                repo_type="dataset",
                tag=ablation_set,
                tag_message=f"Ablation set {ablation_set} (graphids@{sha[:7]})",
            )
        typer.echo(f"\nPushed to https://huggingface.co/datasets/{repo_id}")
        if tag:
            typer.echo(f"Tag: {ablation_set}")
