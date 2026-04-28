"""Export metrics + artifacts + provenance to a HuggingFace dataset repo.

Implements ``~/plans/hf-dataset-design.md`` (Scope A — bug fixes + manifest
scoping; manifests/datasets/README/CITATION buckets are Scope B). Pattern
lifted from ``~/osc-usage/collect.py::push_to_hf``: stage to tmpdir,
``HfApi.upload_folder``, optionally ``create_tag``. Login-node only; no GPU,
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


def _git_sha(repo_root: Path, *, allow_dirty: bool) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not allow_dirty:
        rc = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=repo_root
        ).returncode
        if rc != 0:
            sys.exit(
                "ERROR: working tree is dirty (uncommitted changes). "
                "Re-run with --allow-dirty to push anyway."
            )
    return sha


def _load_manifest(path: Path) -> dict:
    """Parse and validate the snapshot manifest JSON.

    Schema: ``{datasets: [...], variants: ['group/variant', ...], seeds: [...],
    graphids_sha?: '...'}``. ``graphids_sha`` is optional; when present the
    push aborts unless HEAD matches.
    """
    spec = json.loads(path.read_text())
    required = {"datasets", "variants", "seeds"}
    missing = required - spec.keys()
    if missing:
        sys.exit(f"ERROR: manifest {path} missing keys: {sorted(missing)}")
    for key in required:
        if not spec[key]:
            sys.exit(f"ERROR: manifest {path} has empty '{key}' list")
    bad = [v for v in spec["variants"] if not isinstance(v, str) or "/" not in v]
    if bad:
        sys.exit(f"ERROR: manifest variants must be 'group/variant'; got {bad}")
    return spec


def _stage_run_artifacts(run_dir: Path, key: str, dest_root: Path) -> dict[str, int]:
    """Copy paper-relevant artifacts out of one run_dir into the staging tree.

    ``key`` is ``f"{group}_{variant}_{seed}"`` (matches ``hf-dataset-design.md``
    layout). Returns a count of files copied per bucket so the caller can
    report a summary.
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
        typer.Option("--ablation-set", help="Snapshot name → HF git tag (e.g. 'set_01_v1')."),
    ],
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            help=(
                "JSON: {datasets:[], variants:['group/variant',...], seeds:[], "
                "graphids_sha?}. Defines snapshot membership."
            ),
        ),
    ],
    repo_id: Annotated[str, typer.Option("--repo-id")] = _DEFAULT_REPO,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Test-phase metric name to aggregate."),
    ] = _DEFAULT_METRIC,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty/--require-clean-tree",
            help="Allow push when the working tree has uncommitted changes.",
        ),
    ] = False,
    tag: Annotated[
        bool,
        typer.Option("--tag/--no-tag", help="Create HF git tag named --ablation-set."),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Stage to a tmpdir and print what would be uploaded; do not push.",
        ),
    ] = False,
) -> None:
    """Push metrics + provenance bundle to the configured HF dataset repo.

    Membership is read from ``--manifest`` — only MLflow rows + run_dirs whose
    ``(group, variant, dataset, seed)`` tuple appears in the manifest are
    staged. Re-pushing identical content produces a new commit (the upload
    timestamp lives in the commit message, not in tracked content), so the
    parquets/json bytes are stable across pushes. Tag creation is one-shot
    per ``--ablation-set`` — re-using a tag aborts before upload.
    """
    import mlflow
    import pandas as pd

    from graphids._mlflow import ensure_tracking_uri
    from graphids.analysis.compare import (
        effect_size,
        expected_max,
        leaderboard,
        tie_candidates,
    )

    spec = _load_manifest(manifest)
    sel_datasets = set(spec["datasets"])
    sel_variants = set(spec["variants"])
    sel_seeds = {int(s) for s in spec["seeds"]}

    repo_root = Path(__file__).resolve().parents[2]
    sha = _git_sha(repo_root, allow_dirty=allow_dirty)
    expected_sha = spec.get("graphids_sha")
    if expected_sha and sha != expected_sha:
        sys.exit(
            f"ERROR: manifest pins graphids_sha={expected_sha[:7]} but HEAD={sha[:7]}; "
            "checkout the pinned commit or update the manifest."
        )

    token = os.environ.get("HF_TOKEN")
    if not token and not dry_run:
        sys.exit("HF_TOKEN not set in env. Use --dry-run to materialize without pushing.")

    uri = ensure_tracking_uri()
    if not uri:
        sys.exit("MLflow tracking URI not set (GRAPHIDS_LAKE_ROOT or MLFLOW_TRACKING_URI).")
    mlflow.set_tracking_uri(uri)

    test_df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="tags.\"graphids.phase\" = 'test' and attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    if test_df.empty:
        sys.exit("No FINISHED test-phase runs found in MLflow.")

    test_df = _filter_to_manifest(test_df, sel_variants, sel_datasets, sel_seeds)
    if test_df.empty:
        sys.exit(
            "No FINISHED test rows match the manifest filters "
            f"(datasets={sorted(sel_datasets)}, n_variants={len(sel_variants)}, "
            f"seeds={sorted(sel_seeds)})."
        )

    pairs = (
        test_df[["tags.graphids.group", "tags.graphids.dataset"]]
        .dropna()
        .drop_duplicates()
        .rename(
            columns={
                "tags.graphids.group": "group",
                "tags.graphids.dataset": "dataset",
            }
        )
        .sort_values(["dataset", "group"])
        .reset_index(drop=True)
    )
    typer.echo(
        f"Manifest matches {len(test_df)} test row(s) across {len(pairs)} (group, dataset) pair(s)."
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
        empty_pairs: list[str] = []
        for _, row in pairs.iterrows():
            grp, ds = row["group"], row["dataset"]
            row_n = 0
            for name, fn in fns.items():
                df_ = fn(grp, ds, metric=metric, seeds=sel_seeds)
                if df_.empty:
                    continue
                df_ = df_.copy()
                df_["group"] = grp
                df_["dataset"] = ds
                bundles[name].append(df_)
                row_n += len(df_)
            typer.echo(f"  {grp}/{ds}: {row_n} metric rows")
            if row_n == 0:
                empty_pairs.append(f"{grp}/{ds}")

        if empty_pairs:
            typer.echo(
                f"  WARN: {len(empty_pairs)} pair(s) had no rows for metric={metric!r}: "
                f"{empty_pairs}"
            )

        for name, parts in bundles.items():
            if not parts:
                continue
            out = metrics_dir / f"{name}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
            typer.echo(f"  wrote metrics/{name}.parquet ({sum(len(p) for p in parts)} rows)")

        artifact_summary, n_runs_staged = _stage_fit_artifacts(
            mlflow, sel_variants, sel_datasets, sel_seeds, tmp
        )

        # Stable metadata (no timestamp — keeps re-pushes byte-identical).
        # Push timestamp lives in the commit message instead.
        metadata = {
            "ablation_set": ablation_set,
            "graphids_sha": sha,
            "metric": metric,
            "groups": sorted(pairs["group"].unique().tolist()),
            "datasets": sorted(pairs["dataset"].unique().tolist()),
            "n_pairs": int(len(pairs)),
            "n_runs_staged": n_runs_staged,
            "artifact_counts": artifact_summary,
        }
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

        if dry_run:
            typer.echo(
                f"\n--dry-run: would upload to {repo_id}, tag={ablation_set if tag else '(none)'}"
            )
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    typer.echo(f"    {p.relative_to(tmp)} ({p.stat().st_size} bytes)")
            return

        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        api = HfApi(token=token)
        try:
            api.dataset_info(repo_id)
            repo_exists = True
        except RepositoryNotFoundError:
            repo_exists = False

        if tag and repo_exists:
            refs = api.list_repo_refs(repo_id, repo_type="dataset")
            if any(r.name == ablation_set for r in refs.tags):
                sys.exit(
                    f"ERROR: tag '{ablation_set}' already exists on {repo_id}. "
                    "Pick a new --ablation-set or use --no-tag."
                )

        if not repo_exists:
            typer.echo(f"Creating private dataset repo {repo_id}")
            api.create_repo(repo_id, repo_type="dataset", private=True)

        pushed_at = datetime.now(UTC).isoformat()
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(f"metrics push: {ablation_set} (graphids@{sha[:7]}) at {pushed_at}"),
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


def _filter_to_manifest(
    df,
    sel_variants: set[str],
    sel_datasets: set[str],
    sel_seeds: set[int],
):
    """Filter an MLflow ``search_runs`` frame down to the manifest membership."""
    import pandas as pd

    gv = df["tags.graphids.group"].astype(str) + "/" + df["tags.graphids.variant"].astype(str)
    seed = pd.to_numeric(df["tags.graphids.seed"], errors="coerce").astype("Int64")
    mask = (
        gv.isin(sel_variants)
        & df["tags.graphids.dataset"].isin(sel_datasets)
        & seed.isin(sel_seeds)
    )
    return df[mask].reset_index(drop=True)


def _stage_fit_artifacts(
    mlflow,
    sel_variants: set[str],
    sel_datasets: set[str],
    sel_seeds: set[int],
    tmp: Path,
) -> tuple[dict[str, int], int]:
    """Discover the latest FINISHED fit row per (variant, dataset, seed) and stage."""
    summary = {k: 0 for k in ("checkpoints", "predictions", "analysis", "traces", "provenance")}
    fit_df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=("tags.\"graphids.phase\" = 'fit' and attributes.status = 'FINISHED'"),
        output_format="pandas",
    )
    if fit_df.empty:
        typer.echo("WARN: no FINISHED fit-phase runs in MLflow; artifacts/ not staged.")
        return summary, 0

    fit_df = _filter_to_manifest(fit_df, sel_variants, sel_datasets, sel_seeds)
    if fit_df.empty:
        typer.echo("WARN: no fit-phase runs match the manifest; artifacts/ not staged.")
        return summary, 0

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
    n_runs_staged = 0
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
        key = f"{grp}_{var}_{int(seed)}"
        counts = _stage_run_artifacts(run_dir, key, tmp)
        for k, v in counts.items():
            summary[k] += v
        n_runs_staged += 1
    if missing_run_dirs:
        typer.echo(f"  WARN: {missing_run_dirs} run_dir(s) missing on disk")
    typer.echo(
        f"  staged: {summary['checkpoints']} ckpt, "
        f"{summary['predictions']} pred, "
        f"{summary['analysis']} analysis, "
        f"{summary['traces']} traces, "
        f"{summary['provenance']} provenance"
    )
    return summary, n_runs_staged
