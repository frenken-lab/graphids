"""Export metrics + artifacts + provenance to a HuggingFace dataset repo.

Implements ``~/plans/hf-dataset-design.md`` (Scope A — bug fixes + manifest
scoping; manifests/datasets/README/CITATION buckets are Scope B). Pattern
lifted from ``~/osc-usage/collect.py::push_to_hf``: stage to tmpdir,
``HfApi.upload_folder``, optionally ``create_tag``. Login-node only; no GPU,
no SLURM dependency.

All MLflow filter strings flow through ``_mlflow.build_search_filter``;
all run_dir → identity decoding flows through ``_mlflow.parse_run_dir``;
the ckpt subpath is ``constants.CKPT_SUBPATH``. The remaining subpath
literals (``traces.jsonl``, ``predictions/test``, ``artifacts``,
``resolved.json``, ``overrides.json``) mirror their writers in
``_otel.py``, ``core/models/base.py``, ``core/analysis/runner.py``, and
``cli/training.py`` respectively — change here only when the writer changes.
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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from graphids.cli.app import app
from graphids.config.constants import CKPT_SUBPATH
from graphids.config.settings import get_settings

_DEFAULT_METRIC = "f1_macro"
_BUCKETS: tuple[str, ...] = (
    "checkpoints",
    "predictions",
    "analysis",
    "traces",
    "provenance",
)


class Manifest(BaseModel):
    """Snapshot membership: the (datasets, variants, seeds) tuple set."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[str] = Field(min_length=1)
    variants: list[str] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    graphids_sha: str | None = None

    @field_validator("variants")
    @classmethod
    def _slash_form(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if "/" not in x]
        if bad:
            raise ValueError(f"must be 'group/variant'; got {bad}")
        return v

    @property
    def variant_set(self) -> set[str]:
        return set(self.variants)

    @property
    def dataset_set(self) -> set[str]:
        return set(self.datasets)

    @property
    def seed_set(self) -> set[int]:
        return set(self.seeds)


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


def _load_manifest(path: Path) -> Manifest:
    try:
        return Manifest.model_validate_json(path.read_text())
    except ValidationError as e:
        sys.exit(f"ERROR: invalid manifest {path}:\n{e}")


def _stage_run_artifacts(run_dir: Path, key: str, dest_root: Path) -> dict[str, int]:
    """Copy paper-relevant artifacts out of one run_dir into the staging tree.

    ``key`` is ``f"{group}_{variant}_{seed}"`` (matches ``hf-dataset-design.md``
    layout). Returns a count of files copied per bucket so the caller can
    report a summary.
    """
    counts: dict[str, int] = {b: 0 for b in _BUCKETS}

    best = run_dir / CKPT_SUBPATH
    if best.exists():
        ck_dest = dest_root / "artifacts" / "checkpoints"
        ck_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, ck_dest / f"{key}.ckpt")
        counts["checkpoints"] += 1
        sha_sidecar = best.with_suffix(best.suffix + ".sha256")
        if sha_sidecar.exists():
            shutil.copy2(sha_sidecar, ck_dest / f"{key}.ckpt.sha256")
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
    repo_id: Annotated[str, typer.Option("--repo-id")] = get_settings().hf_repo_id,
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

    from graphids._mlflow import build_search_filter, ensure_tracking_uri, parse_run_dir
    from graphids.analysis.compare import (
        effect_size,
        expected_max,
        leaderboard,
        tie_candidates,
    )

    spec = _load_manifest(manifest)

    repo_root = Path(__file__).resolve().parents[2]
    sha = _git_sha(repo_root, allow_dirty=allow_dirty)
    if spec.graphids_sha and not sha.startswith(spec.graphids_sha):
        sys.exit(
            f"ERROR: manifest pins graphids_sha={spec.graphids_sha} but HEAD={sha[:7]}; "
            "checkout the pinned commit or update the manifest."
        )

    token = os.environ.get("HF_TOKEN")
    if not token and not dry_run:
        sys.exit("HF_TOKEN not set in env. Use --dry-run to materialize without pushing.")

    uri = ensure_tracking_uri()
    if not uri:
        sys.exit("MLflow tracking URI not set (GRAPHIDS_LAKE_ROOT or MLFLOW_TRACKING_URI).")
    mlflow.set_tracking_uri(uri)

    # One round-trip for both phases — split client-side.
    all_df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=build_search_filter(status="FINISHED"),
        output_format="pandas",
    )
    if all_df.empty:
        sys.exit("No FINISHED runs found in MLflow.")

    test_df = _filter_to_manifest(all_df[all_df["tags.graphids.phase"] == "test"], spec)
    if test_df.empty:
        sys.exit(
            "No FINISHED test rows match the manifest filters "
            f"(datasets={sorted(spec.dataset_set)}, n_variants={len(spec.variant_set)}, "
            f"seeds={sorted(spec.seed_set)})."
        )
    fit_df = _filter_to_manifest(all_df[all_df["tags.graphids.phase"] == "fit"], spec)

    pairs = (
        test_df[["tags.graphids.group", "tags.graphids.dataset"]]
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
        missing_metric_pairs: list[str] = []
        for _, row in pairs.iterrows():
            grp, ds = row["group"], row["dataset"]
            row_n = 0
            metric_missing = False
            for name, fn in fns.items():
                try:
                    df_ = fn(grp, ds, metric=metric, seeds=spec.seed_set)
                except LookupError:
                    # Group has rows but doesn't produce this metric — all 4
                    # fns share _fetch_test_runs so they'd raise identically.
                    metric_missing = True
                    break
                if df_.empty:
                    continue
                df_ = df_.copy()
                df_["group"] = grp
                df_["dataset"] = ds
                bundles[name].append(df_)
                row_n += len(df_)
            if metric_missing:
                missing_metric_pairs.append(f"{grp}/{ds}")
                typer.echo(f"  {grp}/{ds}: metric {metric!r} not produced — skipped")
            elif row_n == 0:
                empty_pairs.append(f"{grp}/{ds}")
                typer.echo(f"  {grp}/{ds}: 0 rows after seed filter — skipped")
            else:
                typer.echo(f"  {grp}/{ds}: {row_n} metric rows")

        if missing_metric_pairs:
            typer.echo(
                f"  WARN: {len(missing_metric_pairs)} pair(s) don't produce metric={metric!r}: "
                f"{missing_metric_pairs}"
            )
        if empty_pairs:
            typer.echo(
                f"  WARN: {len(empty_pairs)} pair(s) had no rows after seed filter: {empty_pairs}"
            )

        for name, parts in bundles.items():
            if not parts:
                continue
            out = metrics_dir / f"{name}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
            typer.echo(f"  wrote metrics/{name}.parquet ({sum(len(p) for p in parts)} rows)")

        artifact_summary, n_runs_staged = _stage_fit_artifacts(fit_df, parse_run_dir, tmp)

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


def _filter_to_manifest(df, spec: Manifest):
    """Filter an MLflow ``search_runs`` frame down to the manifest membership."""
    import pandas as pd

    if df.empty:
        return df.reset_index(drop=True)
    gv = df["tags.graphids.group"].astype(str) + "/" + df["tags.graphids.variant"].astype(str)
    seed = pd.to_numeric(df["tags.graphids.seed"], errors="coerce").astype("Int64")
    mask = (
        gv.isin(spec.variant_set)
        & df["tags.graphids.dataset"].isin(spec.dataset_set)
        & seed.isin(spec.seed_set)
    )
    return df[mask].reset_index(drop=True)


def _stage_fit_artifacts(fit_df, parse_run_dir, tmp: Path) -> tuple[dict[str, int], int]:
    """Stage the latest FINISHED fit row's artifacts per (variant, dataset, seed)."""
    summary = {b: 0 for b in _BUCKETS}
    if fit_df.empty:
        typer.echo("WARN: no fit-phase runs match the manifest; artifacts/ not staged.")
        return summary, 0

    # Latest FINISHED fit per identity.
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
    off_tree = 0
    for _, frow in fit_df.iterrows():
        rd = frow.get("tags.graphids.run_dir")
        if not rd:
            continue
        run_dir = Path(rd)
        if not run_dir.exists():
            missing_run_dirs += 1
            continue
        identity = parse_run_dir(run_dir)
        if identity is None:
            off_tree += 1
            continue
        key = f"{identity.group}_{identity.variant}_{identity.seed}"
        counts = _stage_run_artifacts(run_dir, key, tmp)
        for k, v in counts.items():
            summary[k] += v
        n_runs_staged += 1
    if missing_run_dirs:
        typer.echo(f"  WARN: {missing_run_dirs} run_dir(s) missing on disk")
    if off_tree:
        typer.echo(f"  WARN: {off_tree} run_dir(s) off-tree (non-ablation layout)")
    typer.echo("  staged: " + ", ".join(f"{summary[b]} {b}" for b in _BUCKETS))
    return summary, n_runs_staged
