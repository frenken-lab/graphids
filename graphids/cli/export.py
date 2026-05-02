"""Push metrics + artifacts + provenance to a HuggingFace dataset repo.

Stage to tmpdir → HfApi.upload_folder → optionally create_tag.
Filter strings through _mlflow.build_search_filter; ckpt subpath from constants.CKPT_SUBPATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from graphids.cli.app import app
from graphids.config.constants import CKPT_SUBPATH

_DEFAULT_METRIC = "f1_macro"
_BUCKETS: tuple[str, ...] = ("checkpoints", "predictions", "analysis", "traces", "provenance")


def _die(msg: str, code: int = 1) -> NoReturn:
    typer.echo(f"ERROR: {msg}", err=True)
    raise typer.Exit(code=code)


class Manifest(BaseModel):
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


def _git_sha(repo_root: Path, *, allow_dirty: bool) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not allow_dirty:
        rc = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=repo_root
        ).returncode
        if rc != 0:
            _die(
                "working tree is dirty (uncommitted changes). Re-run with --allow-dirty to push anyway."
            )
    return sha


def _load_manifest(path: Path) -> Manifest:
    try:
        return Manifest.model_validate_json(path.read_text())
    except ValidationError as e:
        _die(f"invalid manifest {path}:\n{e}")


def _stage_run_artifacts(run_dir: Path, key: str, dest_root: Path) -> dict[str, int]:
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


def _filter_to_manifest(df, spec: Manifest):
    import pandas as pd

    if df.empty:
        return df.reset_index(drop=True)
    gv = df["tags.graphids.group"].astype(str) + "/" + df["tags.graphids.variant"].astype(str)
    seed = pd.to_numeric(df["tags.graphids.seed"], errors="coerce").astype("Int64")
    mask = (
        gv.isin(set(spec.variants))
        & df["tags.graphids.dataset"].isin(set(spec.datasets))
        & seed.isin(set(spec.seeds))
    )
    return df[mask].reset_index(drop=True)


def _stage_fit_artifacts(fit_df, parse_run_dir, tmp: Path) -> tuple[dict[str, int], int]:
    summary = {b: 0 for b in _BUCKETS}
    if fit_df.empty:
        typer.echo("WARN: no fit-phase runs match the manifest; artifacts/ not staged.")
        return summary, 0

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
    n_runs_staged = missing_run_dirs = off_tree = 0
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


@app.command("push-hf")
def push_hf(
    ablation_set: Annotated[
        str, typer.Option("--ablation-set", help="Snapshot name → HF git tag (e.g. 'set_01_v1').")
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, help="Membership manifest JSON."),
    ],
    repo_id: Annotated[str | None, typer.Option("--repo-id")] = None,
    metric: Annotated[str, typer.Option("--metric", help="Metric.")] = _DEFAULT_METRIC,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty/--require-clean-tree")] = False,
    tag: Annotated[bool, typer.Option("--tag/--no-tag")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Push metrics + provenance bundle to the configured HF dataset repo."""
    import mlflow
    import pandas as pd

    from graphids._mlflow import build_search_filter, ensure_tracking_uri, parse_run_dir
    from graphids.analysis.compare import (
        effect_size,
        expected_max,
        leaderboard,
        tie_candidates,
    )
    from graphids.config.settings import get_settings

    repo_id = repo_id or get_settings().hf_repo_id

    spec = _load_manifest(manifest)
    sha = _git_sha(Path(__file__).resolve().parents[2], allow_dirty=allow_dirty)
    if spec.graphids_sha and not sha.startswith(spec.graphids_sha):
        _die(
            f"manifest pins graphids_sha={spec.graphids_sha} but HEAD={sha[:7]}; checkout the pinned commit or update the manifest."
        )

    token = os.environ.get("HF_TOKEN")
    if not token and not dry_run:
        _die("HF_TOKEN not set in env. Use --dry-run to materialize without pushing.")

    uri = ensure_tracking_uri()
    if not uri:
        _die("MLflow tracking URI not set (GRAPHIDS_LAKE_ROOT or MLFLOW_TRACKING_URI).")
    mlflow.set_tracking_uri(uri)

    all_df = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=build_search_filter(status="FINISHED"),
        output_format="pandas",
    )
    if all_df.empty:
        _die("No FINISHED runs found in MLflow.")

    test_df = _filter_to_manifest(all_df[all_df["tags.graphids.phase"] == "test"], spec)
    if test_df.empty:
        _die(
            f"No FINISHED test rows match the manifest "
            f"(datasets={sorted(set(spec.datasets))}, n_variants={len(spec.variants)}, "
            f"seeds={sorted(set(spec.seeds))})."
        )
    fit_df = _filter_to_manifest(all_df[all_df["tags.graphids.phase"] == "fit"], spec)

    pairs = (
        test_df[["tags.graphids.group", "tags.graphids.dataset"]]
        .drop_duplicates()
        .rename(columns={"tags.graphids.group": "group", "tags.graphids.dataset": "dataset"})
        .sort_values(["dataset", "group"])
        .reset_index(drop=True)
    )
    typer.echo(f"Matched {len(test_df)} test rows, {len(pairs)} group/dataset pairs.")

    _compare_fns = {
        "leaderboard": leaderboard,
        "effect_size": effect_size,
        "expected_max": expected_max,
        "tie_candidates": tie_candidates,
    }

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "metrics").mkdir()

        bundles: dict[str, list[pd.DataFrame]] = {name: [] for name in _compare_fns}
        no_metric: list[str] = []
        no_rows: list[str] = []
        for _, row in pairs.iterrows():
            grp, ds = row["group"], row["dataset"]
            row_n, metric_missing = 0, False
            for name, fn in _compare_fns.items():
                try:
                    df_ = fn(grp, ds, metric=metric, seeds=set(spec.seeds))
                except LookupError:
                    metric_missing = True
                    break
                if not df_.empty:
                    bundles[name].append(df_.assign(group=grp, dataset=ds))
                    row_n += len(df_)
            label = f"{grp}/{ds}"
            if metric_missing:
                no_metric.append(label)
                typer.echo(f"  {label}: metric {metric!r} not produced — skipped")
            elif row_n == 0:
                no_rows.append(label)
                typer.echo(f"  {label}: 0 rows after seed filter — skipped")
            else:
                typer.echo(f"  {label}: {row_n} metric rows")
        if no_metric:
            typer.echo(f"  WARN: {len(no_metric)} pair(s) don't produce {metric!r}: {no_metric}")
        if no_rows:
            typer.echo(f"  WARN: {len(no_rows)} pair(s) had 0 rows after seed filter: {no_rows}")

        for name, parts in bundles.items():
            if not parts:
                continue
            out = tmp / "metrics" / f"{name}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
            typer.echo(f"  wrote metrics/{name}.parquet ({sum(len(p) for p in parts)} rows)")

        artifact_summary, n_runs_staged = _stage_fit_artifacts(fit_df, parse_run_dir, tmp)

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
                _die(
                    f"tag '{ablation_set}' already exists on {repo_id}. Pick a new --ablation-set or use --no-tag."
                )

        if not repo_exists:
            typer.echo(f"Creating private dataset repo {repo_id}")
            api.create_repo(repo_id, repo_type="dataset", private=True)

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"metrics push: {ablation_set} (graphids@{sha[:7]}) at {datetime.now(UTC).isoformat()}",
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
