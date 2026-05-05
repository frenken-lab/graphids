"""Test runners — login-node pytest + cluster smoke submission.

``graphids test unit`` runs login-node tests (markers: ``not slurm and
not slow``) directly. ``graphids test smoke`` renders the
``smoke.gat_taunorm`` plan and submits the fit→test row pair via
the existing ``graphids submit`` chassis (afterok-chained, ckpt-path
threaded from the test row's upstream).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app

test_app = typer.Typer(
    name="test",
    help="Test runners (unit on login node, smoke on cluster).",
    no_args_is_help=True,
)
app.add_typer(test_app, name="test")


@test_app.command("unit")
def unit(
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="pytest paths (default: tests/)"),
    ] = None,
    extra: Annotated[
        list[str] | None,
        typer.Option("--extra", help="Extra pytest args (repeat per arg)."),
    ] = None,
) -> None:
    """Login-node pytest, markers ``not slurm and not slow``.

    Exit code propagates from pytest. ``conftest.py`` auto-skips
    ``slurm``-marked tests on login nodes regardless, but the explicit
    marker filter also drops ``slow`` so this stays a fast feedback loop.
    """
    cmd = ["pytest", "-m", "not slurm and not slow"]
    if extra:
        cmd.extend(extra)
    cmd.extend(paths or ["tests"])
    raise typer.Exit(subprocess.run(cmd).returncode)


@test_app.command("smoke")
def smoke(
    dataset: Annotated[str, typer.Option("--dataset")] = "hcrl_sa",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    cluster: Annotated[str, typer.Option("--cluster")] = "pitzer",
) -> None:
    """Submit the GAT+TauNorm smoke plan (fit→test, afterok-chained)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="smoke_plan_", delete=False
    ) as f:
        plan_path = f.name
    subprocess.run(
        [
            sys.executable, "-m", "graphids", "run", "smoke.gat_taunorm",
            "--dataset", dataset, "--seed", str(seed), "-o", plan_path,
        ],
        check=True,
    )
    rows = json.loads(Path(plan_path).read_text())

    prev_jid: str | None = None
    for row in rows:
        cmd = [
            sys.executable, "-m", "graphids", "submit",
            "--row", json.dumps(row),
            "--cluster", cluster,
            "--length", "short",
        ]
        if row["action"] == "test":
            if prev_jid is None:
                raise typer.BadParameter("test row appeared before any fit row")
            cmd.extend(["--depends-on-afterok", prev_jid])
            upstreams = row.get("upstreams") or []
            if upstreams:
                cmd.extend(["--ckpt-path", upstreams[0]["ckpt_path"]])
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        jid = result.stdout.strip().splitlines()[-1]
        print(jid)
        prev_jid = jid
