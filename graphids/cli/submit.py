"""Typer ``submit`` command — SLURM submitter front-end.

Thin wrapper: parse flags, hand off to :func:`graphids.slurm.submit.submit`.
Preset mode is the positional argument; ``--command`` is the ops form.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import SetList, TlaList, app


@app.command("submit", rich_help_panel="SLURM", no_args_is_help=True)
def submit_cli(  # noqa: PLR0913 — every flag is a real surface
    preset: Annotated[
        Path | None,
        typer.Argument(
            help="Jsonnet preset path (training mode). Omit when using --command.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    command: Annotated[
        str | None,
        typer.Option("--command", help="Arbitrary shell command (ops mode). Requires --mode."),
    ] = None,
    action: Annotated[str, typer.Option("--action", help="fit|test (preset mode only)")] = "fit",
    mode: Annotated[
        str | None, typer.Option("--mode", help="gpu|cpu (required with --command)")
    ] = None,
    length: Annotated[str, typer.Option("--length", help="short|long")] = "long",
    smoke: Annotated[bool, typer.Option("--smoke", help="Shorthand for --length short")] = False,
    cpu: Annotated[bool, typer.Option("--cpu", help="Shorthand for --mode cpu")] = False,
    cluster: Annotated[
        str | None,
        typer.Option("--cluster", help="Target cluster (default: $GRAPHIDS_CLUSTER or pitzer)"),
    ] = None,
    dataset: Annotated[str | None, typer.Option("--dataset", help="Dataset TLA")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed TLA")] = None,
    scale: Annotated[str | None, typer.Option("--scale", help="Scale TLA")] = None,
    ckpt: Annotated[
        str | None,
        typer.Option("--ckpt", help="ckpt_path TLA (jsonnet field, distinct from --ckpt-path)"),
    ] = None,
    ckpt_path: Annotated[
        str | None,
        typer.Option(
            "--ckpt-path", help="Passthrough to `python -m graphids {fit,test} --ckpt-path`"
        ),
    ] = None,
    vgae_ckpt: Annotated[str | None, typer.Option("--vgae-ckpt", help="vgae_ckpt_path TLA")] = None,
    gat_ckpt: Annotated[str | None, typer.Option("--gat-ckpt", help="gat_ckpt_path TLA")] = None,
    lake_root: Annotated[str | None, typer.Option("--lake-root", help="lake_root TLA")] = None,
    mem_gb: Annotated[
        int | None, typer.Option("--mem-gb", help="Override profile memory (integer GB)")
    ] = None,
    timeout_min: Annotated[
        int | None, typer.Option("--timeout-min", help="Override profile walltime (minutes)")
    ] = None,
    time_from_history: Annotated[
        bool,
        typer.Option(
            "--time-from-history",
            help="Size walltime from MLflow history (fit + --long + --dataset only)",
        ),
    ] = False,
    tla: TlaList = None,
    set_: SetList = None,
    dep: Annotated[
        list[int] | None,
        typer.Option(
            "--dep",
            metavar="JID",
            help="afterok dep jid (repeatable). Env SBATCH_DEP is a fallback.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print sbatch argv to stderr; do not submit")
    ] = False,
) -> None:
    """Submit one SLURM job.

    Two shapes:

    * ``graphids submit <preset.jsonnet> [--dataset X --seed N ...]`` — training
    * ``graphids submit --command "..." --mode {gpu|cpu}`` — ops

    On success, prints the jid on stdout (for bash-capture compat:
    ``jid=$(graphids submit ...)``).
    """
    from graphids.slurm.submit import DRY_RUN_JID, submit

    if preset is None and not command:
        raise typer.BadParameter('supply a preset path or --command "..."')

    # Flat flags → TLA pairs. ``ckpt`` becomes the ``ckpt_path`` TLA (jsonnet
    # field) — distinct from ``ckpt_path`` (fit/test --ckpt-path passthrough).
    flag_tlas: list[tuple[str, object]] = []
    for key, val in (
        ("dataset", dataset),
        ("scale", scale),
        ("ckpt_path", ckpt),
        ("vgae_ckpt_path", vgae_ckpt),
        ("gat_ckpt_path", gat_ckpt),
        ("lake_root", lake_root),
    ):
        if val:
            flag_tlas.append((key, val))
    if seed is not None:
        flag_tlas.append(("seed", seed))
    flag_tlas.extend(tla or ())

    jid = submit(
        preset=preset,
        command=command,
        action=action,
        mode="cpu" if cpu else mode,
        length="short" if smoke else length,
        cluster=cluster,
        tlas=flag_tlas,
        sets=set_ or (),
        ckpt_path=ckpt_path,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
        time_from_history=time_from_history,
        dep_jids=tuple(dep or ()),
        dry_run=dry_run,
    )

    # Stdout contract: jid only (bash callers do ``jid=$(graphids submit ...)``).
    # Dry-run emits 0 as a non-chaining sentinel.
    if dry_run and jid == DRY_RUN_JID:
        print(DRY_RUN_JID)
    else:
        print(jid)
    sys.stdout.flush()
