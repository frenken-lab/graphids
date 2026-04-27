"""Typer ``submit`` command — Typer-decorated forward to ``slurm.submit.submit``.

All submission logic (depends-on resolution, skip-if-finished MLflow
check, flat-flag → TLA construction, submitit dispatch) lives in
:func:`graphids.slurm.submit.submit`. This module only declares the
Typer surface; the body forwards every flag verbatim and prints the
returned jid (``0`` when skipped or dry-run, for ``jid=$(graphids submit ...)``
bash-capture compat).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import SetList, TlaList, app


@app.command("submit", rich_help_panel="SLURM", no_args_is_help=True)
def submit_cli(  # noqa: PLR0913 — every flag is a real CLI surface
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
    ckpt_tla: Annotated[
        str | None,
        typer.Option(
            "--ckpt-tla",
            help=(
                "Set the ``ckpt_path`` TLA (jsonnet field). Distinct from "
                "--ckpt-path (resume passthrough) and --depends-on (MLflow lookup)."
            ),
        ),
    ] = None,
    ckpt_path: Annotated[
        str | None,
        typer.Option(
            "--ckpt-path",
            help="Resume current preset — passthrough to `python -m graphids {fit,test} --ckpt-path`",
        ),
    ] = None,
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
    depends_on: Annotated[
        str | None,
        typer.Option(
            "--depends-on",
            metavar="<variant>[:<seed>][,...]",
            help=(
                "Resolve upstream teacher(s) via MLflow → inject ckpt path TLA, "
                "and (when upstream is RUNNING) add its slurm_job_id as an "
                "afterok dep. FINISHED upstream → TLA only; RUNNING → TLA + "
                "afterok; missing/FAILED/KILLED → hard error. ':seed' defaults "
                "to --seed if omitted. Single primitive — see "
                ".claude/rules/single-submission-primitive.md."
            ),
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help=(
                "<group>/<variant> identity (e.g. 'unsupervised/vgae'). "
                "Default: inferred from preset path "
                "configs/ablations/<group>/<variant>.jsonnet. Required when "
                "preset is off-convention and --skip-if-finished is set."
            ),
        ),
    ] = None,
    skip_if_finished: Annotated[
        bool,
        typer.Option(
            "--skip-if-finished",
            help=(
                "Query MLflow before submitting; if latest run for "
                "(dataset, group, variant, seed, phase=fit|test) is FINISHED, "
                "print 0 to stdout and exit 0 without submitting. Plans "
                "use this per-node for reentrant `graphids run`."
            ),
        ),
    ] = False,
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
    from graphids.slurm.submit import submit

    jid = submit(
        preset=preset,
        command=command,
        action=action,
        mode=mode,
        length=length,
        smoke=smoke,
        cpu=cpu,
        cluster=cluster,
        dataset=dataset,
        seed=seed,
        scale=scale,
        ckpt_tla=ckpt_tla,
        ckpt_path=ckpt_path,
        lake_root=lake_root,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
        time_from_history=time_from_history,
        tla=tla,
        set_=set_,
        depends_on=depends_on,
        name=name,
        skip_if_finished=skip_if_finished,
        dry_run=dry_run,
    )

    # Stdout contract: jid on submit, 0 on skip/dry-run (non-chaining sentinel
    # so ``afterok:$jid`` doesn't reference a real job).
    print(jid if jid is not None else 0)
    sys.stdout.flush()
