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


def _infer_group_variant(preset: Path, name: str | None) -> tuple[str, str]:
    """Resolve ``(group, variant)`` from ``--name`` or the preset path convention.

    Convention: ``configs/ablations/<group>/<variant>.jsonnet``. ``--name``
    overrides the convention as ``"group/variant"``. Raises
    :class:`typer.BadParameter` when neither resolves — used by
    ``--skip-if-finished`` to feed the MLflow filter.
    """
    if name:
        if "/" not in name:
            raise typer.BadParameter(f"--name must be 'group/variant' (got {name!r})")
        group, _, variant = name.partition("/")
        return group, variant
    parts = preset.parts
    if "ablations" in parts:
        idx = parts.index("ablations")
        if idx + 2 < len(parts):
            return parts[idx + 1], preset.stem
    raise typer.BadParameter(
        f"--skip-if-finished cannot infer group/variant from {preset}. "
        "Pass --name <group>/<variant> explicitly."
    )


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

    if preset is None and not command:
        raise typer.BadParameter('supply a preset path or --command "..."')

    # --depends-on (MLflow-resolved upstream ckpts) and --ckpt-path
    # (resume the current preset) look similar but mean different things.
    # Refuse to guess.
    if depends_on and ckpt_path:
        raise typer.BadParameter(
            "--ckpt-path resumes the *current* preset; --depends-on injects "
            "upstream teacher ckpts. Different semantics — pass them on "
            "separate invocations."
        )

    # Flat flags → TLA pairs. --ckpt-tla writes the ``ckpt_path`` TLA
    # (jsonnet field), distinct from --ckpt-path (fit/test passthrough).
    flag_tlas: list[tuple[str, object]] = []
    for key, val in (
        ("dataset", dataset),
        ("scale", scale),
        ("ckpt_path", ckpt_tla),
        ("lake_root", lake_root),
    ):
        if val:
            flag_tlas.append((key, val))
    if seed is not None:
        flag_tlas.append(("seed", seed))

    # Resolve --depends-on BEFORE user --tla so explicit --tla overrides
    # (last-wins on flag_tlas). Hard error on resolution failure: deps are
    # load-bearing — silently continuing with no teacher TLAs would render
    # but produce a wrong run. RUNNING upstreams contribute afterok jids.
    afterok_jids: list[int] = []
    if depends_on:
        from graphids.slurm.dependencies import (
            DependencyResolutionError,
            parse_depends_on,
            resolve_all,
        )

        if not dataset:
            raise typer.BadParameter("--depends-on requires --dataset")
        try:
            specs = parse_depends_on(depends_on, default_seed=seed)
            dep_tlas, afterok_jids = resolve_all(specs, dataset)
            flag_tlas.extend(dep_tlas)
        except DependencyResolutionError as exc:
            raise typer.BadParameter(str(exc)) from exc

    flag_tlas.extend(tla or ())

    if skip_if_finished:
        if preset is None:
            raise typer.BadParameter("--skip-if-finished requires a preset (no --command form)")
        group, variant = _infer_group_variant(preset, name)
        ds = next((v for k, v in flag_tlas if k == "dataset"), None)
        sd = next((v for k, v in flag_tlas if k == "seed"), None)
        if not ds or sd is None:
            raise typer.BadParameter(
                "--skip-if-finished needs both --dataset and --seed so the MLflow lookup is unambiguous"
            )
        from graphids._mlflow import is_finished

        phase = "test" if action == "test" else "fit"
        if is_finished(dataset=str(ds), group=group, variant=variant, seed=int(sd), phase=phase):
            print(0)
            sys.stdout.flush()
            return

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
        dep_jids=tuple(afterok_jids),
        dry_run=dry_run,
    )

    # Stdout contract: print the (fit) jid only — bash callers do
    # ``jid=$(graphids submit ...)``. Dry-run / skip-if-finished emit 0 as a
    # non-chaining sentinel so ``afterok:$jid`` doesn't reference a real job.
    print(jid if jid is not None else 0)
    sys.stdout.flush()
