"""Campaign CLI — declared ablations over the existing OTel trace log.

Subcommands: ``status`` (derives from traces.jsonl spans), ``next``
(prints / ``--exec`` submits the next pending cell), ``freeze``,
``verify``. No parallel status log — OTel tags spans with
``campaign.cell_id``; :func:`cell_statuses` reads them back.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app

campaign_app = typer.Typer(
    name="campaign",
    help="Campaign manifest — declared ablations over the OTel trace log.",
    no_args_is_help=True,
)
app.add_typer(campaign_app, name="campaign", rich_help_panel="Orchestration")


ManifestPath = Annotated[
    Path,
    typer.Argument(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
        help="Path to the campaign <name>.yaml manifest.",
    ),
]

_COLORS = {
    "pending": typer.colors.WHITE,
    "running": typer.colors.YELLOW,
    "completed": typer.colors.GREEN,
    "failed": typer.colors.RED,
}


def _lake_root() -> Path:
    return Path(os.environ.get("GRAPHIDS_LAKE_ROOT", "experimentruns")).resolve()


@campaign_app.command("status")
def status_cmd(manifest: ManifestPath) -> None:
    """Print per-cell status derived from traces.jsonl spans."""
    from graphids.campaigns.manifest import cell_statuses, load_campaign

    campaign = load_campaign(manifest)
    statuses = cell_statuses(campaign, manifest_path=manifest, lake_root=_lake_root())

    header = f"campaign: {campaign.name}  (frozen: {campaign.is_frozen})"
    typer.echo(header)
    typer.echo("-" * len(header))

    width = max((len(c.id) for c in campaign.cells), default=8)
    counters: dict[str, int] = {}
    for cell in campaign.cells:
        state = statuses.get(cell.id, "pending")
        counters[state] = counters.get(state, 0) + 1
        typer.echo(
            f"  {cell.id:<{width}}  "
            f"{typer.style(state, fg=_COLORS.get(state, typer.colors.WHITE))}"
        )
    typer.echo("")
    typer.echo("  ".join(f"{k}={v}" for k, v in counters.items()) or "no cells")


@campaign_app.command("next")
def next_cmd(
    manifest: ManifestPath,
    cell: Annotated[
        str | None, typer.Option("--cell", help="Explicit cell id override.")
    ] = None,
    include_failed: Annotated[
        bool, typer.Option("--include-failed", help="Include 'failed' in auto-selection.")
    ] = False,
    exec_: Annotated[
        bool, typer.Option("--exec", help="Submit via scripts/slurm/submit.sh.")
    ] = False,
    submit_profile: Annotated[
        str, typer.Option("--submit-profile", help="submit.sh profile name.")
    ] = "pipeline-run",
) -> None:
    """Pick the next pending cell; print or (--exec) submit."""
    from graphids.campaigns.manifest import cell_statuses, load_campaign

    campaign = load_campaign(manifest)
    if not campaign.is_frozen:
        typer.secho(
            f"campaign {campaign.name!r} is a draft — run `graphids campaign freeze` first",
            fg=typer.colors.YELLOW, err=True,
        )

    if cell is not None:
        picked = campaign.get_cell(cell)
    else:
        statuses = cell_statuses(
            campaign, manifest_path=manifest, lake_root=_lake_root()
        )
        eligible_states = {"pending"} | ({"failed"} if include_failed else set())
        picked = next(
            (c for c in campaign.cells
             if statuses.get(c.id, "pending") in eligible_states),
            None,
        )
        if picked is None:
            typer.secho("no eligible cells — all done or all failed", fg=typer.colors.GREEN)
            raise typer.Exit()

    argv = _submit_argv(submit_profile, campaign.merged_config(picked.id))
    if not exec_:
        typer.echo(f"# cell: {picked.id}")
        typer.echo(f"# export GRAPHIDS_CAMPAIGN_CELL={manifest}::{picked.id}")
        typer.echo(" ".join(argv))
        return

    env = {**os.environ, "GRAPHIDS_CAMPAIGN_CELL": f"{manifest}::{picked.id}"}
    typer.secho(f"submitting cell {picked.id!r}…", fg=typer.colors.CYAN)
    raise typer.Exit(subprocess.run(argv, env=env, check=False).returncode)


def _submit_argv(profile: str, merged) -> list[str]:
    argv = [
        "scripts/slurm/submit.sh", profile,
        "--dataset", merged.dataset,
        "--seed", str(merged.seed),
        "--scale", merged.scale,
        "--fusion-method", merged.fusion_method,
        "--stages", ",".join(merged.stages),
        "--conv-type", merged.conv_type,
        "--loss-fn", merged.loss_fn,
        "--variational" if merged.variational else "--no-variational",
    ]
    if merged.lake_root:
        argv += ["--lake-root", merged.lake_root]
    if merged.max_retries != 2:
        argv += ["--max-retries", str(merged.max_retries)]
    for k, v in (merged.tla_overrides or {}).items():
        argv += ["--trainer-override", f"{k}={json.dumps(v)}"]
    return argv


@campaign_app.command("freeze")
def freeze_cmd(manifest: ManifestPath) -> None:
    """Set ``frozen_at: today`` on a draft manifest (rewrites via safe_dump — comments lost)."""
    import yaml

    from graphids.campaigns.manifest import load_campaign

    campaign = load_campaign(manifest)
    if campaign.is_frozen:
        typer.secho(
            f"campaign {campaign.name!r} already frozen at {campaign.frozen_at}",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)
    if not campaign.cells:
        typer.secho("refusing to freeze an empty campaign", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    raw = yaml.safe_load(manifest.read_text())
    raw["frozen_at"] = date.today().isoformat()
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=False))
    tmp.replace(manifest)
    typer.secho(
        f"froze {campaign.name!r} at {raw['frozen_at']} ({len(campaign.cells)} cells)",
        fg=typer.colors.GREEN,
    )


@campaign_app.command("verify")
def verify_cmd(manifest: ManifestPath) -> None:
    """Re-validate every cell's merged config. Exit non-zero on any failure."""
    from graphids.campaigns.manifest import load_campaign

    campaign = load_campaign(manifest)
    failures: list[tuple[str, str]] = []
    for cell in campaign.cells:
        try:
            campaign.merged_config(cell.id)
        except Exception as exc:  # noqa: BLE001 — report all failures
            failures.append((cell.id, str(exc)))

    if failures:
        for cid, err in failures:
            typer.secho(f"  {cid}: {err}", fg=typer.colors.RED, err=True)
        typer.secho(
            f"{len(failures)}/{len(campaign.cells)} cells failed validation",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)
    typer.secho(f"{len(campaign.cells)}/{len(campaign.cells)} cells valid", fg=typer.colors.GREEN)
