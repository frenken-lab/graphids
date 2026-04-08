"""SLURM commands: submit-profile, probe-budget."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("submit-profile", rich_help_panel="SLURM")
def submit_profile(
    job: Annotated[str | None, typer.Argument(help="Job profile name")] = None,
) -> None:
    """Print SLURM resource profile for scripts/slurm/submit.sh."""
    from graphids.slurm.resources import print_submit_profile

    print_submit_profile(job)


@app.command("probe-budget", rich_help_panel="SLURM")
def probe_budget(
    dataset: Annotated[list[str] | None, typer.Option(help="Dataset(s) to probe")] = None,
    model_type: Annotated[list[str] | None, typer.Option(help="Model type(s) to probe")] = None,
    scale: Annotated[list[str] | None, typer.Option(help="Scale(s) to probe")] = None,
    conv_type: Annotated[
        list[str] | None,
        typer.Option(
            help="Conv type(s) to probe (e.g. gatv2, gps). Default: model's jsonnet default"
        ),
    ] = None,
    lake_root: Annotated[str | None, typer.Option(help="Lake root path")] = None,
    json_: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
    dry_run: Annotated[bool, typer.Option(help="Print plan without probing")] = False,
) -> None:
    """Measure hardware cost model (VRAM probe + calibration). Requires GPU.

    Sweeps (model_type × scale × conv_type × dataset). Use --conv-type to
    compare O(E) sparse convs (gatv2, gat, transformer) vs O(N²) dense
    attention (gps).
    """
    from graphids.config.constants import LAKE_ROOT, VALID_MODEL_TYPES, VALID_SCALES
    from graphids.core.data.budget_probe import run_probe_budget

    run_probe_budget(
        model_types=model_type or sorted(VALID_MODEL_TYPES),
        scales=scale or sorted(VALID_SCALES),
        conv_types=conv_type,
        datasets=dataset,
        lake_root=lake_root or LAKE_ROOT,
        json_output=json_,
        dry_run=dry_run,
    )
