"""CAN compatibility inventory without model imports or split assignment."""

from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app

can_app = typer.Typer(
    help="Audit CAN recording compatibility before constructing a reviewed manifest."
)
app.add_typer(can_app, name="can")


@can_app.command("audit")
def audit(
    output: Annotated[Path, typer.Option(help="JSON compatibility report destination")],
    raw_root: Annotated[
        Path | None,
        typer.Option(help="Raw dataset root; defaults to the lake raw directory"),
    ] = None,
):
    import json

    from graphids.core.data.recordings import audit_catalog
    from graphids.paths import lake_root, load_catalog

    raw_root = raw_root or Path(lake_root()) / "raw"
    report = audit_catalog(raw_root, load_catalog())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    typer.echo(
        f"Wrote {output}; provenance review and explicit recording roles are required before fitting."
    )


@can_app.command("verify-events")
def verify_events(
    config: Annotated[Path, typer.Option(help="Event experiment YAML")],
    output: Annotated[Path, typer.Option(help="Verification JSON destination")],
):
    """Verify every source frame against its reviewed CSV, in a Slurm allocation."""
    from graphids.core.data.events.verify import verify_events as verify
    result = verify(config, output)
    typer.echo(f"Verified {result['frames']} source frames; wrote {output}")
