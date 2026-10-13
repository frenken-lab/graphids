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

    from graphids.core.data.snapshots.manifest import audit_catalog
    from graphids.paths import lake_root, load_catalog

    raw_root = raw_root or Path(lake_root()) / "raw"
    report = audit_catalog(raw_root, load_catalog())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    typer.echo(
        f"Wrote {output}; provenance review and explicit recording roles are required before fitting."
    )
