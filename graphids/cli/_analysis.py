"""Analysis command: generate artifacts from trained checkpoints."""

from __future__ import annotations

from graphids.cli.app import ConfigPath, TlaList, app, parse_tla


@app.command(rich_help_panel="Analysis")
def analyze(
    config: ConfigPath,
    tla: TlaList = None,
) -> None:
    """Generate analysis artifacts (embeddings, attention, CKA, landscape) from a checkpoint."""
    from graphids.config.jsonnet import render_config
    from graphids.core.analysis.analyzer import Analyzer
    from graphids.core.analysis.schemas import AnalysisSpec

    tla_dict = parse_tla(tla)
    rendered = render_config(config, tla=tla_dict or None)
    spec = AnalysisSpec.model_validate(rendered)
    Analyzer(**spec.model_dump(exclude={"metadata"})).run()
