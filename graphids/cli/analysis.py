"""Analysis command: generate artifacts from a trained checkpoint.

Auto-dispatches by model type — derived from the checkpoint's
self-describing ``class_path``. Per-model artifact toggles live in
``core/analysis/schemas.ARTIFACTS_BY_MODEL_TYPE``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import _complete_dataset, app


@app.command(rich_help_panel="Analysis")
def analyze(
    ckpt_path: Annotated[
        Path,
        typer.Option(
            "--ckpt-path",
            exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True,
            help="Checkpoint file (class_path read for model-type dispatch)",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset name", autocompletion=_complete_dataset),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Output dir (defaults to {ckpt.parent.parent}/artifacts)",
        ),
    ] = None,
    vgae_ckpt: Annotated[
        Path | None,
        typer.Option("--vgae-ckpt", help="Upstream VGAE ckpt (required for fusion models)"),
    ] = None,
    gat_ckpt: Annotated[
        Path | None,
        typer.Option("--gat-ckpt", help="Upstream GAT ckpt (required for fusion models)"),
    ] = None,
    cka_teacher_ckpt: Annotated[
        Path | None,
        typer.Option("--cka-teacher-ckpt", help="Teacher ckpt for GAT CKA analysis"),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Run all artifacts applicable to the checkpoint's model type."""
    from graphids.core.analysis.runner import analysis_spec_for, run_single_analysis
    from graphids.core.analysis.schemas import derive_model_type

    model_type = derive_model_type(ckpt_path)

    # Fusion needs upstream checkpoints to unlock fusion_policy. Pass them
    # through the same ``upstream_ckpts`` + ``upstream_families`` contract
    # the pipeline uses so both call paths hit ``analysis_spec_for`` the
    # same way.
    upstream_ckpts: dict[str, str] | None = None
    upstream_families: dict[str, str] | None = None
    if model_type == "fusion":
        if not (vgae_ckpt and gat_ckpt):
            raise typer.BadParameter(
                "fusion checkpoints require --vgae-ckpt and --gat-ckpt for fusion_policy"
            )
        upstream_ckpts = {"vgae": str(vgae_ckpt), "gat": str(gat_ckpt)}
        upstream_families = {"vgae": "unsupervised", "gat": "supervised"}

    spec = analysis_spec_for(
        ckpt_path, dataset=dataset, model_type=model_type, seed=seed,
        upstream_ckpts=upstream_ckpts, upstream_families=upstream_families,
    )
    if output_dir is not None:
        spec = spec.model_copy(update={"output_dir": str(output_dir)})
    if cka_teacher_ckpt is not None:
        spec = spec.model_copy(update={"cka_teacher_ckpt": str(cka_teacher_ckpt)})

    # CKA requires teacher ckpt; surface the contract early with a clean
    # error rather than deep inside Analyzer.__init__ FileNotFoundError.
    if spec.cka and not spec.cka_teacher_ckpt:
        raise typer.BadParameter(
            "GAT CKA analysis requires --cka-teacher-ckpt (or disable cka in the dispatch table)"
        )

    run_single_analysis(spec)
