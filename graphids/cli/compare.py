"""``python -m graphids compare`` — cross-variant MLflow comparison tables.

Thin Typer wrapper over :mod:`graphids.analysis.compare`. Each subcommand
prints a markdown table to stdout; pipe to a file for paper tables.
"""

from __future__ import annotations

from typing import Annotated

import typer

from .app import app

_compare = typer.Typer(
    name="compare",
    help="Cross-variant leaderboards, tie candidates, effect sizes, and expected-max curves.",
    no_args_is_help=True,
)
app.add_typer(_compare, name="compare", rich_help_panel="Analysis")


@_compare.command("leaderboard")
def leaderboard_cmd(
    group: Annotated[str, typer.Argument(help="Ablation group (e.g. conv_type)")],
    dataset: Annotated[str, typer.Argument(help="Dataset (e.g. set_01)")],
    metric: Annotated[str, typer.Option("--metric", "-m", help="Primary metric")] = "f1_macro",
) -> None:
    """Mean + 95 % bootstrap BCa CI per variant, sorted desc by mean."""
    from graphids.analysis.compare import leaderboard

    df = leaderboard(group, dataset, metric=metric)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


@_compare.command("ties")
def ties_cmd(
    group: Annotated[str, typer.Argument()],
    dataset: Annotated[str, typer.Argument()],
    metric: Annotated[str, typer.Option("--metric", "-m")] = "f1_macro",
    tol: Annotated[float, typer.Option("--tol", help="Gap tolerance")] = 0.005,
) -> None:
    """Variant pairs within ``tol`` on mean — flag for promote-to-N=3."""
    from graphids.analysis.compare import tie_candidates

    df = tie_candidates(group, dataset, metric=metric, tol=tol)
    if df.empty:
        print(f"(no pairs within tol={tol} for metric={metric!r})")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


@_compare.command("effect-size")
def effect_size_cmd(
    group: Annotated[str, typer.Argument()],
    dataset: Annotated[str, typer.Argument()],
    metric: Annotated[str, typer.Option("--metric", "-m")] = "f1_macro",
) -> None:
    """Pairwise Cohen's d + mean-difference bootstrap CI (no p-values)."""
    from graphids.analysis.compare import effect_size

    df = effect_size(group, dataset, metric=metric)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


@_compare.command("expected-max")
def expected_max_cmd(
    group: Annotated[str, typer.Argument()],
    dataset: Annotated[str, typer.Argument()],
    metric: Annotated[str, typer.Option("--metric", "-m")] = "f1_macro",
) -> None:
    """Dodge 2019 expected-max curve per variant, truncated at N."""
    from graphids.analysis.compare import expected_max

    df = expected_max(group, dataset, metric=metric)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
