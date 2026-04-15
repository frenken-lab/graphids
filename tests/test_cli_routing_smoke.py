"""Smoke tests for Typer CLI routing.

Verifies that commands are registered and --help works without importing torch.
"""

from typer.testing import CliRunner

# Register all command modules
import graphids.cli.analysis  # noqa: F401
import graphids.cli.data  # noqa: F401
import graphids.cli.training  # noqa: F401
from graphids.cli.app import app

runner = CliRunner()


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Training" in result.output
    assert "Analysis" in result.output


def test_fit_requires_config() -> None:
    result = runner.invoke(app, ["fit"])
    assert result.exit_code != 0


def test_fit_help() -> None:
    result = runner.invoke(app, ["fit", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--tla" in result.output
    assert "--set" in result.output


def test_analyze_help() -> None:
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--ckpt-path" in result.output
    assert "--dataset" in result.output


def test_all_expected_commands_registered() -> None:
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    expected = {
        "fit",
        "test",
        "analyze",
        "rebuild-caches",
        "extract-fusion-states",
    }
    missing = expected - names
    assert not missing, f"Missing commands: {missing}"
