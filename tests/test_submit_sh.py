"""Tests for scripts/submit.sh argument quoting and profile parsing."""

from __future__ import annotations

import subprocess

import pytest

PROJECT_ROOT = "/users/PAS2022/rf15/KD-GAT"
SUBMIT_SH = f"{PROJECT_ROOT}/scripts/submit.sh"


def _get_wrap_string(*extra_args: str) -> str:
    """Run submit.sh under bash -x and extract the --wrap value."""
    # Use 'ablation' profile since it has no signal (tests NONE sentinel too)
    cmd = ["bash", "-x", SUBMIT_SH, "ablation", *extra_args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=30,
    )
    # bash -x prints the sbatch command to stderr; find the --wrap arg
    for line in result.stderr.splitlines():
        if "--wrap=" in line:
            # Extract everything after --wrap=
            idx = line.index("--wrap=")
            return line[idx:]
    pytest.fail(f"No --wrap found in bash -x output:\n{result.stderr}")


class TestSubmitShQuoting:
    """Verify special characters survive --wrap string embedding."""

    def test_glob_star_is_escaped(self):
        wrap = _get_wrap_string("--assets", "*")
        # * must be escaped (\*) or quoted, not bare
        assert "\\*" in wrap or "'*'" in wrap, f"Bare glob * in wrap: {wrap}"

    def test_pipe_is_escaped(self):
        wrap = _get_wrap_string("--partition", "hcrl_sa|42")
        assert "\\|" in wrap or "'hcrl_sa|42'" in wrap, f"Bare pipe | in wrap: {wrap}"

    def test_spaces_are_escaped(self):
        wrap = _get_wrap_string("--tag", "my tag")
        assert "\\ " in wrap or "'my tag'" in wrap, f"Bare space in wrap: {wrap}"

    def test_no_args_produces_clean_command(self):
        wrap = _get_wrap_string()
        assert "dg launch" in wrap


class TestSubmitProfileSignal:
    """Verify NONE sentinel prevents --signal from being passed."""

    def test_no_signal_flag_for_ablation(self):
        result = subprocess.run(
            ["bash", "-x", SUBMIT_SH, "ablation"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
        )
        assert "--signal=" not in result.stderr, "sbatch got --signal for profile with no signal"
