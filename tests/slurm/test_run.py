"""Tests for ``graphids/slurm/run.py`` — bash-artifact renderer.

The renderer is a pure function: ``(nodes, dataset, seed, cluster,
skip_finished) → bash string``. Tests assert on the rendered string;
no submission is exercised. A ``bash -n`` parse-only check guards against
syntax-broken artifacts shipping silently.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from graphids.config.jsonnet import render
from graphids.slurm.dag import Node, parse_plan
from graphids.slurm.run import _var_for, render_plan_script


@pytest.fixture(scope="module")
def ofat_nodes() -> tuple[Node, ...]:
    return parse_plan(render("configs/plans/ofat.jsonnet", tla={"dataset": "set_01", "seed": 42}))


@pytest.fixture(scope="module")
def ofat_script(ofat_nodes: tuple[Node, ...]) -> str:
    return render_plan_script(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")


# CONTRACT: ``_var_for`` must produce a bash-safe identifier for any Node name.
# REGRESSION: kebab-case node names (``vgae-test``, ``extract-states``) blew up
# bash assignment if not converted; assert the conversion explicitly.
def test_var_for_kebab_to_snake() -> None:
    assert _var_for("vgae") == "JID_VGAE"
    assert _var_for("vgae-test") == "JID_VGAE_TEST"
    assert _var_for("extract-states") == "JID_EXTRACT_STATES"


# CONTRACT: artifact starts with the bash shebang + strict-mode pragma.
def test_emits_shebang_and_strict_mode(ofat_script: str) -> None:
    lines = ofat_script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "set -euo pipefail" in lines[:5]


# CONTRACT: header records the invocation and a content-addressed plan_hash.
def test_header_has_plan_hash(ofat_script: str) -> None:
    head = ofat_script.splitlines()[:4]
    hash_line = next(line for line in head if "plan_hash=" in line)
    assert "31 nodes" in hash_line  # 15 fits + 15 tests + 1 command
    # 8-char hex digest
    digest = hash_line.split("plan_hash=")[1]
    assert len(digest) == 8 and all(c in "0123456789abcdef" for c in digest)


# CONTRACT: every preset line bakes (dataset, seed, cluster) explicitly.
# REGRESSION risk: forgetting `--cluster` would route every job to the env-var
# fallback ("pitzer"), silently submitting Cardinal-targeted plans to the wrong cluster.
def test_preset_lines_bake_dataset_seed_cluster(ofat_script: str) -> None:
    vgae_line = _line_for(ofat_script, "JID_VGAE=")
    assert "--dataset set_01" in vgae_line
    assert "--seed 42" in vgae_line
    assert "--cluster cardinal" in vgae_line


# CONTRACT: dep wiring uses the upstream's shell variable, not a literal jid.
def test_dep_chain_uses_shell_var(ofat_script: str) -> None:
    test_line = _line_for(ofat_script, "JID_VGAE_TEST=")
    assert '--dep "$JID_VGAE"' in test_line


# CONTRACT: command-mode nodes use --command, NOT a preset path, and reference
# both upstream deps (extract-states fans in vgae + focal).
def test_command_node_uses_command_form(ofat_script: str) -> None:
    line = _line_for(ofat_script, "JID_EXTRACT_STATES=")
    assert "--command " in line
    assert "extract-fusion-states" in line
    assert '--dep "$JID_VGAE"' in line
    assert '--dep "$JID_FOCAL"' in line


# CONTRACT: command-mode nodes do NOT get --skip-if-finished — they have no
# (group, variant) for the MLflow lookup.
def test_command_node_omits_skip_if_finished(ofat_script: str) -> None:
    line = _line_for(ofat_script, "JID_EXTRACT_STATES=")
    assert "--skip-if-finished" not in line


# CONTRACT: every preset line has --skip-if-finished by default; --force omits it.
def test_skip_if_finished_default_on(ofat_nodes: tuple[Node, ...]) -> None:
    script = render_plan_script(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    preset_lines = [l for l in script.splitlines() if "graphids submit configs/" in l]
    assert all("--skip-if-finished" in script.split(l)[1].split("\n\n")[0] for l in preset_lines)


def test_force_omits_skip_if_finished(ofat_nodes: tuple[Node, ...]) -> None:
    script = render_plan_script(
        ofat_nodes, dataset="set_01", seed=42, cluster="cardinal", skip_finished=False
    )
    assert "--skip-if-finished" not in script


# CONTRACT: test-action peers get `--action test` and the test-resource overrides.
def test_test_peer_has_test_action_and_cpu_overrides(ofat_script: str) -> None:
    line = _line_for(ofat_script, "JID_VGAE_TEST=")
    assert "--action test" in line
    assert "--mode cpu" in line
    assert "--mem-gb 32" in line
    assert "--timeout-min 30" in line


# CONTRACT: render is deterministic. Same inputs → byte-identical output.
def test_render_deterministic(ofat_nodes: tuple[Node, ...]) -> None:
    a = render_plan_script(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    b = render_plan_script(ofat_nodes, dataset="set_01", seed=42, cluster="cardinal")
    assert a == b


# CONTRACT: artifact is syntactically valid bash. Catches any quoting / line-
# continuation bug before the user ever pipes the script into a shell.
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_artifact_parses_under_bash_n(ofat_script: str) -> None:
    result = subprocess.run(
        ["bash", "-n"], input=ofat_script, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, f"bash -n rejected the script:\n{result.stderr}"


def _line_for(script: str, prefix: str) -> str:
    """Return the full multi-line `JID_X=$(...)` block whose first line starts with prefix."""
    blocks = script.split("\n\n")
    for block in blocks:
        if block.lstrip().startswith(prefix):
            return block
    raise AssertionError(f"no block starting with {prefix!r} in script")
