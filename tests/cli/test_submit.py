"""Tests for ``graphids/slurm/submit.py:_infer_group_variant`` — preset-path inference.

Used by ``--skip-if-finished`` to feed the MLflow filter. If this returns
the wrong ``(group, variant)`` we either (a) silently skip the wrong run,
or (b) submit a job that should have been skipped — both are correctness
bugs not caught by Typer's own parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from graphids.slurm.submit import _infer_group_variant


# CONTRACT: the canonical layout
# (configs/ablations/<group>/<variant>.jsonnet) resolves both fields from
# the path. Inference is the default code path for every preset referenced
# in configs/plans/ofat.jsonnet.
def test_infer_group_variant_from_convention_path():
    p = Path("/x/configs/ablations/unsupervised/vgae.jsonnet")
    assert _infer_group_variant(p, name=None) == ("unsupervised", "vgae")


# CONTRACT: ``--name`` wins over path inference. Used when the preset
# lives outside ``configs/ablations/`` (smoke / one-off / stage tests).
def test_infer_group_variant_name_overrides_path():
    p = Path("/x/configs/ablations/A/B.jsonnet")
    assert _infer_group_variant(p, name="C/D") == ("C", "D")


# REGRESSION risk: stage files (``configs/stages/foo.jsonnet``) don't
# match the convention. Without this guard, a naive ``parts[-2]``-style
# inference would return ``('configs', 'autoencoder')`` and the MLflow
# lookup would silently miss every row, defeating ``--skip-if-finished``.
def test_infer_group_variant_off_convention_path_raises():
    p = Path("/x/configs/stages/autoencoder.jsonnet")
    with pytest.raises(typer.BadParameter):
        _infer_group_variant(p, name=None)


# REGRESSION risk: ``--name bare`` (no slash) would, with naive
# ``str.partition``, give ``group='bare', variant=''`` and silently
# query MLflow with a blank variant filter.
def test_infer_group_variant_name_without_slash_raises():
    p = Path("/x/configs/ablations/A/B.jsonnet")
    with pytest.raises(typer.BadParameter):
        _infer_group_variant(p, name="bare")
