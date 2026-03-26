"""Pipeline stage dispatch tests."""

from __future__ import annotations

import pytest


def test_unknown_stage_raises():
    from graphids.config import resolve
    from graphids.pipeline.stages import run_stage
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage(resolve(), "nonexistent")


def test_all_stages_have_functions():
    from graphids.config import STAGES
    from graphids.pipeline.stages.runner import STAGE_FNS
    missing = [s for s in STAGES if s != "preprocess" and s not in STAGE_FNS]
    assert not missing, f"Missing stage functions: {missing}"
