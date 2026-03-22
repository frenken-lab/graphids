"""Config layer tests: resolve, presets, overrides, constants."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


def test_resolve_defaults():
    from graphids.config import resolve
    cfg = resolve()
    assert cfg.model_type == "vgae"
    assert cfg.dataset == "hcrl_sa"
    assert cfg.seed == 42


def test_cli_overrides():
    from graphids.config import resolve
    cfg = resolve("model_type=gat", "scale=small", "training.lr=0.01")
    assert cfg.model_type == "gat"
    assert cfg.training.lr == 0.01


def test_preset_merge():
    """Model preset from models.yaml overrides dataclass defaults."""
    from graphids.config import resolve
    cfg = resolve("model_type=vgae", "scale=large")
    assert cfg.training.lr == 0.002  # from models.yaml, not dataclass default
    assert cfg.vgae.proj_dim == 48


def test_cli_beats_preset():
    from graphids.config import resolve
    cfg = resolve("model_type=vgae", "scale=large", "training.lr=0.999")
    assert cfg.training.lr == 0.999


def test_serializable():
    """Config → dict for MLflow/hparams."""
    from graphids.config import resolve
    container = OmegaConf.to_container(resolve(), resolve=True)
    assert isinstance(container, dict)


def test_stages_and_dependencies():
    """pipeline.yaml parsed: stages exist, DAG is valid."""
    from graphids.config.constants import STAGES, STAGE_DEPENDENCIES
    assert "autoencoder" in STAGES
    assert "fusion" in STAGES
    deps = STAGE_DEPENDENCIES["fusion"]
    assert ("vgae", "autoencoder") in deps
    assert ("gat", "curriculum") in deps


def test_new_config_fields():
    """Parameterized fields exist with correct defaults."""
    from graphids.config import resolve
    cfg = resolve()
    assert cfg.num_classes == 2
    assert cfg.fusion.decision_threshold == 0.5
    assert cfg.dqn.reward_correct == 3.0
    assert cfg.evaluation.batch_size == 256
