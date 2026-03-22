"""Config layer tests: Hydra compose, CLI overrides, model preset merge."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# resolve() basics
# ---------------------------------------------------------------------------


def test_resolve_defaults():
    """resolve() with no overrides returns a valid config with defaults."""
    from graphids.config import resolve

    cfg = resolve()
    assert cfg.model_type == "vgae"
    assert cfg.scale == "large"
    assert cfg.stage == "autoencoder"
    assert cfg.dataset == "hcrl_sa"
    assert cfg.seed == 42


def test_resolve_cli_overrides():
    """CLI-style dotlist overrides take precedence over defaults."""
    from graphids.config import resolve

    cfg = resolve("model_type=gat", "scale=small", "training.lr=0.01")
    assert cfg.model_type == "gat"
    assert cfg.scale == "small"
    assert cfg.training.lr == 0.01


def test_resolve_lake_root_default():
    """lake_root resolves to 'experimentruns' when env var is unset."""
    import os

    from graphids.config import resolve

    # Clear env var if set, then resolve
    old = os.environ.pop("KD_GAT_LAKE_ROOT", None)
    try:
        cfg = resolve()
        assert cfg.lake_root == "experimentruns"
    finally:
        if old is not None:
            os.environ["KD_GAT_LAKE_ROOT"] = old


# ---------------------------------------------------------------------------
# Model preset merge
# ---------------------------------------------------------------------------


def test_model_preset_vgae_large():
    """vgae_large preset overrides training.lr and vgae.proj_dim."""
    from graphids.config import resolve

    cfg = resolve("model_type=vgae", "scale=large")
    # From models.yaml: vgae_large overrides lr to 0.002 and proj_dim to 48
    assert cfg.training.lr == 0.002
    assert cfg.vgae.proj_dim == 48


def test_model_preset_gat_small():
    """gat_small preset merges GAT architecture overrides."""
    from graphids.config import resolve

    cfg = resolve("model_type=gat", "scale=small")
    assert cfg.gat.hidden == 24
    assert cfg.gat.layers == 2
    assert cfg.gat.heads == 4
    assert cfg.training.lr == 0.001


def test_model_preset_dqn_small():
    """dqn_small preset overrides DQN architecture params."""
    from graphids.config import resolve

    cfg = resolve("model_type=dqn", "scale=small")
    assert cfg.dqn.hidden == 160
    assert cfg.dqn.layers == 2


def test_cli_overrides_beat_preset():
    """CLI overrides take priority over model preset values."""
    from graphids.config import resolve

    cfg = resolve("model_type=vgae", "scale=large", "training.lr=0.999")
    assert cfg.training.lr == 0.999


def test_nonexistent_preset_uses_defaults():
    """Unknown model_type+scale combo falls back to dataclass defaults."""
    from graphids.config import resolve

    # dqn_large has minimal overrides — check that defaults are intact
    cfg = resolve("model_type=dqn", "scale=large")
    assert cfg.dqn.layers == 3  # dataclass default, not overridden


# ---------------------------------------------------------------------------
# Structured config schema
# ---------------------------------------------------------------------------


def test_config_schema_types():
    """Config fields have correct types after resolution."""
    from graphids.config import resolve

    cfg = resolve()
    assert isinstance(cfg.training.lr, float)
    assert isinstance(cfg.training.max_epochs, int)
    assert isinstance(cfg.training.batch_size, int)
    assert OmegaConf.is_list(cfg.vgae.hidden_dims)
    assert isinstance(cfg.seed, int)


def test_config_to_container():
    """Config can be serialized to a plain dict (needed for MLflow/hparams)."""
    from graphids.config import resolve

    cfg = resolve()
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    assert "training" in container
    assert "vgae" in container


# ---------------------------------------------------------------------------
# Constants from pipeline.yaml
# ---------------------------------------------------------------------------


def test_stages_loaded():
    """STAGES dict is populated from pipeline.yaml."""
    from graphids.config.constants import STAGES

    assert "autoencoder" in STAGES
    assert "curriculum" in STAGES
    assert "fusion" in STAGES
    assert "evaluation" in STAGES


def test_valid_model_types():
    """VALID_MODEL_TYPES includes the three core models."""
    from graphids.config.constants import VALID_MODEL_TYPES

    assert "vgae" in VALID_MODEL_TYPES
    assert "gat" in VALID_MODEL_TYPES
    assert "dqn" in VALID_MODEL_TYPES


def test_valid_scales():
    """VALID_SCALES includes large and small."""
    from graphids.config.constants import VALID_SCALES

    assert "large" in VALID_SCALES
    assert "small" in VALID_SCALES


def test_stage_dependencies():
    """curriculum depends on autoencoder, fusion depends on both."""
    from graphids.config.constants import STAGE_DEPENDENCIES

    # curriculum depends on vgae autoencoder
    assert ("vgae", "autoencoder") in STAGE_DEPENDENCIES["curriculum"]
    # fusion depends on vgae autoencoder AND gat curriculum
    fusion_deps = STAGE_DEPENDENCIES["fusion"]
    assert ("vgae", "autoencoder") in fusion_deps
    assert ("gat", "curriculum") in fusion_deps


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_data_dir_fallback():
    """data_dir falls back to data/automotive/{dataset} when lake path missing."""
    from pathlib import Path

    from graphids.config import data_dir

    result = data_dir("nonexistent_lake_root", "hcrl_sa")
    assert result == Path("data") / "automotive" / "hcrl_sa"


def test_cache_dir_structure():
    """cache_dir produces versioned path under lake_root."""
    from graphids.config import cache_dir
    from graphids.config.constants import PREPROCESSING_VERSION

    result = cache_dir("my_lake", "hcrl_sa")
    assert f"v{PREPROCESSING_VERSION}" in str(result)
    assert "hcrl_sa" in str(result)
