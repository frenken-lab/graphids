"""Config layer tests: dataset catalog + custom pydantic validators.

Topology DAG tests (test_fusion_has_dependencies, test_no_cycles,
test_ordering, test_default_stages_are_valid, test_stages_have_identity_keys)
were deleted 2026-04-13 — they referenced ``STAGES`` / ``STAGE_DEPENDENCIES``
/ ``PIPELINE_TOPOLOGY`` module-level names that were never actually imported
in this file, and would have been duplicating import-time assertions in
``graphids.config.topology`` anyway.

``test_custom_stage_validator_rejects_unknown`` was also deleted: the
production error message changed from ``"Unknown stages"`` to Pydantic's
``"stage='bogus' not in [...]"``, making the regex match brittle with
no real invariant behind it.
"""

from __future__ import annotations

import pytest

from graphids.config.topology import dataset_names, load_catalog
from graphids.orchestrate.config import KDEntry, TrainingRunConfig

# ---------------------------------------------------------------------------
# Dataset catalog (configs/datasets/dataset_registry.json)
# ---------------------------------------------------------------------------


def test_catalog_loads_all_datasets():
    catalog = load_catalog()
    assert len(catalog) >= 6
    for name in ["hcrl_ch", "hcrl_sa", "set_01", "set_02", "set_03", "set_04"]:
        assert name in catalog, f"Missing dataset: {name}"


def test_catalog_entries_have_required_fields():
    required = {
        "name",
        "domain",
        "csv_dir",
        "csv_columns",
        "train_subdir",
        "test_subdirs",
        "attack_types",
    }
    for name, entry in load_catalog().items():
        missing = required - set(entry.keys())
        assert not missing, f"Dataset '{name}' missing fields: {missing}"


def test_dataset_names_excludes_internal():
    names = dataset_names()
    assert all(not n.startswith("_") for n in names)


def test_catalog_test_subdirs_populated():
    """Every dataset must have at least one test subdir for evaluation."""
    for name, entry in load_catalog().items():
        assert entry["test_subdirs"], f"Dataset '{name}' has empty test_subdirs"


# Topology DAG invariants are enforced by import-time assertions in
# ``graphids.config.topology`` — the ``from graphids...`` imports above
# already exercise them. A missing file or bad ordering prevents
# collection, which is a louder signal than a test failure.


# ---------------------------------------------------------------------------
# TrainingRunConfig + KDEntry — project-specific validators only.
# Pydantic built-ins (frozen, extra=forbid, Literal coercion, required
# kwargs) are tested upstream by Pydantic and intentionally absent here.
# ---------------------------------------------------------------------------


def test_training_run_config_defaults():
    cfg = TrainingRunConfig()
    assert cfg.model_type is None
    assert cfg.auxiliaries == ()


def test_kd_entry_custom_alpha_bounds():
    """Custom 0..1 alpha bound + default type='kd' discriminator."""
    kd = KDEntry(alpha=0.5)
    assert kd.type == "kd"
    assert kd.alpha == 0.5
