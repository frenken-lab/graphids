"""Config layer tests: dataset catalog only.

Topology/pipeline-config tests were deleted when the pipeline route
collapsed into the single ablation-preset route (2026-04-15). Pydantic
built-ins are tested upstream and intentionally absent here.
"""

from __future__ import annotations

from graphids.config.catalog import dataset_names, load_catalog


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
