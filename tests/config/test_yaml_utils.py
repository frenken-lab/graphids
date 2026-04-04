"""Tests for graphids.config.yaml_utils — deep_merge, apply_dotted_overrides,
merge_yaml_chain. Pure dict/YAML operations; no torch, no resolver."""

from __future__ import annotations

from pathlib import Path

import yaml

from graphids.config.yaml_utils import (
    apply_dotted_overrides,
    deep_merge,
    merge_yaml_chain,
)


class TestDeepMerge:
    def test_simple_override(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_override(self):
        base = {"trainer": {"max_epochs": 300, "precision": "16-mixed"}}
        overlay = {"trainer": {"max_epochs": 2}}
        result = deep_merge(base, overlay)
        assert result["trainer"]["max_epochs"] == 2
        assert result["trainer"]["precision"] == "16-mixed"

    def test_disjoint_keys(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_empty_overlay(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_non_dict_replaces_dict(self):
        assert deep_merge({"a": {"b": 1}}, {"a": 42}) == {"a": 42}


class TestApplyDottedOverrides:
    def test_simple_dotted_key(self):
        merged = {"trainer": {"max_epochs": 300}}
        result = apply_dotted_overrides(merged, {"trainer.max_epochs": "2"})
        assert result["trainer"]["max_epochs"] == "2"

    def test_creates_intermediate_dicts(self):
        result = apply_dotted_overrides({}, {"a.b.c": "val"})
        assert result["a"]["b"]["c"] == "val"

    def test_no_overrides_is_noop(self):
        assert apply_dotted_overrides({"x": 1}, {}) == {"x": 1}


def _write(tmp_path: Path, name: str, content: dict) -> str:
    p = tmp_path / name
    p.write_text(yaml.dump(content))
    return str(p)


class TestMergeYamlChain:
    def test_chain_merges_in_order_with_overrides(self, tmp_path):
        f1 = _write(tmp_path, "base.yaml", {
            "trainer": {"max_epochs": 300, "precision": "16-mixed"},
            "data": {"init_args": {"num_workers": 3}},
        })
        f2 = _write(tmp_path, "stage.yaml", {
            "trainer": {"max_epochs": 100},
            "data": {"init_args": {"batch_size": 64}},
        })
        merged = merge_yaml_chain((f1, f2), {"trainer.max_epochs": "2"})
        assert merged["trainer"]["max_epochs"] == "2"
        assert merged["trainer"]["precision"] == "16-mixed"
        assert merged["data"]["init_args"]["num_workers"] == 3
        assert merged["data"]["init_args"]["batch_size"] == 64
