"""Tests for graphids.campaigns.manifest.

Scope per .claude/rules/test-writing.md: only assertions that guard
custom validator logic or merge semantics — Pydantic's own behaviour
(required fields, Literal rejection, extra='forbid') is not re-tested.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from graphids.campaigns.manifest import (
    Campaign,
    CampaignDefaults,
    Cell,
    cell_statuses,
    load_campaign,
    merged_pipeline_config,
)


# ---------------------------------------------------------------------------
# Custom validators — CONTRACT tests for behaviour Pydantic itself can't
# enforce (uniqueness across a collection, cross-field invariants).
# ---------------------------------------------------------------------------


def test_duplicate_cell_ids_rejected():
    """CONTRACT: Campaign._unique_ids is the only enforcer of cell-id uniqueness."""
    with pytest.raises(ValidationError, match="duplicate cell ids"):
        Campaign(
            name="test",
            created="2026-04-12",
            cells=(Cell(id="c1"), Cell(id="c1")),
        )


# ---------------------------------------------------------------------------
# Merge semantics — the one place manifest has non-trivial logic.
# ---------------------------------------------------------------------------


def test_cell_overrides_defaults_scalar():
    """CONTRACT: cell's non-null field wins over defaults' value."""
    pc = merged_pipeline_config(
        Cell(id="c1", loss_fn="ce"),
        CampaignDefaults(loss_fn="focal", scale="large"),
    )
    assert pc.loss_fn == "ce"
    assert pc.scale == "large"  # inherited from defaults


def test_unset_cell_field_inherits_from_defaults():
    """CONTRACT: exclude_none semantics — unset = inherit, not = null."""
    pc = merged_pipeline_config(Cell(id="c1"), CampaignDefaults(scale="large"))
    assert pc.scale == "large"


def test_unset_everywhere_falls_back_to_pipeline_config_defaults():
    """CONTRACT: when neither cell nor defaults sets a field,
    PipelineConfig's own default applies (loss_fn='focal' per axes.json)."""
    pc = merged_pipeline_config(Cell(id="c1"), CampaignDefaults())
    assert pc.loss_fn == "focal"


def test_tla_overrides_dict_union_not_replace():
    """CONTRACT: tla_overrides from defaults + cell are merged (dict union),
    not scalar-replaced. Prevents every cell having to re-declare shared TLAs."""
    pc = merged_pipeline_config(
        Cell(id="c1", tla_overrides={"b": 2}),
        CampaignDefaults(tla_overrides={"a": 1}),
    )
    assert pc.tla_overrides == {"a": 1, "b": 2}


def test_tla_overrides_cell_wins_on_key_collision():
    """CONTRACT: on tla_overrides key collision, cell wins over defaults."""
    pc = merged_pipeline_config(
        Cell(id="c1", tla_overrides={"a": 99, "b": 2}),
        CampaignDefaults(tla_overrides={"a": 1}),
    )
    assert pc.tla_overrides == {"a": 99, "b": 2}


def test_merged_config_validates_as_pipeline_config():
    """INVARIANT: merge result passes PipelineConfig validation.

    Bogus stage in defaults must raise at CampaignDefaults construction,
    not leak to merge-time. (Guards early-failure — otherwise the user
    wouldn't find out until `campaign next` shells out.)"""
    with pytest.raises(ValidationError):
        CampaignDefaults(stages=["autoencoder", "bogus"])


# ---------------------------------------------------------------------------
# cell_statuses — deriving state from the existing OTel trace log.
# Replaces the entire custom status-log subsystem (280+ LOC deleted).
# ---------------------------------------------------------------------------


def _write_span(path: Path, *, cell_id: str, stage: str, code: str, manifest: str) -> None:
    """Write one OTel-shaped training.fit span line to a traces.jsonl."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    span = {
        "name": "training.fit",
        "attributes": {
            "campaign.manifest": manifest,
            "campaign.cell_id": cell_id,
            "ml.stage": stage,
        },
        "status": {"status_code": code},
    }
    with path.open("a") as fh:
        fh.write(json.dumps(span) + "\n")


@pytest.mark.parametrize(
    ("cell_stages", "recorded", "expected"),
    [
        # all stages OK → completed
        (["autoencoder", "supervised"], [("autoencoder", "OK"), ("supervised", "OK")], "completed"),
        # any stage ERROR → failed (even with later OKs)
        (["autoencoder", "supervised"], [("autoencoder", "ERROR")], "failed"),
        # partial coverage → running
        (["autoencoder", "supervised", "fusion"], [("autoencoder", "OK")], "running"),
    ],
    ids=["all-ok-completed", "any-error-failed", "partial-running"],
)
def test_cell_statuses_derivation(tmp_path: Path, cell_stages, recorded, expected):
    """CONTRACT: completed = all OK; failed = any ERROR; else running."""
    manifest = tmp_path / "c.yaml"
    manifest.write_text("ok")
    campaign = Campaign(
        name="t", created="2026-04-12",
        defaults=CampaignDefaults(stages=cell_stages),
        cells=(Cell(id="c1"),),
    )
    lake = tmp_path / "lake"
    for i, (stage, code) in enumerate(recorded):
        _write_span(
            lake / f"run{i}" / "traces.jsonl",
            cell_id="c1", stage=stage, code=code, manifest=str(manifest.resolve()),
        )
    assert cell_statuses(campaign, manifest_path=manifest, lake_root=lake) == {"c1": expected}


def test_cell_statuses_ignores_other_campaigns(tmp_path: Path):
    """CONTRACT: spans tagged with a different manifest path are ignored.

    Two campaigns sharing a lake_root must not contaminate each other.
    """
    manifest_a = tmp_path / "a.yaml"; manifest_a.write_text("ok")
    manifest_b = tmp_path / "b.yaml"; manifest_b.write_text("ok")
    campaign_a = Campaign(
        name="a", created="2026-04-12",
        defaults=CampaignDefaults(stages=["autoencoder"]),
        cells=(Cell(id="c1"),),
    )
    lake = tmp_path / "lake"
    _write_span(
        lake / "run1" / "traces.jsonl",
        cell_id="c1", stage="autoencoder", code="OK", manifest=str(manifest_b.resolve()),
    )
    assert cell_statuses(campaign_a, manifest_path=manifest_a, lake_root=lake) == {}


def test_cell_statuses_absent_cell_not_in_result(tmp_path: Path):
    """CONTRACT: cells with no spans yet are absent from the result (→ 'pending').

    CLI callers treat missing keys as pending; explicit 'pending' would
    just be noise in the dict.
    """
    manifest = tmp_path / "c.yaml"; manifest.write_text("ok")
    campaign = Campaign(
        name="t", created="2026-04-12",
        defaults=CampaignDefaults(stages=["autoencoder"]),
        cells=(Cell(id="c1"), Cell(id="c2")),
    )
    lake = tmp_path / "lake"
    _write_span(
        lake / "run1" / "traces.jsonl",
        cell_id="c1", stage="autoencoder", code="OK", manifest=str(manifest.resolve()),
    )
    result = cell_statuses(campaign, manifest_path=manifest, lake_root=lake)
    assert result == {"c1": "completed"}
    assert "c2" not in result


# ---------------------------------------------------------------------------
# YAML load — full-stack smoke: pyyaml → pydantic → merged_config.
# Not re-testing pydantic; asserting the schema accepts realistic input.
# ---------------------------------------------------------------------------


def test_load_campaign_round_trip(tmp_path: Path):
    """INVARIANT: a realistic manifest YAML loads and every cell merges cleanly."""
    manifest = textwrap.dedent(
        """
        version: 1
        name: ablation_2026-04
        description: smoke
        created: 2026-04-12
        defaults:
          dataset: hcrl_sa
          seed: 42
          scale: small
          loss_fn: focal
        cells:
          - id: sup-ce-small
            stages: [autoencoder, supervised]
            loss_fn: ce
          - id: sup-focal-large
            scale: large
        """
    )
    path = tmp_path / "ablation.yaml"
    path.write_text(manifest)

    campaign = load_campaign(path)
    assert [c.id for c in campaign.cells] == ["sup-ce-small", "sup-focal-large"]
    assert not campaign.is_frozen

    pc = campaign.merged_config("sup-ce-small")
    assert pc.loss_fn == "ce"
    assert pc.dataset == "hcrl_sa"
    assert pc.stages == ["autoencoder", "supervised"]

    pc2 = campaign.merged_config("sup-focal-large")
    assert pc2.loss_fn == "focal"  # from defaults
    assert pc2.scale == "large"  # cell override


def test_yaml_loader_safe_not_unsafe(tmp_path: Path):
    """REGRESSION: yaml.safe_load (not yaml.load) — reject object tags.

    A manifest that smuggles `!!python/object/apply:...` should not be
    treated as valid YAML input. safe_load raises YAMLError.
    """
    path = tmp_path / "evil.yaml"
    path.write_text("name: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(yaml.YAMLError):
        load_campaign(path)
