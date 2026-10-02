from __future__ import annotations

import copy

import pytest

from graphids.exp.temporal_cuda_graph import select_pairs
from graphids.primitives_models import TemporalHybridScanCfg


def trial(rate=100.0, replays=10):
    return {
        "coverage": {
            "duplicate": 0,
            "unexpected": 0,
            "incorrectly_scored_warmup_or_padding": 0,
            "duplicate_successor_targets": 0,
            "unexpected_successor_targets": 0,
            "unique": 100,
            "observed_successor_targets": 100,
            "warmup_events": 10,
        },
        "planned_scored_events": 100,
        "planned_successor_targets": 100,
        "planned_warmup_events": 10,
        "initial_weight_sha256": "same",
        "section_queue_sha256": "same",
        "cuda_device": "V100",
        "cuda_graph": {"replays": replays, "disabled": False},
        "optimizer_steps": 10,
        "end_to_end_scored_events_per_second": rate,
    }


def test_public_graph_flag_defaults_off():
    assert not TemporalHybridScanCfg().cuda_graph
    assert TemporalHybridScanCfg(cuda_graph=True).cuda_graph


def test_paired_acceptance_includes_capture_and_requires_repeatable_gain():
    assert select_pairs(
        [(trial(), trial(104)), (trial(), trial(105)), (trial(), trial(99))]
    )["accepted"]
    assert not select_pairs(
        [(trial(), trial(104)), (trial(), trial(101)), (trial(), trial(99))]
    )["accepted"]
    assert not select_pairs([(trial(), trial(110, replays=0))] * 3)["accepted"]


def test_mismatched_geometry_or_optimizer_counts_are_rejected():
    wrong = copy.deepcopy(trial(110))
    wrong["section_queue_sha256"] = "other"
    with pytest.raises(ValueError, match="section_queue"):
        select_pairs([(trial(), wrong)])
    wrong = trial(110)
    wrong["optimizer_steps"] = 9
    with pytest.raises(ValueError, match="update counts"):
        select_pairs([(trial(), wrong)])
