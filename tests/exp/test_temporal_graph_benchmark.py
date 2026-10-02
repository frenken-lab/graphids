from __future__ import annotations

import copy

import pytest

from graphids.exp.temporal_cuda_graph import select_pairs
from graphids.primitives_models import TemporalHybridScanCfg


def test_public_graph_flag_defaults_off():
    assert not TemporalHybridScanCfg().cuda_graph
    assert TemporalHybridScanCfg(cuda_graph=True).cuda_graph


def test_paired_acceptance_includes_capture_and_requires_repeatable_gain(trial):
    assert select_pairs(
        [(trial(), trial(104)), (trial(), trial(105)), (trial(), trial(99))]
    )["accepted"]
    assert not select_pairs(
        [(trial(), trial(104)), (trial(), trial(101)), (trial(), trial(99))]
    )["accepted"]
    assert not select_pairs([(trial(), trial(110, replays=0))] * 3)["accepted"]


def test_mismatched_geometry_or_optimizer_counts_are_rejected(trial):
    wrong = copy.deepcopy(trial(110))
    wrong["section_queue_sha256"] = "other"
    with pytest.raises(ValueError, match="section_queue"):
        select_pairs([(trial(), wrong)])
    wrong = trial(110)
    wrong["optimizer_steps"] = 9
    with pytest.raises(ValueError, match="update counts"):
        select_pairs([(trial(), wrong)])
